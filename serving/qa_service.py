# -*- coding: utf-8 -*-
"""财政政策问答服务：RAGFlow 检索 + Ollama 生成 + 强制引用 + 无依据拒答 + Prometheus 指标。

接口:
    POST /api/ask                 {"question": "..."} -> {answer, citations, refused, ...}
    POST /v1/chat/completions     OpenAI 兼容（供 Dify 使用，支持 stream=false/true）
    GET  /v1/models               OpenAI 兼容模型列表
    GET  /livez                   存活检查（不访问下游，供 liveness / Docker HEALTHCHECK）
    GET  /health                  就绪检查（依赖不可用返回 503，供 readiness）
    GET  /metrics                 Prometheus 文本指标

环境变量: RAGFLOW_BASE / RAGFLOW_KEY / RAGFLOW_DATASET / OLLAMA_BASE / OLLAMA_MODEL /
          TOP_K / SIM_THRESHOLD / MIN_SIM_TO_ANSWER / MAX_TOKENS / CHUNK_CHARS /
          MAX_CTX_CHARS / NUM_CTX / PROMPT_PATH / HOST / PORT
"""
import argparse
import json
import logging
import os
import re
import sys
import threading
import time
import uuid

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

# 拒答词表与评测脚本共用一份（同目录模块；容器里也会被 COPY 进来）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refusal_markers import STRONG_REFUSAL_MARKERS  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('qa_service')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 上下文长度控制：Ollama 默认 num_ctx 较小，超长会直接返回 400 Bad Request
CHUNK_CHARS = int(os.environ.get('CHUNK_CHARS', '600'))
MAX_CTX_CHARS = int(os.environ.get('MAX_CTX_CHARS', '3000'))
NUM_CTX = int(os.environ.get('NUM_CTX', '8192'))
# 检索最高相似度低于该值时，判定知识库不足以回答 → 直接拒答（防硬答）
MIN_SIM_TO_ANSWER = float(os.environ.get('MIN_SIM_TO_ANSWER', '0.33'))
OLLAMA_TIMEOUT = int(os.environ.get('OLLAMA_TIMEOUT', '600'))

CFG = {
    'ragflow_base': os.environ.get('RAGFLOW_BASE', 'http://localhost:9380'),
    # 注意：默认 key 只用于本机开发；生产必须由环境变量/Secret 注入（缺失时启动告警）
    'ragflow_key': os.environ.get('RAGFLOW_KEY', 'ragflow-fiscal2026abcdef'),
    'dataset': os.environ.get('RAGFLOW_DATASET', '794dc7aed20c11f1a3208fc00a9044d2'),
    # 用 127.0.0.1 而不是 localhost：Windows 上 localhost 会先试 IPv6，实测慢 2 秒
    'ollama_base': os.environ.get('OLLAMA_BASE', 'http://127.0.0.1:11434'),
    'model': os.environ.get('OLLAMA_MODEL', 'deepseek-r1:7b'),
    'top_k': int(os.environ.get('TOP_K', '8')),
    'threshold': float(os.environ.get('SIM_THRESHOLD', '0.2')),
    'max_tokens': int(os.environ.get('MAX_TOKENS', '1536')),
}

METRICS = {
    'requests': 0, 'refused': 0, 'refused_gate': 0, 'refused_text': 0,
    'errors': 0, 'latency_total_ms': 0, 'tokens_total': 0, 'prompt_tokens_total': 0,
    'retrieval_hit_chunks': 0,
}
METRICS_LOCK = threading.Lock()


def _metric_add(**kwargs):
    """线程安全地累加指标（FastAPI 同步端点跑在线程池，裸 += 会丢计数）。"""
    with METRICS_LOCK:
        for k, v in kwargs.items():
            METRICS[k] = METRICS.get(k, 0) + v


def _load_system_prompt():
    """加载纯规则版 system prompt（去掉 md 里给 Dify UI 用的开场白/建议问题段）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get('PROMPT_PATH', ''),
        os.path.join(ROOT, 'prompts', 'system_prompt.md'),
        os.path.join(here, 'prompts', 'system_prompt.md'),
        '/app/prompts/system_prompt.md',
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            with open(p, encoding='utf-8-sig') as f:
                raw = f.read()
            # 截掉 UI 文案段（开头到 "# 开场白" / "## 开场白" 之前）
            cut = re.search(r'^#{1,3}\s*开场白', raw, flags=re.M)
            return (raw[:cut.start()] if cut else raw).strip()
    raise FileNotFoundError('找不到 prompts/system_prompt.md（可用 PROMPT_PATH 指定）')


SYSTEM_PROMPT = _load_system_prompt()

CITE_RE = re.compile(r'[\[【]\s*([0-9\s,，、\-~至到]+?)\s*[\]】]')


def parse_citations(text):
    """解析引用编号：支持 [1] / 【1】 / [1,2] / [1、2] / [1-3] / [1至3]，返回编号集合。"""
    nums = set()
    for grp in CITE_RE.findall(text or ''):
        # 先把区间展开（1-3 → 1,2,3），避免只取到端点
        for a, b in re.findall(r'(\d{1,2})\s*[-~至到]\s*(\d{1,2})', grp):
            lo, hi = int(a), int(b)
            if lo <= hi and hi - lo <= 10 and hi <= 99:
                nums.update(range(lo, hi + 1))
        rest = re.sub(r'\d{1,2}\s*[-~至到]\s*\d{1,2}', ' ', grp)
        for n in re.findall(r'\d{1,2}', rest):
            v = int(n)
            if 1 <= v <= 99:
                nums.add(v)
    return nums
REFUSAL_TEXT = ('抱歉，知识库中没有查到与该问题直接相关的规定。'
                '建议查阅财政部官网、中国政府采购网或咨询当地财政部门。')


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)

    @field_validator('question')
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('question 不能为空白')
        return v


class ChatMessage(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    stream_options: dict | None = None      # OpenAI 兼容：include_usage
    temperature: float | None = None


app = FastAPI(title='财政政策问答服务')


@app.middleware('http')
async def _security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('X-Frame-Options', 'DENY')
    resp.headers.setdefault('Referrer-Policy', 'no-referrer')
    if request.url.path == '/':
        resp.headers['Cache-Control'] = 'no-store'
        resp.headers['Content-Security-Policy'] = (
            "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'")
    return resp


def _transient(exc: Exception) -> bool:
    """只对连接/超时/5xx/429 重试；4xx 与业务错误不重试。"""
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return False


def retrieve(question: str):
    """调用 RAGFlow 检索，返回命中的知识片段（失败重试、状态码校验、JSON 兜底）。"""
    payload = {
        'question': question,
        'dataset_ids': [CFG['dataset']],
        'page_size': CFG['top_k'],
        'similarity_threshold': CFG['threshold'],
        'vector_similarity_weight': 0.3,
    }
    last_err = None
    for attempt in range(1, 4):
        try:
            r = requests.post(CFG['ragflow_base'] + '/api/v1/retrieval',
                              headers={'Authorization': 'Bearer ' + CFG['ragflow_key']},
                              json=payload, timeout=60)
            r.raise_for_status()
            try:
                data = r.json()
            except ValueError:
                raise RuntimeError('RAGFlow 返回非 JSON: %s' % r.text[:120])
            if data.get('code') != 0:
                raise RuntimeError('RAGFlow 检索返回错误: %s' % data.get('message'))
            chunks = (data.get('data') or {}).get('chunks') or []
            if chunks and not any(isinstance(c.get('similarity'), (int, float)) for c in chunks):
                log.warning('RAGFlow chunk 缺少 similarity 字段 → 本次跳过相似度闸门（只按命中数判）')
            return chunks
        except Exception as e:                     # noqa: BLE001
            last_err = e
            if not _transient(e):
                raise
            log.warning('RAGFlow 检索失败（第 %d 次，可重试）: %s', attempt, str(e)[:200])
            if attempt < 3:
                time.sleep(1.5 * attempt)
    raise RuntimeError('RAGFlow 检索失败（已重试 3 次）: %s' % str(last_err)[:200])


def _doc_name(chunk):
    return (chunk.get('document_keyword') or chunk.get('document_name')
            or chunk.get('docnm_kwd') or chunk.get('document_id') or '未知文档')


def _sim(chunk):
    """安全取相似度：非数字/NaN/异常值一律按 None 处理，避免 float('N/A') 抛错。"""
    v = chunk.get('similarity')
    if v is None:
        v = chunk.get('score')
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float('inf'), float('-inf')):   # NaN/Inf
        return None
    return f


def build_prompt(question: str, chunks):
    """拼接上下文：单块截断 + 总长上限（按实际头部长度计预算），返回 (system, user, used)。

    used 是**真正进入 prompt** 的片段，引用列表只能基于它生成。
    """
    ctx_parts, used, total = [], [], 0
    for c in chunks:
        piece = (c.get('content') or '').strip()[:CHUNK_CHARS]
        if not piece:
            continue
        header = '[%d] 来源：%s\n' % (len(used) + 1, _doc_name(c))
        if total + len(header) + len(piece) > MAX_CTX_CHARS:
            break
        used.append(c)
        ctx_parts.append(header + piece)
        total += len(header) + len(piece)
    ctx = '\n\n'.join(ctx_parts)
    user = (
        '知识库内容：\n%s\n\n用户问题：%s\n\n'
        '【硬性要求】\n'
        '1. 只依据上面的知识库内容回答，先给结论，再用 [编号] 标注引用来源；引用必须写明政策名称与条款号。\n'
        '2. **所有数字（比例、金额、天数、年限）必须逐字来自上下文**。上下文里没有的数字，一律不要写，'
        '改为"知识库中未找到该具体比例/金额"。\n'
        '3. **不得把相对时限推算成具体日期**（例如把"30 日内"换算成"次年 1 月 11 日"）。\n'
        '4. 如果上下文只提供了相近条款而没有直接对应的条款，明确说"知识库中未找到直接依据"，'
        '不要用相近条款顶替作答。\n'
        '5. 如果问题问的是"某个具体事项能不能/是否属于某类"，而上下文只给了原则性规定（如"目录由财政部门分级管理"），'
        '必须回答"无法依据现有知识库确定，需查具体目录/当地规定"，不得直接回答"可以"或"不可以"。\n'
        '6. 如果问题涉及地方标准、未收录的政策文件或个案判断，直接说明知识库未收录，'
        '并建议查阅财政部官网/中国政府采购网/当地财政部门。\n'
        '7. 回答控制在 300 字以内。'
    ) % (ctx, question)
    return SYSTEM_PROMPT, user, used


def generate(system: str, user: str, max_tokens=None, temperature=0.1):
    """调用 Ollama；对连接/超时/5xx 做一次重试，返回 (text, raw)。"""
    payload = {
        'model': CFG['model'],
        'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
        'stream': False,
        'options': {'temperature': temperature, 'num_predict': max_tokens or CFG['max_tokens'],
                    'num_ctx': NUM_CTX, 'top_p': 0.8},
    }
    last_err = None
    for attempt in range(1, 3):
        try:
            r = requests.post(CFG['ollama_base'] + '/api/chat', json=payload, timeout=OLLAMA_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if data.get('error'):
                raise RuntimeError('Ollama 返回错误: %s' % str(data.get('error'))[:200])
            return (data.get('message') or {}).get('content', ''), data
        except Exception as e:                     # noqa: BLE001
            last_err = e
            if not _transient(e):
                raise
            log.warning('Ollama 调用失败（第 %d 次，可重试）: %s', attempt, str(e)[:200])
            if attempt < 2:
                time.sleep(2)
    raise RuntimeError('Ollama 调用失败（已重试）: %s' % str(last_err)[:200])


def _citations(used, answer):
    """只返回答案里真正引用、且确实进了 prompt 的片段。

    - 答案没有任何 [n] → 返回空列表（不冒充引用）；
    - 越界编号（如 [9]）被忽略，并标记 citation_verified=False；
    - 只认 ASCII 方括号与全角方括号（[1] / 【1】）：政策文本里的 （1） 多为列表编号，不能当引用。
    """
    cited = parse_citations(answer)
    valid = {n for n in cited if 1 <= n <= len(used)}
    out = [{
        'index': i,
        'document': _doc_name(c),
        'similarity': _sim(c),
        'content': (c.get('content') or '')[:300],
    } for i, c in enumerate(used, 1) if i in valid]
    verified = bool(cited) and cited == valid
    if cited and not verified:
        log.warning('答案引用编号越界: 引用=%s 有效范围=1..%d', sorted(cited), len(used))
    return out, verified


def answer_question(question: str, temperature=0.1):
    """问答主流程：检索 → 闸门 → 拼上下文 → 生成 → 引用/拒答判定。"""
    t0 = time.time()
    _metric_add(requests=1)
    result = {}
    try:
        chunks = retrieve(question)
        sims = [s for s in (_sim(c) for c in chunks) if s is not None]
        top_sim = max(sims) if sims else None
        # 相似度闸门：只在能拿到 similarity 时生效；字段缺失时降级为"仅看是否命中"
        gate_blocked = bool(sims) and top_sim < MIN_SIM_TO_ANSWER
        if not chunks or gate_blocked:
            _metric_add(refused_gate=1, refused=1)
            result = {'answer': REFUSAL_TEXT, 'citations': [], 'citation_verified': False,
                      'refused': True, 'refused_by': 'gate', 'retrieved': len(chunks),
                      'top_similarity': round(top_sim, 4) if top_sim is not None else None}
            return result

        system, user, used = build_prompt(question, chunks)
        if not used:
            _metric_add(refused_gate=1, refused=1)
            result = {'answer': REFUSAL_TEXT, 'citations': [], 'citation_verified': False,
                      'refused': True, 'refused_by': 'gate', 'retrieved': len(chunks),
                      'top_similarity': round(top_sim, 4) if top_sim is not None else None}
            return result
        _metric_add(retrieval_hit_chunks=len(used))

        text, raw = generate(system, user, temperature=temperature)
        prompt_tok = int(raw.get('prompt_eval_count') or 0)
        comp_tok = int(raw.get('eval_count') or 0)
        if not text.strip():                       # 推理模型把预算全花在思考上 → 放大重试一次
            text, raw2 = generate(system, user, max_tokens=max(CFG['max_tokens'] * 2, 3072),
                                  temperature=temperature)
            prompt_tok += int(raw2.get('prompt_eval_count') or 0)
            comp_tok += int(raw2.get('eval_count') or 0)
            raw = raw2
        _metric_add(tokens_total=comp_tok, prompt_tokens_total=prompt_tok)
        if not text.strip():
            _metric_add(errors=1)
            log.error('生成结果为空（question=%s）', question[:60])
            result = {'answer': '抱歉，本次生成超时或失败，未能给出回答。请重试，或查阅财政部官网原始政策文本。',
                      'citations': [], 'citation_verified': False, 'refused': False,
                      'refused_by': None, 'retrieved': len(chunks), 'error': 'empty_generation',
                      'top_similarity': round(top_sim, 4) if top_sim is not None else None}
            return result

        citations, verified = _citations(used, text)
        # 文本拒答：强拒答词出现在开头（<120 字）且没有任何"有效"引用才判定，
        # 避免"正常回答 + 末尾补充未找到"或"只引用了越界编号"被误判成整题拒答
        head = text.strip()[:120]
        refused_text = (not citations) and any(m in head for m in STRONG_REFUSAL_MARKERS)
        if refused_text:
            _metric_add(refused_text=1, refused=1)
        context_chunks = [{
            'index': i, 'document': _doc_name(c),
            'similarity': _sim(c),
            'content': (c.get('content') or '')[:260],
        } for i, c in enumerate(used[:5], 1)]
        context_total = len(used)
        result = {'answer': text.strip(), 'citations': citations,
                  'citation_verified': verified, 'refused': refused_text,
                  'context_chunks': context_chunks, 'context_chunks_total': context_total,
                  'refused_by': 'text' if refused_text else None,
                  'retrieved': len(chunks),
                  'top_similarity': round(top_sim, 4) if top_sim is not None else None,
                  'usage': {'prompt_tokens': prompt_tok, 'completion_tokens': comp_tok,
                            'total_tokens': prompt_tok + comp_tok}}
        return result
    except Exception as e:                         # noqa: BLE001
        _metric_add(errors=1)
        log.exception('问答失败（question=%s）', question[:60])
        result = {'answer': '服务暂时不可用，请稍后重试。若持续失败，请查阅财政部官网原始政策文本。',
                  'citations': [], 'citation_verified': False, 'refused': False,
                  'refused_by': None, 'retrieved': 0, 'top_similarity': None,
                  'error': str(e)[:300]}
        return result
    finally:
        latency = int((time.time() - t0) * 1000)
        _metric_add(latency_total_ms=latency)
        result['latency_ms'] = latency


DEMO_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>财政政策问答智能体 · 在线试用</title>
<style>
  :root{--bg:#0f172a;--card:#1e293b;--card2:#243447;--fg:#e2e8f0;--muted:#94a3b8;
        --accent:#38bdf8;--ok:#34d399;--warn:#fbbf24;--err:#f87171;--line:#334155}
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
       background:linear-gradient(180deg,#0b1220,#0f172a 300px);color:var(--fg);min-height:100vh}
  .wrap{max-width:920px;margin:0 auto;padding:28px 18px 60px}
  h1{font-size:22px;margin:0 0 6px}
  .sub{color:var(--muted);font-size:13px;line-height:1.7}
  .links{margin:14px 0 20px;display:flex;flex-wrap:wrap;gap:8px}
  .links a{color:var(--accent);text-decoration:none;font-size:12.5px;border:1px solid var(--line);
           padding:5px 10px;border-radius:999px;background:#111c30}
  .links a:hover{border-color:var(--accent)}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:14px}
  textarea{width:100%;min-height:74px;resize:vertical;background:#0b1526;color:var(--fg);
           border:1px solid var(--line);border-radius:10px;padding:12px;font-size:15px;line-height:1.6;outline:none}
  textarea:focus{border-color:var(--accent)}
  .row{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}
  button{background:var(--accent);color:#06121f;border:0;border-radius:10px;padding:10px 22px;
         font-size:15px;font-weight:600;cursor:pointer}
  button:disabled{opacity:.55;cursor:not-allowed}
  .hint{color:var(--muted);font-size:12px}
  .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
  .chip{font-size:12.5px;color:#cbd5e1;background:#111c30;border:1px solid var(--line);
        padding:6px 10px;border-radius:999px;cursor:pointer}
  .chip:hover{border-color:var(--accent);color:#fff}
  .meta{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
  .tag{font-size:12px;padding:4px 10px;border-radius:999px;background:#111c30;border:1px solid var(--line);color:var(--muted)}
  .tag.ok{color:#062b1e;background:var(--ok);border-color:var(--ok);font-weight:600}
  .tag.warn{color:#3b2600;background:var(--warn);border-color:var(--warn);font-weight:600}
  .tag.err{color:#3b0a0a;background:var(--err);border-color:var(--err);font-weight:600}
  .answer{white-space:pre-wrap;line-height:1.85;font-size:15px}
  .answer b{color:#fff}
  .cite{background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:12px;margin-top:10px}
  .cite .top{display:flex;justify-content:space-between;gap:10px;font-size:12.5px;color:var(--muted);margin-bottom:6px}
  .cite .doc{color:var(--accent);font-weight:600}
  .cite .txt{font-size:13px;line-height:1.7;color:#cbd5e1}
  .bar{height:4px;background:#0b1526;border-radius:2px;margin:8px 0 4px;overflow:hidden}
  .bar i{display:block;height:100%;background:linear-gradient(90deg,#38bdf8,#34d399)}
  .section-title{font-size:13px;color:var(--muted);margin:0 0 8px;letter-spacing:.5px}
  .hidden{display:none}
  .err{color:var(--err);font-size:13px}
  .foot{margin-top:22px;color:#64748b;font-size:12px;line-height:1.8}
  code{background:#0b1526;border:1px solid var(--line);border-radius:5px;padding:1px 5px;font-size:12px}
</style>
</head>
<body>
<div class="wrap">
  <h1>财政政策问答智能体 · 在线试用</h1>
  <div class="sub">
    RAGFlow 检索 → 相似度闸门 → 本地 deepseek-r1:7b 生成 → 强制引用与拒答。<br>
    回答只依据知识库（59 条政策问答 + 5 份政策原文），没依据会明确说"没有查到"。
  </div>
  <div class="links">
    <a href="/docs" target="_blank" rel="noopener noreferrer">API 文档</a>
    <a href="/metrics" target="_blank" rel="noopener noreferrer">服务指标</a>
    <a href="http://localhost:13000/d/fiscal-qa-overview" target="_blank" rel="noopener noreferrer">Grafana 看板</a>
    <a href="http://localhost:8080/app/9cace4e6-01c1-4337-a53a-2c05d8b5fe78/overview" target="_blank" rel="noopener noreferrer">Dify 应用</a>
    <a href="http://localhost/" target="_blank" rel="noopener noreferrer">RAGFlow 知识库</a>
  </div>

  <div class="card">
    <textarea id="q" placeholder="输入你的财政政策问题，例如：中小企业参与政府采购，预留份额比例是多少？"></textarea>
    <div class="row">
      <button id="go">提问</button>
      <span class="hint" id="hint">Ctrl / ⌘ + Enter 提交 · 单题约 10~25 秒（本地 CPU 推理）</span>
    </div>
    <div class="chips" id="chips"></div>
  </div>

  <div class="card hidden" id="result">
    <div class="meta" id="meta"></div>
    <p class="section-title">回答</p>
    <div class="answer" id="answer"></div>
    <div id="citesWrap">
      <p class="section-title" style="margin-top:16px">检索引用</p>
      <div id="cites"></div>
    </div>
    <p class="err hidden" id="err"></p>
  </div>

  <div class="foot">
    数据来源：data/knowledge_base.md（59 条 QA）与 data/policies_clean/ 下 5 份政策原文。<br>
    指标与告警：Prometheus 抓取 <code>127.0.0.1:8901/metrics</code>；LLM 异常预警见 <code>127.0.0.1:8902/alerts</code>。
  </div>
</div>

<script>
const SAMPLES = [
  '中小企业参与政府采购，预留份额比例是多少？',
  '小微企业价格评审优惠是多少？',
  '政府购买服务合同一般能签几年？',
  '专项资金不能用于什么？',
  '财政部门处理投诉要在多少工作日内办完？',
  '广东省对专精特新企业奖励多少钱？'
];
const $ = s => document.querySelector(s);
let inflight = null, reqSeq = 0;
const chips = $('#chips');
SAMPLES.forEach(t => {
  const d = document.createElement('span');
  d.className = 'chip'; d.textContent = t;
  d.onclick = () => { $('#q').value = t; ask(); };
  chips.appendChild(d);
});

function esc(s){
  return (s==null?'':String(s)).replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
function simText(v){ return (typeof v === 'number' && isFinite(v)) ? v.toFixed(3) : '—'; }
function fmt(s){ return esc(s).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>').replace(/\[(\d+)\]/g,'<b>[$1]</b>'); }

async function ask(){
  const q = $('#q').value.trim();
  if(!q){ $('#q').focus(); return; }
  const btn = $('#go');
  btn.disabled = true; btn.textContent = '思考中…';
  $('#result').classList.remove('hidden');
  $('#err').classList.add('hidden');
  $('#answer').textContent = ''; $('#cites').innerHTML = ''; $('#meta').innerHTML = '';
  $('#hint').textContent = '正在检索并生成，约 10~25 秒…';
  const t0 = performance.now();
  if(inflight){ inflight.abort(); }          // 取消上一次未完成的请求，避免旧答案覆盖新问题
  const ctrl = new AbortController();
  inflight = ctrl;
  const mySeq = ++reqSeq;
  const timer = setTimeout(() => ctrl.abort(), 180000);   // 前端兜底超时（服务端更慢时先放弃等待）
  try{
    const r = await fetch('/api/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question:q}), signal: ctrl.signal
    });
    if(!r.ok){ throw new Error('HTTP '+r.status+' '+(await r.text()).slice(0,160)); }
    const d = await r.json();
    if(mySeq !== reqSeq) return;               // 已有更新的请求，丢弃本次结果
    const secs = ((performance.now()-t0)/1000).toFixed(1);
    const meta = [];
    meta.push(`<span class="tag">耗时 ${secs}s</span>`);
    meta.push(`<span class="tag">检索命中 ${d.retrieved ?? 0} 条</span>`);
    if(d.top_similarity != null) meta.push(`<span class="tag">最高相似度 ${d.top_similarity}</span>`);
    if(d.refused) meta.push(`<span class="tag warn">已拒答（${d.refused_by === 'gate' ? '知识库无相关条文' : '模型判定无依据'}）</span>`);
    else meta.push('<span class="tag ok">已回答</span>');
    if(d.citation_verified === false && (d.citations || []).length) meta.push('<span class="tag">引用编号不完整</span>');
    if(d.error) meta.push(`<span class="tag err">服务异常</span>`);
    $('#meta').innerHTML = meta.join('');
    $('#answer').innerHTML = fmt(d.answer || '（空回答）');
    const cs = d.citations || [];
    if(cs.length){
      $('#cites').innerHTML = cs.map(c => `
        <div class="cite">
          <div class="top">
            <span class="doc">[${c.index}] ${esc(c.document)}</span>
            <span>相似度 ${simText(c.similarity)}</span>
          </div>
          <div class="bar"><i style="width:${Math.max(4, Math.min(100, (Number(c.similarity) || 0) * 100))}%"></i></div>
          <div class="txt">${esc(c.content || '')}</div>
        </div>`).join('');
    } else {
      const ctx = d.context_chunks || [];
      if(ctx.length){
        $('#cites').innerHTML = '<div class="hint" style="margin-bottom:8px">本次回答未标注编号引用；以下为模型实际看到的检索片段（节选 '
          + ctx.length + '/' + (d.context_chunks_total || ctx.length) + ' 条，仅供参考，非回答引用）：</div>'
          + ctx.map(c => `
        <div class="cite">
          <div class="top"><span class="doc">片段 ${c.index}｜${esc(c.document)}</span>
          <span>相似度 ${simText(c.similarity)}</span></div>
          <div class="txt">${esc(c.content || '')}</div>
        </div>`).join('');
      } else {
        $('#cites').innerHTML = '<div class="hint">本次回答没有给出编号引用（模型未标注引用时，这里不会伪造来源）</div>';
      }
    }
  }catch(e){
    const msg = (e && e.name === 'AbortError') ? '请求超时（3 分钟），请稍后重试或检查 /health'
              : ('请求失败：' + (e && e.message ? e.message : e));
    $('#err').textContent = msg;
    $('#err').classList.remove('hidden');
    $('#hint').textContent = '可检查服务依赖：/health';
  }finally{
    clearTimeout(timer);
    if(mySeq === reqSeq){ inflight = null; btn.disabled = false; btn.textContent = '提问'; }
    if(mySeq === reqSeq) $('#hint').textContent = 'Ctrl / ⌘ + Enter 提交 · 单题约 10~25 秒（本地 CPU 推理）';
  }
}
$('#go').onclick = ask;
$('#q').addEventListener('keydown', e => {
  if((e.ctrlKey || e.metaKey) && e.key === 'Enter') ask();
});
</script>
</body>
</html>"""


@app.get('/')
def index():
    """内置试用页面（无需额外前端构建）。"""
    return HTMLResponse(DEMO_HTML)


@app.get('/livez')
def livez():
    """存活检查：只证明进程在跑，不访问下游（liveness / Docker HEALTHCHECK 用它）。"""
    return {'status': 'alive'}


@app.get('/health')
def health():
    """就绪检查：依赖不可用时返回 503（readiness 用它）。"""
    deps = {}
    checks = (
        ('ragflow', CFG['ragflow_base'] + '/v1/system/version', (200, 401)),   # 401 = 服务活着但需鉴权
        ('ollama', CFG['ollama_base'] + '/api/tags', (200,)),
    )
    for name, url, ok_codes in checks:
        try:
            r = requests.get(url, timeout=2)
            deps[name] = r.status_code in ok_codes
        except Exception:                          # noqa: BLE001
            deps[name] = False
    ready = all(deps.values())
    body = {'status': 'ok' if ready else 'degraded', 'model': CFG['model'],
            'dataset': CFG['dataset'], 'dependencies': deps}
    return JSONResponse(body, status_code=200 if ready else 503)


@app.get('/metrics', response_class=PlainTextResponse)
def metrics():
    """Prometheus 文本格式指标（counter 用 rate()/increase() 使用，勿直接看原始值）。"""
    with METRICS_LOCK:
        snap = dict(METRICS)
    m = [
        ('fiscal_qa_requests_total', 'counter', '问答请求总数', snap['requests']),
        ('fiscal_qa_refused_total', 'counter', '拒答次数（闸门 + 文本）', snap['refused']),
        ('fiscal_qa_refused_gate_total', 'counter', '相似度闸门拒答次数', snap['refused_gate']),
        ('fiscal_qa_refused_text_total', 'counter', '模型文本拒答次数', snap['refused_text']),
        ('fiscal_qa_errors_total', 'counter', '问答服务错误数', snap['errors']),
        ('fiscal_qa_latency_ms_total', 'counter', '问答累计耗时(ms)', snap['latency_total_ms']),
        ('fiscal_qa_tokens_total', 'counter', '生成 completion token 累计', snap['tokens_total']),
        ('fiscal_qa_prompt_tokens_total', 'counter', '生成 prompt token 累计', snap['prompt_tokens_total']),
        ('fiscal_qa_retrieval_chunks_total', 'counter', '检索命中并进入上下文的片段数',
         snap['retrieval_hit_chunks']),
    ]
    out = []
    for name, typ, help_, val in m:
        out.append('# HELP %s %s' % (name, help_))
        out.append('# TYPE %s %s' % (name, typ))
        out.append('%s %s' % (name, val))
    return '\n'.join(out) + '\n'


@app.post('/api/ask')
def ask(req: AskRequest):
    return answer_question(req.question)


@app.get('/v1/models')
def list_models():
    return {'object': 'list',
            'data': [{'id': CFG['model'], 'object': 'model', 'owned_by': 'local'}]}


@app.post('/v1/chat/completions')
def chat_completions(req: ChatRequest):
    question = ''
    for m in reversed(req.messages):
        if m.role == 'user':
            question = m.content.strip()
            break
    if not question:
        return JSONResponse({'error': {'message': 'messages 中缺少 user 消息', 'type': 'invalid_request_error'}},
                            status_code=400)
    if len(question) > 500:
        return JSONResponse({'error': {'message': 'user 消息超过 500 字，请缩短后重试',
                                       'type': 'invalid_request_error'}}, status_code=400)

    # 客户端传入的 system/knowledge 提示词不采用：本服务自己编排检索与约束
    result = answer_question(question, temperature=req.temperature if req.temperature is not None else 0.1)
    if result.get('error'):
        return JSONResponse({'error': {'message': result['error'], 'type': 'upstream_error'}},
                            status_code=503)

    resp_id = 'chatcmpl-fiscal-%s' % uuid.uuid4().hex[:12]
    created = int(time.time())
    usage = result.get('usage') or {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    if req.stream:
        def sse():
            chunk = {
                'id': resp_id, 'object': 'chat.completion.chunk', 'created': created,
                'model': req.model or CFG['model'],
                'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': result['answer']},
                             'finish_reason': None}],
            }
            yield 'data: ' + json.dumps(chunk, ensure_ascii=False) + '\n\n'
            done = {
                'id': resp_id, 'object': 'chat.completion.chunk', 'created': created,
                'model': req.model or CFG['model'],
                'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
            }
            if not (req.stream_options or {}).get('include_usage'):
                done['usage'] = usage
            yield 'data: ' + json.dumps(done, ensure_ascii=False) + '\n\n'
            if (req.stream_options or {}).get('include_usage'):
                yield 'data: ' + json.dumps({
                    'id': resp_id, 'object': 'chat.completion.chunk', 'created': created,
                    'model': req.model or CFG['model'], 'choices': [], 'usage': usage,
                }, ensure_ascii=False) + '\n\n'
            yield 'data: [DONE]\n\n'
        return StreamingResponse(sse(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    return {
        'id': resp_id,
        'object': 'chat.completion',
        'created': created,
        'model': req.model or CFG['model'],
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': result['answer']},
                     'finish_reason': 'stop'}],
        'usage': usage,
        # 扩展字段：引用与拒答信息（OpenAI 客户端会忽略未知字段）
        'citations': result.get('citations') or [],
        'refused': result.get('refused'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8901')))
    # 默认只监听本机；容器/K8s 部署时传 --host 0.0.0.0（否则探针与 Prometheus 均不可达）
    ap.add_argument('--host', default=os.environ.get('HOST', '127.0.0.1'))
    args = ap.parse_args()
    if CFG['ragflow_key'] == 'ragflow-fiscal2026abcdef':
        log.warning('正在使用内置默认 RAGFLOW_KEY，生产环境请用环境变量/Secret 注入')
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    main()
