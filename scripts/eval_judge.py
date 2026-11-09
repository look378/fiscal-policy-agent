# -*- coding: utf-8 -*-
"""LLM-as-judge 评测：调用问答服务拿到回答，再用本地模型按「期望要点」打分。

相比关键词匹配，LLM 裁判能识别"意思对但措辞不同"，避免把正确答案误判为失败。

用法: python eval_judge.py [--url http://127.0.0.1:8901] [--judge-model deepseek-r1:7b] [--limit 49]
"""
import argparse
import json
import os
import re
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:  # noqa: BLE001
    pass

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLLAMA = os.environ.get('OLLAMA_BASE', 'http://localhost:11434')

JUDGE_SYS = ('你是一个严格的政策问答评测员。给你【问题】【期望回答要点】【模型实际回答】，'
             '判断模型回答是否在核心事实上符合期望要点。')
JUDGE_TPL = '''【问题】{q}
【期望回答要点】{exp}
【依据条款】{basis}
【模型实际回答】{ans}

评判标准：
1. 只看核心要点是否覆盖，不要求措辞一致；引用条款号形式不同也算对。
2. 若模型回答了核心数字/结论且方向正确 → 通过。
3. 若模型答非所问、关键数字错误、或明显编造 → 不通过。
4. 若模型说"知识库没有查到"而期望要点是要求回答 → 不通过。

只输出两行，格式严格如下：
结论：通过
理由：（不超过40字）'''

# 裁判模型（默认与线上生成模型一致；可用 --judge-model 或环境变量覆盖）
JUDGE_MODEL = os.environ.get('JUDGE_MODEL', os.environ.get('OLLAMA_MODEL', 'deepseek-r1:7b'))

# 统一拒答词表：与 serving/refusal_markers.py 共用一份，避免多处定义分叉
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'serving'))
from refusal_markers import REFUSAL_MARKERS as REFUSAL, has_citation  # noqa: E402


def parse_eval_set(path):
    rows = []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) < 5 or not re.match(r'^\d+$', cells[0]):
            continue
        rows.append({'no': int(cells[0]), 'category': cells[1], 'question': cells[2],
                     'expect': cells[3], 'basis': cells[4]})
    return rows


def is_negative(row):
    """负样本判定：优先看 category（评测集里用「负样本·xxx」标注），文本特征作为兜底。"""
    if '负样本' in row.get('category', ''):
        return True
    text = row['expect'] + row['basis'] + row['category']
    return ('拒答' in text or '不确定' in text or '无法回答' in text)


def judge(q, exp, basis, ans, negative):
    ans = (ans or '').strip()
    # 负样本先判定：短拒答（如"知识库没有收录该内容。"）也要算通过，不能先被长度门槛拦掉
    if negative:
        if not ans:
            return False, '负样本：回答为空'
        hit = [m for m in REFUSAL if m in ans]
        if hit and not has_citation(ans):
            return True, '负样本：正确声明未找到（命中" %s "）' % hit[0]
        return False, '负样本：未声明未找到，疑似硬答'
    # 正样本：空/极短回答直接判失败，不浪费裁判调用
    if len(ans) < 20:
        return False, '回答为空或过短（%d 字），直接判失败' % len(ans)
    user = JUDGE_TPL.format(q=q, exp=exp, basis=basis, ans=ans)
    payload = {'model': JUDGE_MODEL, 'messages': [
        {'role': 'system', 'content': JUDGE_SYS},
        {'role': 'user', 'content': user}],
        'stream': False, 'options': {'temperature': 0.0, 'num_predict': 700}}
    try:
        r = requests.post(OLLAMA + '/api/chat', json=payload, timeout=600)
        txt = r.json().get('message', {}).get('content', '')
    except Exception as e:
        return None, 'judge 调用失败: %s' % str(e)[:80]
    try:
        r.raise_for_status()
    except Exception as e:                          # noqa: BLE001
        return None, 'judge HTTP 失败: %s' % str(e)[:80]
    m = re.search(r'结论\s*[：:]\s*(未通过|不能通过|不通过|通过|是|否|符合|不符合)', txt)
    if not m:
        return None, 'judge 输出无法解析: %s' % txt[:80].replace('\n', ' ')
    reason = ''
    m2 = re.search(r'理由\s*[：:]\s*(.+)', txt)
    if m2:
        reason = m2.group(1).strip()[:60]
    passed = m.group(1) in ('通过', '是', '符合')
    # 裁判自相矛盾保护：结论说过、理由却指出错 → 以理由为准判失败
    NEG = ('不符', '错误地', '答非所问', '未覆盖', '遗漏了', '缺少了', '不一致', '不正确', '归类为', '错误的')
    # 否定语境识别：'没有不符合之处' 这类双重否定不算否定
    def _negated(w):
        return re.search(r'(没有|无|不存在|未发现|未见)[^。；]{0,6}' + re.escape(w), reason) is not None
    if passed and any(w in reason and not _negated(w) for w in NEG):
        passed = False
        reason = '（裁判理由否定，已纠正）' + reason
    return passed, reason


def main():
    global JUDGE_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8901')
    ap.add_argument('--limit', type=int, default=999)
    ap.add_argument('--judge-model', default=JUDGE_MODEL, help='裁判模型名（默认与 OLLAMA_MODEL 一致）')
    ap.add_argument('--out', default=os.path.join(ROOT, 'eval', 'eval_judge.md'))
    ap.add_argument('--min-pass', type=float, default=0.0,
                    help='通过率低于该值(0~1)时退出码 1，用于 CI 门禁（默认 0 表示不启用）')
    args = ap.parse_args()
    if not 0 <= args.min_pass <= 1:
        raise SystemExit('--min-pass 必须在 0~1 之间')
    if args.limit < 1:
        raise SystemExit('--limit 必须 >= 1')
    JUDGE_MODEL = args.judge_model
    print('[*] 裁判模型: %s' % JUDGE_MODEL, flush=True)

    rows = parse_eval_set(os.path.join(ROOT, 'eval', 'eval_set.md'))[:args.limit]
    print('[*] 载入 %d 题' % len(rows), flush=True)
    results, t0 = [], time.time()
    for i, row in enumerate(rows, 1):
        st = time.time()
        service_err = None
        try:
            d = requests.post(args.url + '/api/ask', json={'question': row['question']}, timeout=900).json()
        except Exception as e:
            service_err = str(e)[:120]
            d = {'answer': '', 'refused': True, 'latency_ms': 0, 'error': service_err}
        neg = is_negative(row)
        if d.get('error'):
            # 服务端只要报 error（包括"有兜底文案"的情况）都记 SKIP，不能当答对/答错
            pas, reason = None, '服务异常（%s），本题记 SKIP' % str(d.get('error'))[:120]
        else:
            pas, reason = judge(row['question'], row['expect'], row['basis'], d.get('answer') or '', neg)
        row.update({'answer': d.get('answer') or '', 'citations': d.get('citations') or [],
                    'refused': d.get('refused'), 'negative': neg,
                    'latency_ms': d.get('latency_ms') or int((time.time() - st) * 1000),
                    'pass': bool(pas), 'judge_reason': reason, 'judge_raw_ok': pas is not None})
        results.append(row)
        print('[%02d/%d] %s | %s | %s' % (i, len(rows), 'PASS' if pas else ('SKIP' if pas is None else 'FAIL'),
                                          row['latency_ms'] / 1000, row['question'][:26]), flush=True)

    valid = [r for r in results if r['judge_raw_ok']]
    passed = sum(1 for r in valid if r['pass'])
    pos = [r for r in valid if not r['negative']]
    negs = [r for r in valid if r['negative']]
    pos_p = sum(1 for r in pos if r['pass'])
    neg_p = sum(1 for r in negs if r['pass'])
    lats = sorted(r['latency_ms'] for r in valid)
    import math
    p95 = lats[max(0, math.ceil(len(lats) * 0.95) - 1)] if lats else 0

    L = ['# 财政政策问答 · 评测结果（LLM 裁判）', '',
         '> 时间：%s ｜ 生成模型：%s（本地）｜ 裁判：%s' % (
             time.strftime('%Y-%m-%d %H:%M'), os.environ.get('OLLAMA_MODEL', 'deepseek-r1:7b'), JUDGE_MODEL),
         '> 检索：RAGFlow ｜ 题目：%d（负样本 %d，SKIP %d）' % (
             len(valid), len(negs), len(results) - len(valid)), '',
         '## 汇总', '', '| 指标 | 数值 |', '|---|---|',
         '| 总通过率 | %d/%d = **%.1f%%** |' % (passed, len(valid), 100.0 * passed / max(1, len(valid))),
         '| 正样本通过率 | %d/%d = %.1f%% |' % (pos_p, len(pos), 100.0 * pos_p / max(1, len(pos))),
         '| 负样本正确拒答 | %d/%d = %.1f%% |' % (neg_p, len(negs), 100.0 * neg_p / max(1, len(negs))),
         '| 平均延迟 | %.1fs |' % (sum(lats) / max(1, len(lats)) / 1000),
         '| P95 延迟 | %.1fs |' % (p95 / 1000), '',
         '## 逐题结果', '', '| # | 类别 | 问题 | 结果 | 裁判理由 | 延迟 |', '|---|---|---|---|---|---|']
    for r in results:
        L.append('| %d | %s | %s | %s | %s | %.1fs |' % (
            r['no'], r['category'], r['question'][:28],
            '✅' if r['pass'] else ('⚠️' if not r['judge_raw_ok'] else '❌'),
            r['judge_reason'] or ('（未判定）' if not r['judge_raw_ok'] else '（无理由）'),
            r['latency_ms'] / 1000))
    L += ['', '## 失败题原始回答', '']
    for r in results:
        if not r['pass']:
            L.append('### %d. %s' % (r['no'], r['question']))
            L.append('')
            L.append('- 期望：%s' % r['expect'][:200])
            L.append('- 实际：%s' % (r['answer'] or '（空）').replace('\n', ' ')[:400])
            L.append('- 理由：%s' % r['judge_reason'])
            L.append('')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    # 原始结果写到与报告同名的 .json（如 eval_v4.md -> eval_v4.json），
    # 避免固定文件名 eval_judge.json 被后续小规模试跑（如 --limit 1）覆盖导致历史证据丢失
    json_path = os.path.splitext(args.out)[0] + '.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print('[报告] %s' % args.out)
    # 有裁判无法判定的题（SKIP）视为评测不完整，返回非零便于 CI 发现
    skipped = len(results) - len(valid)
    rate = passed / max(1, len(valid))
    if args.min_pass and rate < args.min_pass:
        print('[FAIL] 通过率 %.1f%% 低于门禁 %.0f%%' % (rate * 100, args.min_pass * 100))
        sys.exit(1)
    if skipped:
        print('[WARN] %d 题裁判无法判定，评测不完整' % skipped)
        sys.exit(1)
    print('\n[汇总] 通过 %d/%d (%.1f%%) | 正 %d/%d | 负 %d/%d | 平均 %.1fs' % (
        passed, len(valid), 100.0 * passed / max(1, len(valid)), pos_p, len(pos), neg_p, len(negs),
        sum(lats) / max(1, len(lats)) / 1000))
    print('[输出] %s' % args.out)


if __name__ == '__main__':
    main()
