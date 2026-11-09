# -*- coding: utf-8 -*-
"""本地 Embedding 服务（OpenAI 兼容），给 RAGFlow / Dify 提供 bge-m3 向量化。

用法:
    python embed_server.py --model-path "<模型目录>" --port 8900 [--host 0.0.0.0]
    # 也支持环境变量 MODEL_PATH，便于容器内用 uvicorn/gunicorn 直接启动
    MODEL_PATH=/models/bge-m3 uvicorn embed_server:app --host 0.0.0.0 --port 8900

接口:
    POST /v1/embeddings   {"model": "bge-m3", "input": "文本" | ["文本1", ...]}
    GET  /health          模型未加载时返回 503
"""
import argparse
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 单条文本最大字符数（bge-m3 支持 8192 token，超出会被静默截断 → 入库向量与原文不符）
MAX_CHARS_PER_TEXT = int(os.environ.get('EMBED_MAX_CHARS', '12000'))
MAX_BATCH = int(os.environ.get('EMBED_MAX_BATCH', '64'))
DEVICE = os.environ.get('EMBED_DEVICE', 'cpu')
MODEL_PATH = os.environ.get('MODEL_PATH', '')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('embed_server')

MODEL = None
MODEL_NAME = os.environ.get('EMBED_NAME', 'bge-m3')
LOCK = threading.Lock()


def _load_model(path):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path, device=DEVICE)


@asynccontextmanager
async def lifespan(app):
    """模型加载放到应用生命周期里，避免只在 __main__ 加载导致 uvicorn/gunicorn 方式启动即 500。"""
    global MODEL
    path = MODEL_PATH
    if not path:
        log.warning('未设置 MODEL_PATH，/v1/embeddings 将返回 503（可用 --model-path 或环境变量指定）')
    else:
        log.info('加载 embedding 模型: %s (device=%s)', path, DEVICE)
        MODEL = _load_model(path)
        log.info('模型就绪，维度=%s', MODEL.get_embedding_dimension())
    yield


app = FastAPI(title='Local BGE-M3 Embedding Server', lifespan=lifespan)


class EmbedRequest(BaseModel):
    model: str | None = None
    input: str | list[str]
    encoding_format: str | None = None


def err(msg, code, typ='invalid_request_error'):
    """OpenAI 兼容错误体：{"error": {"message": ..., "type": ...}}"""
    return JSONResponse({'error': {'message': msg, 'type': typ}}, status_code=code)


@app.get('/health')
def health():
    if MODEL is None:
        return JSONResponse({'status': 'loading', 'model': MODEL_NAME, 'dim': None}, status_code=503)
    return {'status': 'ok', 'model': MODEL_NAME, 'dim': int(MODEL.get_embedding_dimension())}


@app.get('/v1/models')
def models():
    return {'object': 'list', 'data': [{'id': MODEL_NAME, 'object': 'model', 'owned_by': 'local'}]}


@app.post('/v1/embeddings')
def embeddings(req: EmbedRequest):
    if MODEL is None:
        return err('模型未加载（检查 MODEL_PATH）', 503, 'service_unavailable')
    if req.encoding_format not in (None, 'float'):
        return err("不支持 encoding_format=%s，仅支持 float" % req.encoding_format, 400)
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    if not texts:
        return err('input 不能为空', 400)
    if len(texts) > MAX_BATCH:
        return err('单次最多 %d 条' % MAX_BATCH, 400)
    too_long = [i for i, t in enumerate(texts) if len(t) > MAX_CHARS_PER_TEXT]
    if too_long:
        # 静默截断会让入库向量与原文不一致，必须让调用方先分块
        return err('第 %s 条输入超过 %d 字符，请先分块' % (too_long[:3], MAX_CHARS_PER_TEXT), 400)

    t0 = time.time()
    with LOCK:      # 本地 CPU 推理串行化，避免并发把内存打爆
        vecs = MODEL.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False)
    data = [{'object': 'embedding', 'index': i, 'embedding': v.tolist()} for i, v in enumerate(vecs)]
    char_count = sum(len(t) for t in texts)
    return {
        'object': 'list',
        'data': data,
        'model': req.model or MODEL_NAME,
        # 注：这是字符数，不是 BPE token 数；OpenAI 语义下 usage 应计 token，这里显式区分
        'usage': {'prompt_tokens': 0, 'total_tokens': 0, 'char_count': char_count},
        'elapsed_ms': int((time.time() - t0) * 1000),
    }


def main():
    global MODEL, MODEL_NAME, MODEL_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', default=os.environ.get('MODEL_PATH', ''))
    ap.add_argument('--port', type=int, default=8900)
    ap.add_argument('--name', default=os.environ.get('EMBED_NAME', 'bge-m3'))
    # 默认只监听本机；容器化时传 --host 0.0.0.0（否则 RAGFlow/Dify 容器访问不到）
    ap.add_argument('--host', default=os.environ.get('HOST', '127.0.0.1'))
    args = ap.parse_args()

    MODEL_PATH = args.model_path
    MODEL_NAME = args.name
    # 允许 --model-path 直接生效：lifespan 会在启动时读取 MODEL_PATH
    os.environ['MODEL_PATH'] = MODEL_PATH
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
