# -*- coding: utf-8 -*-
"""配置 Dify 0.15.3 模型（customizable-model：先加自定义模型，再填凭据）。

Ollama（chat）: deepseek-r1:7b
OpenAI-API-compatible（embedding）: bge-m3（指向本地 embed_server 8900）

用法: python scripts/setup_dify015_models.py
退出码: 0=chat/embedding 模型均可用；1=配置或校验失败
"""
import argparse
import json
import os
import sys

import requests

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get('DIFY_BASE', 'http://localhost:8080')
EMAIL = os.environ.get('DIFY_EMAIL', 'admin@fiscal.local')
PASSWORD = os.environ.get('DIFY_PASSWORD', 'Fiscal@2026')
if PASSWORD == 'Fiscal@2026' and not os.environ.get('DIFY_PASSWORD'):
    print('[WARN] 使用默认 Dify 口令，仅限本机；生产请设置 DIFY_PASSWORD', file=sys.stderr)
TOK_FILE = os.path.join(ROOT, '.dify015_token')

LLM = os.environ.get('DIFY_MODEL', 'deepseek-r1:7b')
LLM_PROVIDER = os.environ.get('DIFY_MODEL_PROVIDER', 'ollama')
EMBED = 'bge-m3'
EMBED_PROVIDER = 'openai_api_compatible'


class CfgError(RuntimeError):
    pass


def login():
    r = requests.post(BASE + '/console/api/login',
                      json={'email': EMAIL, 'password': PASSWORD, 'remember_me': True}, timeout=30)
    if r.status_code != 200:
        raise CfgError('Dify 登录失败（HTTP %s）: %s' % (r.status_code, r.text[:200]))
    tok = r.json()['data']['access_token']
    with open(TOK_FILE, 'w', encoding='utf-8') as f:
        f.write(tok)
    return tok


def req(method, path, tok, **kw):
    """统一请求；遇到 401 自动重登一次（timeout 复制，避免第一次 pop 后重试回落到默认值）。"""
    timeout = kw.pop('timeout', 120)

    def _do(t):
        return requests.request(method, BASE + '/console/api' + path,
                                headers={'Authorization': 'Bearer ' + t},
                                timeout=timeout, **kw)
    r = _do(tok)
    if r.status_code == 401:
        tok = login()
        r = _do(tok)
    return r


def add_model(tok, provider, model, model_type, credentials):
    body = {'model': model, 'model_type': model_type, 'credentials': credentials}
    r = req('POST', '/workspaces/current/model-providers/%s/models' % provider, tok, json=body)
    print('  [%s/%s] HTTP %s %s' % (provider, model, r.status_code, r.text[:180].replace('\n', ' ')))
    # 409/400 "already exists" 视为成功
    if r.status_code in (200, 201):
        return True
    if r.status_code in (400, 409) and ('exist' in r.text.lower() or 'duplicate' in r.text.lower()):
        return True
    return False


def available(tok, model_type):
    r = req('GET', '/workspaces/current/models/model-types/%s' % model_type, tok)
    if r.status_code != 200:
        raise CfgError('查询可用模型失败（HTTP %s）: %s' % (r.status_code, r.text[:200]))
    return [(m.get('provider'), m.get('model')) for m in (r.json().get('data') or [])]


def main():
    global BASE
    ap = argparse.ArgumentParser(description='配置 Dify 0.15.3 的 chat / embedding 模型')
    ap.add_argument('--base', default=BASE, help='Dify 控制台地址')
    args = ap.parse_args()
    BASE = args.base.rstrip('/')
    try:
        tok = login()
        print('[*] 供应商配置：')
        ok1 = add_model(tok, LLM_PROVIDER, LLM, 'llm',
                        {'base_url': 'http://host.docker.internal:11434',
                         'mode': 'chat', 'context_size': '8192', 'max_tokens': '4096',
                         'vision_support': 'false'})
        ok2 = add_model(tok, EMBED_PROVIDER, EMBED, 'text-embedding',
                        {'endpoint_url': 'http://host.docker.internal:8900/v1',
                         'api_key': 'x', 'context_size': '8192'})
        if not (ok1 and ok2):
            raise CfgError('自定义模型添加失败（见上方 HTTP 状态）')

        # 校验：必须查 model-types 接口（/model-providers 响应里没有 models 字段）
        llms = available(tok, 'llm')
        embeds = available(tok, 'text-embedding')
        print('[*] 可用 llm 模型: %s' % (llms[:6] or '（空）'))
        print('[*] 可用 embedding 模型: %s' % (embeds[:6] or '（空）'))
        # 同时校验 provider + 模型名，避免同名模型来自别的供应商被误判为配置成功
        if (LLM_PROVIDER, LLM) not in llms:
            raise CfgError('校验失败：%s/%s 未出现在可用 llm 列表（%s）' % (LLM_PROVIDER, LLM, llms[:5]))
        if (EMBED_PROVIDER, EMBED) not in embeds:
            raise CfgError('校验失败：%s/%s 未出现在可用 embedding 列表（%s）' % (EMBED_PROVIDER, EMBED, embeds[:5]))
        print('[OK] 模型配置校验通过')
    except CfgError as e:
        raise SystemExit('[FAIL] %s' % e)


if __name__ == '__main__':
    main()
