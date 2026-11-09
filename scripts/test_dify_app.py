# -*- coding: utf-8 -*-
"""测试 Dify 0.15.3 对话应用：拿 app API Key 调 /v1/chat-messages，断言知识库真的参与检索。

用法: python scripts/test_dify_app.py ["问题"]
退出码: 0=通过；1=调用/登录失败；2=检索引用为 0（知识库未生效）
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
APP_NAME = '财政政策问答助手'


def console_token():
    r = requests.post(BASE + '/console/api/login',
                      json={'email': EMAIL, 'password': PASSWORD, 'remember_me': True}, timeout=30)
    if r.status_code != 200:
        sys.exit('[!] Dify 登录失败（HTTP %s）: %s' % (r.status_code, r.text[:200]))
    return r.json()['data']['access_token']


def get_app(tok):
    r = requests.get(BASE + '/console/api/apps?page=1&limit=50',
                     headers={'Authorization': 'Bearer ' + tok}, timeout=60)
    if r.status_code != 200:
        sys.exit('[!] 查询应用列表失败（HTTP %s）: %s' % (r.status_code, r.text[:200]))
    for a in r.json().get('data') or []:
        if a.get('name') == APP_NAME:
            return a
    return None


def ensure_api_key(tok, app_id):
    r = requests.get(BASE + '/console/api/apps/%s/api-keys' % app_id,
                     headers={'Authorization': 'Bearer ' + tok}, timeout=60)
    keys = (r.json().get('data') or []) if r.status_code == 200 else []
    if keys:
        return keys[0].get('token')
    r = requests.post(BASE + '/console/api/apps/%s/api-keys' % app_id,
                      headers={'Authorization': 'Bearer ' + tok}, json={}, timeout=60)
    if r.status_code in (200, 201):
        js = r.json()
        return js.get('token') or (js.get('data') or {}).get('token')
    print('[!] 创建 API Key 失败:', r.status_code, r.text[:200])
    return None


def ask(app_key, question, user='tester'):
    return requests.post(BASE + '/v1/chat-messages',
                         headers={'Authorization': 'Bearer ' + app_key,
                                  'Content-Type': 'application/json'},
                         json={'inputs': {}, 'query': question, 'response_mode': 'blocking',
                               'user': user, 'conversation_id': ''}, timeout=600)


def main():
    global BASE
    ap = argparse.ArgumentParser(description='测试 Dify 对话应用并断言知识库引用非空')
    ap.add_argument('question', nargs='?', default='中小企业参与政府采购，预留份额比例是多少？')
    ap.add_argument('--base', default=BASE, help='Dify 控制台地址')
    args = ap.parse_args()
    BASE = args.base.rstrip('/')
    question = args.question
    tok = console_token()
    app = get_app(tok)
    if not app:
        sys.exit('[!] 没找到应用（先跑 scripts/setup_dify015_app.py）')
    key = ensure_api_key(tok, app['id'])
    if not key:
        sys.exit('[!] 没拿到 API Key')

    print('[*] app=%s key=%s...' % (app['id'], key[:12]))
    r = ask(key, question)
    if r.status_code != 200:
        print('[!] 调用失败: HTTP %s %s' % (r.status_code, r.text[:300]))
        sys.exit(1)

    d = r.json()
    print('=' * 70)
    print('问：', question)
    print('答：', (d.get('answer') or '')[:900])
    refs = ((d.get('metadata') or {}).get('retriever_resources') or [])
    print('检索引用: %d 条' % len(refs))
    for c in refs[:3]:
        print('  - [%.3f] %s' % (c.get('score') or 0, (c.get('content') or '')[:70].replace('\n', ' ')))

    # 硬断言：引用 0 条说明知识库没真正参与检索（可能是 dataset 未 enabled / 引用开关未开）
    if not refs:
        print('[FAIL] 检索引用为 0 —— 知识库未生效（检查 dataset_configs 中 dataset 条目 enabled 与 retriever_resource）')
        print('       （失败结果不覆盖上次成功的 dify_test_result.json）')
        sys.exit(2)

    os.makedirs(os.path.join(ROOT, 'eval'), exist_ok=True)
    with open(os.path.join(ROOT, 'eval', 'dify_test_result.json'), 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    print('[PASS] Dify 知识库检索生效，引用 %d 条' % len(refs))


if __name__ == '__main__':
    main()
