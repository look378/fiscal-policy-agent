# -*- coding: utf-8 -*-
"""给已有 Dify 应用（重新）绑定知识库 + 设置系统提示词。

背景：0.15.3 的 `POST /apps` 会丢弃 model_config/dataset_configs，必须用 model-config 接口；
而 `GET /apps`（列表）不返回 dataset_configs，校验必须用 `GET /apps/{id}`。

用法: python scripts/bind_dify_dataset.py
退出码: 0=成功且回读校验通过；1=任一环节失败
"""
import argparse
import json
import os
import re
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
DATASET_NAME = '财政政策知识库'
MODEL_PROVIDER = os.environ.get('DIFY_MODEL_PROVIDER', 'ollama')
MODEL_ID = os.environ.get('DIFY_MODEL', 'deepseek-r1:7b')


def build_pre_prompt():
    """只取 system_prompt.md 的正文规则段：去掉标题行与"开场白/建议问题"等 UI 文案段。"""
    p = os.path.join(ROOT, 'prompts', 'system_prompt.md')
    with open(p, encoding='utf-8-sig') as f:
        raw = f.read()
    cut = re.search(r'^#{1,3}\s*(开场白|建议问题)', raw, flags=re.M)
    if cut:
        raw = raw[:cut.start()]
    lines = [l for l in raw.splitlines() if not l.strip().startswith('#')]
    return '\n'.join(lines).strip()


def main():
    ap = argparse.ArgumentParser(description='给 Dify 应用绑定知识库并写入系统提示词')
    ap.add_argument('--base', default=BASE, help='Dify 控制台地址（默认 http://localhost:8080）')
    args = ap.parse_args()
    base = args.base.rstrip('/')

    r = requests.post(base + '/console/api/login',
                      json={'email': EMAIL, 'password': PASSWORD, 'remember_me': True}, timeout=30)
    if r.status_code != 200:
        sys.exit('[FAIL] Dify 登录失败（HTTP %s）: %s' % (r.status_code, r.text[:200]))
    H = {'Authorization': 'Bearer ' + r.json()['data']['access_token']}

    r = requests.get(base + '/console/api/apps?page=1&limit=50', headers=H, timeout=60)
    app = next((a for a in (r.json().get('data') or []) if a.get('name') == APP_NAME), None) \
        if r.status_code == 200 else None
    r = requests.get(base + '/console/api/datasets?page=1&limit=20', headers=H, timeout=60)
    ds = next((d for d in (r.json().get('data') or []) if d.get('name') == DATASET_NAME), None) \
        if r.status_code == 200 else None
    if not app or not ds:
        sys.exit('[FAIL] 应用或知识库不存在（先跑 scripts/setup_dify015_app.py）')
    print('[*] app=%s dataset=%s' % (app['id'], ds['id']))

    pre_prompt = build_pre_prompt()
    pre_prompt += '\n\n知识库内容：\n{{#context#}}\n\n用户问题：{{#query#}}'

    body = {
        'model': {'provider': MODEL_PROVIDER, 'name': MODEL_ID, 'mode': 'chat',
                  'completion_params': {'temperature': 0.1, 'max_tokens': 2048, 'top_p': 0.8}},
        'pre_prompt': pre_prompt,
        'prompt_type': 'simple',
        'dataset_configs': {
            'retrieval_model': 'multiple',
            'top_k': 4,
            'score_threshold_enabled': False,
            'score_threshold': 0.3,
            'reranking_enable': False,
            'reranking_mode': None,
            'reranking_model': {'reranking_provider_name': '', 'reranking_model_name': ''},
            'weights': None,
            # Dify 0.15.3 的 DatasetConfigManager 是 `dataset = item['dataset']` 后再判 `dataset['enabled']`，
            # 所以 enabled 必须写在 **dataset 对象内部**（写在外层会被 continue 跳过 → 检索静默为 0）
            'datasets': {'datasets': [
                {'dataset': {'id': ds['id'], 'name': ds.get('name'), 'enabled': True}}
            ]},
        },
        'dataset_query_variable': '',
        # 引用与归属：不开则 retriever_resources 不记录
        'retriever_resource': {'enabled': True},
        'opening_statement': ('你好，我是财政政策问答助手。政府采购、中小企业扶持、政府购买服务、'
                              '专项资金等问题都可以问我，回答会注明政策名称与条款。'),
        'suggested_questions': [
            '中小企业参与政府采购，预留份额比例是多少？',
            '政府购买服务合同一般能签几年？',
            '专项资金不能用于什么？',
        ],
        'user_input_form': [],
        'file_upload': {'image': {'enabled': False, 'number_limits': 3, 'detail': 'high',
                                  'transfer_methods': ['local_file', 'remote_url']}},
        'sensitive_word_avoidance': {'enabled': False},
        'agent_mode': {'enabled': False, 'strategy': 'function_call', 'tools': [], 'max_iteration': 5},
        'more_like_this': {'enabled': False},
        'speech_to_text': {'enabled': False},
        'text_to_speech': {'enabled': False, 'language': '', 'voice': ''},
        'external_data_tools': [],
        'system_parameters': {'audio_type': 'none', 'file_size_limit': 15,
                              'image_file_size_limit': 10, 'video_file_size_limit': 100},
    }
    r = requests.post(base + '/console/api/apps/%s/model-config' % app['id'], headers=H,
                      json=body, timeout=180)
    print('[*] 更新配置: HTTP %s' % r.status_code)
    if r.status_code not in (200, 201):
        sys.exit('[FAIL] 写应用配置失败: %s' % r.text[:300])

    # 回读校验（必须读详情接口）
    r = requests.get(base + '/console/api/apps/%s' % app['id'], headers=H, timeout=60)
    if r.status_code != 200:
        sys.exit('[FAIL] 读取应用详情失败（HTTP %s）' % r.status_code)
    cfg = (r.json() or {}).get('model_config') or {}
    dcfg = cfg.get('dataset_configs') or {}
    items = ((dcfg.get('datasets') or {}).get('datasets') or [])
    ok_ds = [x for x in items if isinstance(x, dict)
             and (x.get('dataset') or {}).get('enabled')
             and (x.get('dataset') or {}).get('id') == ds['id']]
    ok_ref = bool((cfg.get('retriever_resource') or {}).get('enabled'))
    print('[*] 回读：知识库条目=%d，引用开关=%s' % (len(ok_ds), ok_ref))
    if not ok_ds:
        sys.exit('[FAIL] 回读校验失败：dataset_configs 未包含启用的知识库')
    if not ok_ref:
        print('[WARN] 引用开关未启用，Dify 不会返回 retriever_resources')
    print('[OK] 绑定完成并通过校验')
    print(json.dumps(dcfg, ensure_ascii=False)[:400])


if __name__ == '__main__':
    main()
