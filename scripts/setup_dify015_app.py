# -*- coding: utf-8 -*-
"""在 Dify 0.15.3 里建知识库（文件上传方式）、索引文档、建对话应用并绑定知识库。

关键背景（0.15.3 的坑，已实测）：
- `POST /console/api/datasets` 只认 name/description/indexing_technique/provider 等，
  `embedding_model`/`retrieval_model` 会被**静默丢弃** → 建库后必须 `PATCH /datasets/{id}` 再回读断言；
- `POST /console/api/apps` 只认 name/description/mode/icon，`model_config`/`dataset_configs`
  同样被丢弃 → 建应用后必须单独 `POST /apps/{id}/model-config`；
- 列表接口 `GET /apps` 不返回 dataset_configs，校验必须用 `GET /apps/{id}`。

用法: python scripts/setup_dify015_app.py
"""
import argparse
import json
import os
import re
import sys
import time

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
DATASET_NAME = '财政政策知识库'
APP_NAME = '财政政策问答助手'
MODEL_PROVIDER = os.environ.get('DIFY_MODEL_PROVIDER', 'ollama')
MODEL_ID = os.environ.get('DIFY_MODEL', 'deepseek-r1:7b')
EMBED_MODEL = 'bge-m3'
EMBED_PROVIDER = 'openai_api_compatible'
RETRIEVAL = {
    'search_method': 'hybrid_search',
    'reranking_enable': False,
    'reranking_mode': None,
    'reranking_model': {'reranking_provider_name': '', 'reranking_model_name': ''},
    'weights': None,
    'top_k': 4,
    'score_threshold_enabled': False,
    'score_threshold': 0.3,
}


class SetupError(RuntimeError):
    pass


def api(method, path, tok, **kw):
    return requests.request(method, BASE + '/console/api' + path,
                            headers={'Authorization': 'Bearer ' + tok},
                            timeout=kw.pop('timeout', 180), **kw)


def login():
    r = requests.post(BASE + '/console/api/login',
                      json={'email': EMAIL, 'password': PASSWORD, 'remember_me': True}, timeout=30)
    if r.status_code != 200:
        raise SetupError('Dify 登录失败（HTTP %s）: %s' % (r.status_code, r.text[:200]))
    tok = r.json()['data']['access_token']
    with open(TOK_FILE, 'w', encoding='utf-8') as f:
        f.write(tok)
    return tok


def find_dataset(tok):
    r = api('GET', '/datasets?page=1&limit=20', tok)
    if r.status_code != 200:
        raise SetupError('查询知识库失败（HTTP %s）: %s' % (r.status_code, r.text[:200]))
    for d in r.json().get('data') or []:
        if d.get('name') == DATASET_NAME:
            return d['id']
    return None


def ensure_dataset(tok):
    ds = find_dataset(tok)
    if ds:
        print('[*] 已存在知识库: %s (%s)' % (DATASET_NAME, ds))
    else:
        body = {
            'name': DATASET_NAME,
            'description': '财政政策（政府采购/政府购买服务/专项资金/直达资金）问答知识库',
            'indexing_technique': 'high_quality',
            'permission': 'only_me',
            'provider': 'vendor',
        }
        r = api('POST', '/datasets', tok, json=body)
        if r.status_code not in (200, 201):
            raise SetupError('建库失败（HTTP %s）: %s' % (r.status_code, r.text[:300]))
        js = r.json()
        ds = (js.get('data') or js).get('id') or js.get('id')
        if not ds:
            raise SetupError('建库返回缺少 id: %s' % r.text[:300])
        print('[*] 建库成功: %s' % ds)

    # 建库接口会丢弃 embedding/retrieval → PATCH 补上并回读断言
    patch_body = {
        'name': DATASET_NAME,
        'indexing_technique': 'high_quality',
        'embedding_model': EMBED_MODEL,
        'embedding_model_provider': EMBED_PROVIDER,
        'retrieval_model': RETRIEVAL,
    }
    r = api('PATCH', '/datasets/%s' % ds, tok, json=patch_body)
    print('[*] PATCH 检索配置: HTTP %s' % r.status_code)
    if r.status_code not in (200, 201):
        print('[WARN] PATCH 失败（%s），将只做回读校验: %s' % (r.status_code, r.text[:200]))

    detail = api('GET', '/datasets/%s' % ds, tok).json()
    rm = detail.get('retrieval_model_dict') or {}
    actual = {k: rm.get(k) for k in ('search_method', 'top_k')}
    want = {'search_method': RETRIEVAL['search_method'], 'top_k': RETRIEVAL['top_k']}
    print('[*] 回读检索配置: %s（期望 %s）' % (json.dumps(actual, ensure_ascii=False),
                                          json.dumps(want, ensure_ascii=False)))
    if actual.get('search_method') != want['search_method'] or actual.get('top_k') != want['top_k']:
        if args.allow_mismatch:
            print('[WARN] 检索配置未按期望生效（--allow-mismatch 允许继续）')
        else:
            raise SetupError('检索配置未生效：实际 %s / 期望 %s（可加 --allow-mismatch 降级为警告）'
                             % (json.dumps(actual, ensure_ascii=False), json.dumps(want, ensure_ascii=False)))
    if detail.get('embedding_model') != EMBED_MODEL:
        print('[WARN] embedding_model=%s 与期望 %s 不一致（建库后通常不可改，需重建知识库）'
              % (detail.get('embedding_model'), EMBED_MODEL))
    return ds


def upload_file(tok, path):
    with open(path, 'rb') as fh:
        r = requests.post(BASE + '/console/api/files/upload',
                          headers={'Authorization': 'Bearer ' + tok},
                          files={'file': (os.path.basename(path), fh, 'text/markdown')},
                          data={'source': 'datasets'}, timeout=120)
    if r.status_code not in (200, 201):
        return None, 'HTTP %s %s' % (r.status_code, r.text[:150])
    try:
        return r.json().get('id'), ''
    except ValueError:
        return None, '返回非 JSON: %s' % r.text[:150]


def create_documents(tok, dataset_id, file_ids):
    body = {
        'data_source': {'info_list': {'data_source_type': 'upload_file',
                                      'file_info_list': {'file_ids': file_ids}}},
        'indexing_technique': 'high_quality',
        'embedding_model': EMBED_MODEL,
        'embedding_model_provider': EMBED_PROVIDER,
        'process_rule': {'mode': 'automatic'},
        'doc_form': 'text_model',
        'doc_language': 'Chinese',
    }
    r = api('POST', '/datasets/%s/documents' % dataset_id, tok, json=body, timeout=300)
    print('[*] 创建文档: HTTP %s' % r.status_code)
    if r.status_code not in (200, 201):
        raise SetupError('创建文档失败: %s' % r.text[:300])
    data = r.json().get('data')
    docs = data.get('documents') if isinstance(data, dict) else None
    if docs is not None and len(docs) == 0:
        raise SetupError('创建文档返回 0 个文档: %s' % r.text[:300])


def wait_index(tok, dataset_id, timeout=900):
    """返回 (docs, ok)；有 error 文档或超时都算失败。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = api('GET', '/datasets/%s/documents?page=1&limit=100' % dataset_id, tok)
        if r.status_code != 200:
            raise SetupError('查询索引状态失败（HTTP %s）: %s' % (r.status_code, r.text[:200]))
        docs = r.json().get('data') or []
        running = [d for d in docs if d.get('indexing_status') in
                   ('waiting', 'parsing', 'cleaning', 'splitting', 'indexing')]
        done = [d for d in docs if d.get('indexing_status') == 'completed']
        err = [d for d in docs if d.get('indexing_status') == 'error']
        print('    [.] 索引: 完成 %d / 进行 %d / 失败 %d' % (len(done), len(running), len(err)))
        for d in err:
            print('        [x] %s: %s' % (d.get('name'), (d.get('error') or '')[:120]))
        if err:
            return docs, False
        if docs and not running:
            return docs, True
        time.sleep(10)
    print('    [!] 索引超时')
    return [], False


def find_app(tok):
    r = api('GET', '/apps?page=1&limit=50', tok)
    for a in r.json().get('data') or []:
        if a.get('name') == APP_NAME:
            return a['id']
    return None


def build_pre_prompt():
    """与 bind_dify_dataset.py 同一来源：system_prompt.md 正文规则段（不含 UI 文案）。"""
    p = os.path.join(ROOT, 'prompts', 'system_prompt.md')
    with open(p, encoding='utf-8-sig') as f:
        raw = f.read()
    cut = re.search(r'^#{1,3}\s*(开场白|建议问题)', raw, flags=re.M)
    if cut:
        raw = raw[:cut.start()]
    lines = [l for l in raw.splitlines() if not l.strip().startswith('#')]
    return '\n'.join(lines).strip()


def model_config_body(dataset_id):
    return {
        'pre_prompt': build_pre_prompt(),
        'prompt_type': 'simple',
        'chat_prompt_config': {},
        'completion_prompt_config': {},
        'user_input_form': [],
        'dataset_query_variable': '',
        'more_like_this': {'enabled': False},
        'sensitive_word_avoidance': {'enabled': False, 'type': '', 'configs': []},
        'speech_to_text': {'enabled': False},
        'text_to_speech': {'enabled': False, 'voice': '', 'language': ''},
        'suggested_questions': [],
        'suggested_questions_after_answer': {'enabled': False},
        'retriever_resource': {'enabled': True},          # 开启"引用与归属"，否则检索结果不记录
        'annotation_reply': {'enabled': False},
        'agent_mode': {'enabled': False, 'strategy': 'function_call', 'tools': [], 'prompt': None},
        'model': {'provider': MODEL_PROVIDER, 'name': MODEL_ID,
                  'mode': 'chat', 'completion_params': {'temperature': 0.1, 'max_tokens': 2048}},
        'dataset_configs': {
            'retrieval_model': 'multiple',
            'top_k': 4,
            'score_threshold_enabled': False,
            'score_threshold': 0.3,
            'reranking_enable': False,
            'reranking_mode': None,
            'reranking_model': {'reranking_provider_name': '', 'reranking_model_name': ''},
            'weights': None,
            'datasets': {'datasets': [
                # enabled 必须在 dataset 内部（0.15.3 DatasetConfigManager 逐层取 dataset 后再判 enabled）
                {'dataset': {'id': dataset_id, 'name': DATASET_NAME, 'enabled': True}}
            ]},
        },
    }


def ensure_app(tok, dataset_id):
    app_id = find_app(tok)
    if app_id:
        print('[*] 已存在应用: %s (%s)' % (APP_NAME, app_id))
    else:
        body = {'name': APP_NAME, 'mode': 'chat', 'icon': '💰', 'icon_background': '#FFEAD5'}
        r = api('POST', '/apps', tok, json=body)
        if r.status_code not in (200, 201):
            raise SetupError('建应用失败（HTTP %s）: %s' % (r.status_code, r.text[:300]))
        js = r.json()
        app_id = (js.get('app') or js.get('data') or {}).get('id') or js.get('id')
        if not app_id:
            raise SetupError('建应用返回缺少 id: %s' % r.text[:300])
        print('[*] 建应用成功: %s' % app_id)

    # 建应用接口会丢弃 model_config/dataset_configs → 单独调用 model-config
    r = api('POST', '/apps/%s/model-config' % app_id, tok, json=model_config_body(dataset_id))
    if r.status_code not in (200, 201):
        raise SetupError('写应用配置失败（HTTP %s）: %s' % (r.status_code, r.text[:300]))

    # 校验：必须读 /apps/{id}（列表接口不返回 dataset_configs）
    detail = api('GET', '/apps/%s' % app_id, tok).json()
    cfg = detail.get('model_config') or {}
    dcfg = cfg.get('dataset_configs') or {}
    ds_list = ((dcfg.get('datasets') or {}).get('datasets') or [])
    # enabled 在 dataset 内部（0.15.3 的解析逻辑决定）
    enabled = [x for x in ds_list if isinstance(x, dict)
               and (x.get('dataset') or {}).get('enabled')
               and (x.get('dataset') or {}).get('id') == dataset_id]
    if not enabled:
        raise SetupError('应用配置校验失败：dataset_configs 未包含启用的目标知识库\n%s'
                         % json.dumps(dcfg, ensure_ascii=False)[:400])
    if not (cfg.get('retriever_resource') or {}).get('enabled'):
        print('[WARN] retriever_resource 未启用，Dify 将不记录检索引用')
    print('[*] 应用配置校验通过：知识库已启用、引用开关=%s'
          % (cfg.get('retriever_resource') or {}).get('enabled'))
    return app_id


def existing_doc_names(tok, dataset_id):
    r = api('GET', '/datasets/%s/documents?page=1&limit=200' % dataset_id, tok)
    if r.status_code != 200:
        return set()
    return {d.get('name') for d in (r.json().get('data') or []) if isinstance(d, dict)}


def main():
    global BASE, args
    ap = argparse.ArgumentParser(description='在 Dify 0.15.3 建知识库/索引文档/建应用并绑定')
    ap.add_argument('--base', default=BASE, help='Dify 控制台地址')
    ap.add_argument('--allow-mismatch', action='store_true', help='检索配置回读不一致时只告警不失败')
    args = ap.parse_args()
    BASE = args.base.rstrip('/')
    try:
        tok = login()
        ds = ensure_dataset(tok)

        paths = [os.path.join(ROOT, 'data', 'knowledge_base.md')]
        pdir = os.path.join(ROOT, 'data', 'policies_clean')
        if os.path.isdir(pdir):
            paths += [os.path.join(pdir, f) for f in sorted(os.listdir(pdir)) if f.endswith('.md')]
        paths = [p for p in paths if os.path.exists(p)]

        # 幂等：已存在的同名文档跳过，避免重复上传导致文档/分块翻倍
        have = existing_doc_names(tok, ds)
        todo = [p for p in paths if os.path.basename(p) not in have]
        if len(todo) < len(paths):
            print('[*] 跳过已存在文档 %d 个' % (len(paths) - len(todo)))

        file_ids = []
        for p in todo:
            fid, err = upload_file(tok, p)
            print('  [%s] %s %s' % ('OK' if fid else 'FAIL', os.path.basename(p), err))
            if fid:
                file_ids.append(fid)
        if todo and not file_ids:
            raise SetupError('没有成功上传的文件')
        if file_ids:
            create_documents(tok, ds, file_ids)
            docs, ok = wait_index(tok, ds)
            if not ok:
                raise SetupError('索引失败或超时（文档数 %d）' % len(docs))
            print('[*] 索引完成: %d 个文档' % len(docs))
        else:
            print('[*] 无需上传新文档，仅校验已有索引')
            docs, ok = wait_index(tok, ds)
            if not ok:
                raise SetupError('已有文档索引状态异常（文档数 %d）' % len(docs))

        app_id = ensure_app(tok, ds)
        print('[OK] dataset=%s app=%s' % (ds, app_id))
    except SetupError as e:
        raise SystemExit('[FAIL] %s' % e)


if __name__ == '__main__':
    main()
