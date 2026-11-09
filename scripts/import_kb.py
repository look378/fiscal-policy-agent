# -*- coding: utf-8 -*-
"""把财政政策知识库导入 RAGFlow（建库 → 上传 → 解析 → 轮询 → 冒烟检索）。

用法（文件路径相对仓库根）:
    python scripts/import_kb.py \
        --key ragflow-fiscal2026abcdef \
        --name 财政政策知识库 \
        --embed bge-m3 \
        --files data/knowledge_base.md "data/policies_clean/*.md"
"""
import argparse
import glob
import json
import os
import sys
import time

import requests

# Windows 控制台默认 GBK，模型/日志里出现非 GBK 字符会直接崩
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ApiError(RuntimeError):
    pass


def api(base, key, method, path, **kw):
    url = base.rstrip('/') + path
    headers = {'Authorization': 'Bearer ' + key}
    timeout = kw.pop('timeout', 120)
    resp = requests.request(method, url, headers=headers, timeout=timeout, **kw)
    try:
        data = resp.json()
    except ValueError:
        data = {'code': -1, 'message': resp.text[:300]}
    return resp.status_code, data


def _ok(data):
    """RAGFlow 的坑：鉴权失败也返回 HTTP 200 + {"code":0,"data":false}，必须同时校验 code 类型与 data。"""
    if not isinstance(data, dict):
        return False
    code = data.get('code')
    if isinstance(code, bool) or not isinstance(code, int) or code != 0:
        return False
    d = data.get('data')
    return isinstance(d, (list, dict)) and d not in (None, False)


def ensure_dataset(base, key, name, embed):
    code, data = api(base, key, 'GET', '/api/v1/datasets', params={'page': 1, 'page_size': 100})
    if code != 200 or not isinstance(data, dict) or data.get('code') != 0:
        raise ApiError('无法访问 RAGFlow（HTTP %s）: %s\n提示：检查 --key 是否有效、--base 是否正确'
                       % (code, json.dumps(data, ensure_ascii=False)[:300]))
    items = data.get('data')
    if isinstance(items, dict):
        items = items.get('datasets') or items.get('items') or []
    for d in (items or []):
        if isinstance(d, dict) and d.get('name') == name:
            print('[=] 已存在知识库: %s (%s)' % (name, d.get('id')))
            return d['id']
    payload = {'name': name, 'embedding_model': embed, 'chunk_method': 'naive', 'permission': 'me'}
    code, data = api(base, key, 'POST', '/api/v1/datasets', json=payload)
    print('[*] 建库返回: HTTP %s code=%s msg=%s' % (code, data.get('code'), data.get('message')))
    if not _ok(data):
        raise ApiError('建库失败: %s' % json.dumps(data, ensure_ascii=False)[:400])
    d = data['data']
    if isinstance(d, dict):
        return d['id']
    if isinstance(d, list) and d and isinstance(d[0], dict):
        return d[0]['id']
    raise ApiError('建库返回结构异常: %s' % json.dumps(data, ensure_ascii=False)[:300])


def existing_names(base, key, dataset_id):
    """已存在文档名；列表接口失败必须抛错（否则会被当成"无文档"而全量重传）。"""
    names, page = set(), 1
    while True:
        code, data = api(base, key, 'GET', '/api/v1/datasets/%s/documents' % dataset_id,
                         params={'page': page, 'page_size': 200})
        if not _ok(data):
            raise ApiError('查询已有文档失败（HTTP %s）: %s' % (code, json.dumps(data, ensure_ascii=False)[:200]))
        raw = data['data']
        docs = raw.get('docs') if isinstance(raw, dict) else raw
        for d in (docs or []):
            if isinstance(d, dict) and d.get('name'):
                names.add(d['name'])
        total = raw.get('total') if isinstance(raw, dict) else None
        if not docs or (isinstance(total, int) and len(names) >= total) or len(docs) < 200:
            break
        page += 1
    return names


def upload(base, key, dataset_id, files):
    """上传文档；已存在的同名文档跳过（幂等）。"""
    have = existing_names(base, key, dataset_id)
    ids, skipped, failed = [], [], []
    for f in files:
        if not os.path.exists(f):
            print('    [!] 跳过不存在: %s' % f)
            continue
        name = os.path.basename(f)
        if name in have:
            skipped.append(name)
            print('    [=] 已存在，跳过: %s' % name)
            continue
        with open(f, 'rb') as fh:
            code, data = api(base, key, 'POST',
                             '/api/v1/datasets/%s/documents' % dataset_id,
                             files={'file': (name, fh, 'text/markdown')}, timeout=300)
        if _ok(data):
            docs = data['data'] if isinstance(data['data'], list) else [data['data']]
            for d in docs:
                if isinstance(d, dict) and d.get('id'):
                    ids.append(d['id'])
                    print('    [+] 上传成功: %s -> %s' % (name, d['id']))
        else:
            failed.append(name)
            print('    [!] 上传失败 %s (HTTP %s): %s' % (name, code, data.get('message')))
    if skipped:
        print('[*] 跳过已存在文档 %d 个' % len(skipped))
    if failed:
        print('[!] 上传失败 %d 个: %s' % (len(failed), '、'.join(failed[:5])))
    return ids, failed


def trigger_parse(base, key, dataset_id, doc_ids):
    code, data = api(base, key, 'POST', '/api/v1/datasets/%s/chunks' % dataset_id,
                     json={'document_ids': doc_ids})
    print('[*] 触发解析: HTTP %s code=%s msg=%s' % (code, data.get('code'), data.get('message')))
    if not _ok(data):
        raise ApiError('触发解析失败: %s' % json.dumps(data, ensure_ascii=False)[:300])


def wait_parsed(base, key, dataset_id, doc_ids, timeout=900):
    """轮询解析状态；返回 (docs, ok)。出现 FAIL 立即返回失败。"""
    want = set(doc_ids)
    t0 = time.time()
    while time.time() - t0 < timeout:
        code, data = api(base, key, 'GET', '/api/v1/datasets/%s/documents' % dataset_id,
                         params={'page': 1, 'page_size': 200})
        raw = data.get('data') if isinstance(data, dict) else None
        docs = (raw.get('docs') if isinstance(raw, dict) else raw) or []
        docs = [d for d in docs if isinstance(d, dict)]
        if want:
            docs = [d for d in docs if d.get('id') in want]
        total = len(docs)
        done = [d for d in docs if d.get('run') == 'DONE' or (d.get('progress') or 0) >= 1.0]
        failed = [d for d in docs if str(d.get('run', '')).upper() in ('FAIL', 'FAILED', 'ERROR')]
        running = [d for d in docs if d.get('run') in ('RUNNING', 'UNSTART')]
        print('    [.] 解析中: %d/%d 完成, %d 进行中, %d 失败 (%.0fs)' % (
            len(done), total, len(running), len(failed), time.time() - t0))
        if failed:
            for d in failed[:3]:
                print('        [x] %s: %s' % (d.get('name'), (d.get('progress_msg') or '')[:100]))
            return docs, False
        if total and len(done) == total:
            return docs, True
        time.sleep(10)
    print('    [!] 解析超时')
    return [], False


def smoke_retrieval(base, key, dataset_id, question):
    payload = {'question': question, 'dataset_ids': [dataset_id], 'page_size': 5}
    code, data = api(base, key, 'POST', '/api/v1/retrieval', json=payload)
    if not _ok(data):
        raise ApiError('冒烟检索失败: %s' % json.dumps(data, ensure_ascii=False)[:300])
    chunks = data['data'].get('chunks') or []
    print('[*] 冒烟检索「%s」: 命中 %d 条' % (question, len(chunks)))
    for c in chunks[:3]:
        print('    - %.3f | %s' % (c.get('similarity') or 0, (c.get('content') or '')[:90].replace('\n', ' ')))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://localhost:9380')
    ap.add_argument('--key', default=os.environ.get('RAGFLOW_KEY', ''))
    ap.add_argument('--name', default='财政政策知识库')
    ap.add_argument('--embed', default='bge-m3')
    ap.add_argument('--files', nargs='+', default=[])
    args = ap.parse_args()

    if not args.key:
        raise SystemExit('缺少 RAGFlow API key：请用 --key 或环境变量 RAGFLOW_KEY 提供')

    files = []
    for f in args.files:
        pattern = f if os.path.isabs(f) else os.path.join(ROOT, f)
        files.extend(glob.glob(pattern))
    files = sorted(set(files))
    print('[*] 待上传 %d 个文件' % len(files))
    if not files:
        raise SystemExit('没有匹配到任何文件（--files 请用相对仓库根的路径或绝对路径）')

    try:
        ds = ensure_dataset(args.base, args.key, args.name, args.embed)
        print('[*] dataset_id = %s' % ds)
        ids, up_failed = upload(args.base, args.key, ds, files)
        if up_failed:
            raise ApiError('有 %d 个文件上传失败（%s），已中止，避免"部分导入"被当成成功'
                           % (len(up_failed), '、'.join(up_failed[:5])))
        if not ids:
            # 全部已存在也算成功，但要有文档
            have = existing_names(args.base, args.key, ds)
            if not have:
                raise ApiError('没有成功上传的文档')
            print('[*] 全部文档已存在，直接进入解析校验')
        else:
            trigger_parse(args.base, args.key, ds, ids)
        docs, ok = wait_parsed(args.base, args.key, ds, ids)
        if not ok:
            raise ApiError('解析失败或超时（参与文档数 %d）' % len(docs))
        print('[*] 解析完成: %d 个文档' % len(docs))
        chunks = smoke_retrieval(args.base, args.key, ds, '中小企业参与政府采购有什么优惠政策？')
        if not chunks:
            raise ApiError('冒烟检索无命中，索引可能异常')
    except ApiError as e:
        raise SystemExit('[FAIL] %s' % e)
    print('\n[DONE] dataset_id=%s' % ds)


if __name__ == '__main__':
    main()
