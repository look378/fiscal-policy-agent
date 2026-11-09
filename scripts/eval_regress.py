# -*- coding: utf-8 -*-
"""回归测试：只跑指定编号的题目（用于验证 badcase 修复效果）。

用法: python scripts/eval_regress.py --nos 6 11 12 38 45 49
退出码: 0=全部通过；1=有失败题；2=服务异常（题目记 SKIP）
"""
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:  # noqa: BLE001
    pass

import requests

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from eval_judge import parse_eval_set, is_negative, judge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8901')
    ap.add_argument('--nos', nargs='+', type=int, required=True)
    args = ap.parse_args()

    rows = {r['no']: r for r in parse_eval_set(os.path.join(ROOT, 'eval', 'eval_set.md'))}
    out, skipped = [], 0
    for no in args.nos:
        row = rows.get(no)
        if not row:
            print('[!] 题号 %d 不存在' % no)
            continue
        try:
            rr = requests.post(args.url + '/api/ask', json={'question': row['question']},
                               timeout=900).json()
        except Exception as e:                      # noqa: BLE001
            print('#%d 服务调用失败: %s' % (no, str(e)[:120]))
            out.append({'no': no, 'question': row['question'], 'answer': '',
                        'pass': None, 'reason': '服务异常: %s' % str(e)[:120], 'latency_ms': None})
            skipped += 1
            continue
        if rr.get('error'):
            # 服务端报错（包括"有兜底文案"的情况）记 SKIP，不能当答错
            print('#%d 服务端错误（%s），记 SKIP' % (no, str(rr.get('error'))[:100]))
            out.append({'no': no, 'question': row['question'], 'answer': rr.get('answer') or '',
                        'pass': None, 'reason': '服务端错误: %s' % str(rr.get('error'))[:100],
                        'latency_ms': rr.get('latency_ms')})
            skipped += 1
            continue
        ans = rr.get('answer') or ''
        neg = is_negative(row)
        pas, reason = judge(row['question'], row['expect'], row['basis'], ans, neg)
        out.append({'no': no, 'question': row['question'], 'answer': ans,
                    'pass': pas, 'reason': reason, 'latency_ms': rr.get('latency_ms')})
        print('=' * 70)
        print('#%d %s' % (no, row['question']))
        print('期望: %s' % row['expect'][:150])
        print('实际: %s' % ans.replace('\n', ' ')[:400])
        verdict = 'PASS' if pas else ('SKIP' if pas is None else 'FAIL')
        if pas is None:
            skipped += 1
        print('判定: %s | %s' % (verdict, reason))

    if not out:
        # 题号全部无效：绝不能打印 PASS 或覆盖上一次结果
        print('[FAIL] 没有任何有效题号，未生成结果')
        sys.exit(1)
    p = os.path.join(ROOT, 'eval', 'regress.json')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('\n[保存] %s' % p)
    failed = [o['no'] for o in out if o['pass'] is False]
    if failed:
        print('[FAIL] 回归未通过: %s' % failed)
        sys.exit(1)
    if skipped:
        print('[SKIP] %d 题未能判定（服务异常）' % skipped)
        sys.exit(2)
    print('[PASS] 回归全部通过')


if __name__ == '__main__':
    main()
