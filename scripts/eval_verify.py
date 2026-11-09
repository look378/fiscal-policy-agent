# -*- coding: utf-8 -*-
"""评测结果复核：在 LLM 裁判基础上加一层确定性硬校验，避免裁判误判/放水。

复核规则（可预期、可测试）：
1. **正样本**：裁判通过 且（若期望要点含关键数字）回答里出现这些数字（带数字边界，避免
   `200万元` 命中 `1200万元`；区间只要命中任一合法值即可）→ PASS；否则 FAIL。
2. **负样本**：必须同时满足 ①裁判通过 ②无编号引用 ③没有编造具体数字（金额/比例/期限）
   → PASS。任一不满足即 FAIL；**裁判 FAIL 不会被词表翻转成 PASS**。
3. 裁判无法判定（SKIP）→ 结果记 `⚠️ 未判定`，不计入通过率、不进 badcase。
4. 短回答（<20 字）：正样本按"空/过短"判失败；负样本允许（拒答本来就短）。

用法:
    python scripts/eval_verify.py [--src eval/eval_judge.json] [--out eval/eval_final.md] [--min-pass 0.8]
退出码: 0=正常（未启用门禁或达标）；1=低于 --min-pass；2=源数据缺失/为空
"""
import argparse
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:  # noqa: BLE001
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'eval', 'eval_judge.json')
OUT = os.path.join(ROOT, 'eval', 'eval_final.md')

NEG_REASONS = ('不符', '错误地', '答非所问', '未覆盖', '遗漏了', '缺少了', '不一致', '不正确',
               '归类为', '错误的', '未涉及', '未提及', '没有提及', '但未', '并未涉及')

sys.path.insert(0, os.path.join(ROOT, 'serving'))
from refusal_markers import REFUSAL_MARKERS as REFUSAL, has_citation  # noqa: E402

# 负样本里出现这些"具体数字"就视为编造
# 具体数字（金额/比例/期限）；注意排除 2026 年这类"年份"，它不是编造金额
FABRICATION_RE = re.compile(
    r'(?<!\d)\d+(?:\.\d+)?\s*(?:万元|亿元|亿|元|%|％|个?工作日|个月|天)'
    r'|(?<!\d)\d{1,3}\s*年')


def expected_numbers(expect):
    """抽取期望要点里的关键数字（区间整体保留，避免被拆成端点后重复计数）。"""
    ranges = re.findall(r'\d+(?:\.\d+)?%\s*[—\-~至到]\s*\d+(?:\.\d+)?%', expect)
    nums = list(ranges)
    rest = expect
    for r in ranges:
        rest = rest.replace(r, ' ')
    nums += re.findall(r'\d+(?:\.\d+)?%', rest)
    nums += re.findall(r'\d+(?:\.\d+)?\s*万元', rest)
    nums += re.findall(r'\d+\s*个?工作日', rest)
    nums += re.findall(r'\d+\s*年', rest)
    out = []
    for n in nums:
        n = re.sub(r'\s+', '', n)
        if n not in out:
            out.append(n)
    return out


def _num_ok(num, answer):
    """数字是否出现（带边界，避免 200万元 命中 1200万元）。"""
    a = re.sub(r'\s+', '', answer)
    if '%' in num and re.search(r'[—\-~至到]', num):
        # 区间：命中任一合法端点即算出现
        parts = re.findall(r'\d+(?:\.\d+)?', num)
        return any(re.search(r'(?<![\d.])' + re.escape(p) + r'%', a) for p in parts)
    if num.endswith('万元'):
        core = num[:-2]
        return re.search(r'(?<![\d.])' + re.escape(core) + r'万元', a) is not None
    if num.endswith('工作日'):
        core = num[:-3]
        return re.search(r'(?<![\d.])' + re.escape(core) + r'个?工作日', a) is not None
    if num.endswith('年'):
        core = num[:-1]
        return re.search(r'(?<![\d.])' + re.escape(core) + r'年', a) is not None
    return re.search(r'(?<![\d.])' + re.escape(num) + r'(?![\d.])', a) is not None


def has_fabricated_number(answer):
    return FABRICATION_RE.search(answer or '') is not None


def main():
    ap = argparse.ArgumentParser(description='对 LLM 裁判结果做确定性硬校验复核')
    ap.add_argument('--src', default=SRC, help='eval_judge 产出的原始 JSON（应与 --out 报告同名的 .json）')
    ap.add_argument('--out', default=OUT, help='复核报告输出路径（相对仓库根或绝对路径）')
    ap.add_argument('--min-pass', type=float, default=0.0,
                    help='通过率低于该值(0~1)时退出码 1；默认 0 表示不启用门禁')
    args = ap.parse_args()
    if not 0 <= args.min_pass <= 1:
        raise SystemExit('--min-pass 必须在 0~1 之间')

    src = args.src if os.path.isabs(args.src) else os.path.join(ROOT, args.src)
    if not os.path.exists(src):
        raise SystemExit('找不到原始评测数据: %s\n请先运行 eval_judge.py（原始结果保存在与报告同名的 .json）' % src)
    with open(src, encoding='utf-8') as f:
        rows = json.load(f)
    if not rows:
        raise SystemExit('原始评测数据为空: %s' % src)

    final, downgraded = [], []
    for r in rows:
        judge_pass = r.get('pass')
        reason = r.get('judge_reason') or ''
        answer = r.get('answer') or ''
        neg = r.get('negative')
        skipped = r.get('judge_raw_ok') is False

        if skipped:
            final.append(dict(r, final_pass=None, final_reason='未判定（服务异常/裁判无法解析）'))
            continue

        verdict, why = bool(judge_pass), reason

        if neg:
            # 负样本：裁判通过 + 无引用 + 无编造数字，三者缺一不可；裁判 FAIL 不被翻转
            if not judge_pass:
                verdict, why = False, reason or '负样本：裁判未通过'
            elif has_citation(answer):
                verdict, why = False, '负样本：回答带编号引用，不算拒答'
            elif has_fabricated_number(answer):
                verdict, why = False, '负样本：拒答后仍给出具体数字，疑似编造'
            elif not REFUSAL or not any(m in answer for m in REFUSAL):
                verdict, why = False, '负样本：未声明未找到'
            else:
                verdict, why = True, '负样本：明确声明未找到且无编造'
        else:
            if len(answer.strip()) < 20:
                verdict, why = False, '回答为空或过短（%d 字）' % len(answer.strip())
            elif judge_pass:
                nums = expected_numbers(r.get('expect') or '')
                if nums:
                    hit = [n for n in nums if _num_ok(n, answer)]
                    if not hit:
                        verdict = False
                        why = '期望数字一个都没出现（%s）' % '、'.join(nums[:4])
                    elif len(hit) < len(nums):
                        why = (why + '（数字命中 %d/%d）' % (len(hit), len(nums)))[:120]
            # 裁判通过 + 数字校验通过 → PASS

        if judge_pass and not verdict:
            downgraded.append({'no': r.get('no'), 'question': r.get('question'), 'final_reason': why})
        final.append(dict(r, final_pass=verdict, final_reason=why))

    valid = [r for r in final if r['final_pass'] is not None]
    passed = sum(1 for r in valid if r['final_pass'])
    pos = [r for r in valid if not r['negative']]
    negs = [r for r in valid if r['negative']]
    pos_p = sum(1 for r in pos if r['final_pass'])
    neg_p = sum(1 for r in negs if r['final_pass'])
    lats = [r.get('latency_ms', 0) for r in valid]
    skipped_n = len(final) - len(valid)

    def pct(a, b):
        return 100.0 * a / max(1, b)

    L = ['# 财政政策问答 · 评测复核（含数字硬校验）', '',
         '> 生成：`python scripts/eval_judge.py` → `python scripts/eval_verify.py`',
         '> 口径：LLM 裁判 + 确定性复核（区间数字带边界、负样本要求"裁判通过 + 无引用 + 无编造数字"）；'
         '未判定题不计入通过率', '',
         '## 汇总', '', '| 指标 | 数值 |', '|---|---|',
         '| 总通过率 | %d/%d = **%.1f%%** |' % (passed, len(valid), pct(passed, len(valid))),
         '| 正样本通过率 | %d/%d = %.1f%% |' % (pos_p, len(pos), pct(pos_p, len(pos))),
         '| 负样本正确拒答率 | %d/%d = %.1f%% |' % (neg_p, len(negs), pct(neg_p, len(negs))),
         '| 平均延迟 | %.1fs |' % (sum(lats) / max(1, len(lats)) / 1000),
         '| 未判定（SKIP） | %d 条 |' % skipped_n,
         '| 裁判误判（复核后降级） | %d 条 |' % len(downgraded), '',
         '## 逐题结果', '', '| # | 类别 | 问题 | 结果 | 判定说明 | 延迟 |', '|---|---|---|---|---|---|']
    for r in final:
        mark = '✅' if r['final_pass'] is True else ('⚠️' if r['final_pass'] is None else '❌')
        L.append('| %d | %s | %s | %s | %s | %.1fs |' % (
            r['no'], r['category'], r['question'][:28], mark,
            (r['final_reason'] or '')[:70], (r.get('latency_ms') or 0) / 1000))
    L += ['', '## 失败题明细（badcase）', '']
    for r in [x for x in final if x['final_pass'] is False]:
        L.append('### #%d %s' % (r['no'], r['question']))
        L.append('')
        L.append('- **期望**：%s' % (r['expect'] or '')[:200])
        L.append('- **实际**：%s' % (r['answer'] or '（空）').replace('\n', ' ')[:400])
        L.append('- **判定**：%s' % r['final_reason'])
        L.append('')

    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))

    rate = passed / max(1, len(valid))
    print('[汇总] 通过 %d/%d (%.1f%%) | 正 %d/%d | 负 %d/%d | SKIP %d | 降级 %d' % (
        passed, len(valid), pct(passed, len(valid)), pos_p, len(pos), neg_p, len(negs),
        skipped_n, len(downgraded)))
    print('[输出] %s' % out_path)
    for r in downgraded:
        print('  降级: #%s %s -> %s' % (r['no'], (r['question'] or '')[:24], (r['final_reason'] or '')[:60]))
    if args.min_pass and rate < args.min_pass:
        print('[FAIL] 通过率 %.1f%% 低于门禁 %.0f%%' % (rate * 100, args.min_pass * 100))
        sys.exit(1)


if __name__ == '__main__':
    main()
