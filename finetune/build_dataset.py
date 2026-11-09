# -*- coding: utf-8 -*-
"""从知识库构造 LoRA 微调数据集（**与线上 qa_service 的 prompt 结构对齐**）。

为什么这样构造（train-serve 一致）：
- 线上 `qa_service.build_prompt()` 的 user 消息是「知识库内容（带 [n] 来源）+ 用户问题 + 硬性要求」，
  答案要求先结论、后依据、用 [n] 标注引用、无依据要拒答；
- 如果训练样本只有「问题→答案」，模型学到的是"没有上下文也直接作答"，与线上冲突（会忽略检索结果）。
- 所以这里每条样本都带上**检索上下文片段**与线上同款约束，答案统一带 `[1]` 引用；
  拒答样本带"（无）"上下文或无关片段，答案明确说未查到、且**不带引用**——正是服务端拒答判定要的行为。

用法: python finetune/build_dataset.py [--out_dir ../data/finetune]
输出: train.jsonl / dev.jsonl / stats.json
"""
import argparse
import json
import os
import random
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_PATH = os.path.join(ROOT, 'prompts', 'system_prompt.md')

# 库外问题（训练"拒答纪律"）。注意：**不要**与 eval/eval_set.md 负样本逐字相同，
# 否则训练前后跑同一套评测会被污染（评测时应报告"已见/未见"两组）。
REFUSAL_QUESTIONS = [
    '广东省对专精特新中小企业的奖补上限是多少？',
    '江苏省政府采购分散采购限额标准具体是多少万元？',
    '2026年中央对地方转移支付总规模预计多少亿元？',
    '我们公司注册资本500万，能不能直接中标？',
    '今年新增地方政府专项债券额度是多少？',
    '供应商对采购文件提出质疑的答复期限是几个工作日？',
    '建筑业中小企业划型标准的营业收入门槛是多少？',
    '单一来源采购需要哪些专家论证材料？',
    '我们市今年直达资金分配到多少亿元？',
    '政府购买服务指导性目录里是否包含养老服务？',
]

REFUSAL_ANSWER = ('**结论**：知识库中没有查到相关规定，无法给出具体数值或结论。\n'
                  '**依据**：当前知识库仅收录 5 份财政政策原文与配套问答，未包含该地方标准/该文件/该口径。\n'
                  '**提示**：建议查阅财政部官网、中国政府采购网或咨询当地财政部门，以当地最新规定为准。')

# 与线上 qa_service.build_prompt 中的硬性要求保持一致（改了这里也要改服务端）
HARD_RULES = (
    '【硬性要求】\n'
    '1. 只依据上面的知识库内容回答，先给结论，再用 [编号] 标注引用来源；引用必须写明政策名称与条款号。\n'
    '2. **所有数字（比例、金额、天数、年限）必须逐字来自上下文**。上下文里没有的数字，一律不要写，'
    '改为"知识库中未找到该具体比例/金额"。\n'
    '3. **不得把相对时限推算成具体日期**。\n'
    '4. 如果上下文只提供了相近条款而没有直接对应的条款，明确说"知识库中未找到直接依据"，'
    '不要用相近条款顶替作答。\n'
    '5. 如果问题问的是"某个具体事项能不能/是否属于某类"，而上下文只给了原则性规定，'
    '必须回答"无法依据现有知识库确定，需查具体目录/当地规定"。\n'
    '6. 如果问题涉及地方标准、未收录的政策文件或个案判断，直接说明知识库未收录，'
    '并建议查阅财政部官网/中国政府采购网/当地财政部门。\n'
    '7. 回答控制在 300 字以内。'
)


def load_system_prompt():
    """与线上共用同一份 system prompt（去掉 md 里给 Dify UI 的开场白段）。"""
    with open(PROMPT_PATH, encoding='utf-8-sig') as f:
        raw = f.read()
    cut = re.search(r'^#{1,3}\s*开场白', raw, flags=re.M)
    return (raw[:cut.start()] if cut else raw).strip()


SYSTEM_PROMPT = load_system_prompt()


def parse_kb(path):
    """解析 knowledge_base.md 的 QA 条目（多行标签用累加，避免换行被截断）。"""
    with open(path, encoding='utf-8-sig') as f:
        text = f.read()
    blocks = re.split(r'\n### Q：', text)
    items = []
    for b in blocks[1:]:
        lines = b.splitlines()
        if not lines:
            continue
        question = lines[0].strip().rstrip('？') .strip()
        if not question:
            continue
        question += '？'
        fields = {'结论': [], '依据': [], '提示': []}
        cur = None
        for ln in lines[1:]:
            ln = ln.strip()
            m = re.match(r'^\*\*(结论|依据|提示)\*\*[：:]\s*(.*)$', ln)
            if m:
                cur = m.group(1)
                fields[cur].append(m.group(2).strip())
            elif cur and ln:
                fields[cur].append(ln)       # 续行
        conclusion = ' '.join(fields['结论']).strip()
        basis = ' '.join(fields['依据']).strip()
        tip = ' '.join(fields['提示']).strip()
        if conclusion and basis:
            items.append({'question': question, 'conclusion': conclusion,
                          'basis': basis, 'tip': tip})
    return items


def to_sample(question, context, answer):
    """构造与线上一致的 (system, user, assistant) 三段式样本。"""
    user = '知识库内容：\n%s\n\n用户问题：%s\n\n%s' % (context, question, HARD_RULES)
    return {'messages': [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user},
        {'role': 'assistant', 'content': answer},
    ], 'instruction': question, 'output': answer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default=os.path.join(ROOT, 'data', 'finetune'))
    ap.add_argument('--dev_ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if not 0 < args.dev_ratio < 0.5:
        raise SystemExit('--dev_ratio 必须在 (0, 0.5) 之间')
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    items = parse_kb(os.path.join(ROOT, 'data', 'knowledge_base.md'))
    print('[*] 知识库解析到 %d 条 QA' % len(items))
    if len(items) < 50:
        raise SystemExit('知识库解析条数异常（%d < 50），检查 knowledge_base.md 的 "### Q：" 结构' % len(items))

    samples = []
    for it in items:
        # 上下文 = 该条的「依据」原文（模拟检索到的条款片段），并带 [1] 来源头
        context = '[1] 来源：%s\n%s' % ('财政政策知识库', it['basis'])
        answer = '**结论**：%s\n**依据**：%s [1]' % (it['conclusion'], it['basis'])
        if it['tip']:
            answer += '\n**提示**：%s' % it['tip']
        samples.append(to_sample(it['question'], context, answer))

    # 拒答样本：上下文为"（无）"或无关片段，答案不含 [n]
    for q in REFUSAL_QUESTIONS:
        samples.append(to_sample(q, '（无）', REFUSAL_ANSWER))

    random.shuffle(samples)
    n_dev = max(1, int(len(samples) * args.dev_ratio))
    n_dev = min(n_dev, len(samples) - 1)
    dev, train = samples[:n_dev], samples[n_dev:]
    if not train or not dev:
        raise SystemExit('切分后训练集或验证集为空（total=%d）' % len(samples))

    for name, part in (('train.jsonl', train), ('dev.jsonl', dev)):
        with open(os.path.join(args.out_dir, name), 'w', encoding='utf-8') as f:
            for s in part:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')

    cites = sum(1 for s in samples if '[1]' in s['output'] and '未包含' not in s['output'])
    stats = {
        'kb_qa': len(items),
        'refusal': len(REFUSAL_QUESTIONS),
        'total': len(samples),
        'train': len(train),
        'dev': len(dev),
        'with_citation': cites,
        'avg_output_chars': round(sum(len(s['output']) for s in samples) / len(samples), 1),
        'avg_prompt_chars': round(sum(len(s['messages'][1]['content']) for s in samples) / len(samples), 1),
    }
    with open(os.path.join(args.out_dir, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print('[OK] 训练 %d 条 / 验证 %d 条 -> %s' % (len(train), len(dev), args.out_dir))
    print('     带引用样本 %d 条，平均输出 %.0f 字，平均 prompt %.0f 字'
          % (cites, stats['avg_output_chars'], stats['avg_prompt_chars']))


if __name__ == '__main__':
    main()
