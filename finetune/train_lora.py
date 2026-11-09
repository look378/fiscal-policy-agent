# -*- coding: utf-8 -*-
"""Qwen2.5 LoRA 微调脚本（bf16，适配 RTX 5060 8G 显存 / 也可 CPU 跑小模型）。

设计说明：
- 基座选择 Qwen2.5-1.5B-Instruct：8G 显存下 bf16 LoRA 可训，且中文政策问答够用；
  3B 需梯度检查点+小 batch 勉强可行，7B 在本机不可行。
- 不用 QLoRA（bitsandbytes 对 Blackwell/sm_120 支持不稳），改 bf16 LoRA。
- 目标不是灌知识，而是**固化输出纪律**：结论先行 + 依据引用 + 无依据拒答。

用法:
    python train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --data ../data/finetune/train.jsonl --out ./output
    python train_lora.py --model <本地模型目录> --cpu   # 无 GPU 时用 CPU 跑 0.5B 试通流程
"""
import argparse
import json
import os
import re
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_system_prompt():
    """与线上 serving/qa_service.py 用同一份 system prompt，避免 train-serve 不一致。"""
    p = os.path.join(ROOT, 'prompts', 'system_prompt.md')
    with open(p, encoding='utf-8-sig') as f:
        raw = f.read()
    cut = re.search(r'^#{1,3}\s*开场白', raw, flags=re.M)
    return (raw[:cut.start()] if cut else raw).strip()


SYSTEM = load_system_prompt()


def build_prompt(tokenizer, sample, max_len=1024):
    """分词并只对答案段计算 loss。

    关键：prompt 与 answer **分别编码再拼接**，避免同一段文本两次 tokenize 在 BPE 边界处
    重新合并导致 mask 偏 1~2 token（会把 prompt 尾巴算进 loss，或把答案首个 token 掩掉）。
    """
    # 优先用 build_dataset.py 产出的 messages（已含检索上下文与线上硬性要求）
    if sample.get('messages'):
        msgs = [{'role': m['role'], 'content': m['content']} for m in sample['messages']
                if m['role'] in ('system', 'user')]
        answer = [m['content'] for m in sample['messages'] if m['role'] == 'assistant'][0]
    else:
        msgs = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': sample['instruction']}]
        answer = sample['output']
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    if len(prompt_ids) >= max_len:
        # 整个 prompt 被截断 → labels 全 -100 → loss=NaN，必须显式拒绝
        raise ValueError('prompt 长度 %d >= max_len %d，调大 --max_len 或缩短上下文'
                         % (len(prompt_ids), max_len))
    answer_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
    eos_ids = tokenizer(tokenizer.eos_token, add_special_tokens=False)['input_ids']
    input_ids = (prompt_ids + answer_ids + eos_ids)[:max_len]
    labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[len(prompt_ids):]
    labels = labels[:len(input_ids)]
    return {
        'input_ids': input_ids,
        'attention_mask': [1] * len(input_ids),
        'labels': labels,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='HF 模型名或本地路径')
    ap.add_argument('--data', default=os.path.join(ROOT, 'data', 'finetune', 'train.jsonl'))
    ap.add_argument('--dev', default=os.path.join(ROOT, 'data', 'finetune', 'dev.jsonl'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'finetune', 'output'))
    ap.add_argument('--epochs', type=float, default=3)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--grad_accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--lora_r', type=int, default=8)
    ap.add_argument('--lora_alpha', type=int, default=16)
    ap.add_argument('--max_len', type=int, default=1024)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available() and not args.cpu
    dtype = torch.bfloat16 if use_cuda else torch.float32
    print('[*] device=%s dtype=%s' % ('cuda' if use_cuda else 'cpu', dtype))

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
        device_map='auto' if use_cuda else None,
    )
    model.config.use_cache = False
    if use_cuda:
        model.gradient_checkpointing_enable()
        # gradient checkpointing 下必须显式要求输入梯度，否则冻结基座 + LoRA 可能整段无梯度
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset('json', data_files={'train': args.data, 'dev': args.dev})
    if len(ds['train']) == 0 or len(ds['dev']) == 0:
        raise SystemExit('训练集或验证集为空，检查 %s / %s' % (args.data, args.dev))
    tok = lambda s: build_prompt(tokenizer, s, args.max_len)
    train_ds = ds['train'].map(tok, remove_columns=ds['train'].column_names)
    dev_ds = ds['dev'].map(tok, remove_columns=ds['dev'].column_names)

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)
    # 注意：transformers 5.x 已移除 warmup_ratio，只能用 warmup_steps
    num_steps = max(1, int(len(train_ds) / max(1, args.batch * args.grad_accum) * args.epochs))
    warmup_steps = max(1, int(num_steps * 0.1))
    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_steps=warmup_steps,
        logging_steps=5,
        save_strategy='epoch',
        eval_strategy='epoch',
        save_total_limit=2,
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        fp16=False,
        # 不用 paged_adamw_8bit：它依赖 bitsandbytes，而本项目设计明确避开 bnb（Blackwell 支持不稳）
        optim='adamw_torch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                      data_collator=collator)
    trainer.train()
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print('[OK] LoRA adapter 已保存到 %s' % args.out)

    # 推理自检
    model.eval()
    sample = json.loads(open(args.dev, encoding='utf-8').readline())
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': sample['instruction']}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=256, do_sample=False)
    print('--- 推理样例 ---')
    print(tokenizer.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True))


if __name__ == '__main__':
    main()
