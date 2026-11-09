# -*- coding: utf-8 -*-
"""把 LoRA adapter 合并回基座并导出，供 Ollama / vLLM 使用。

用法:
    python finetune/merge_export.py --base Qwen/Qwen2.5-1.5B-Instruct \
        --adapter finetune/output --out finetune/merged
    # 然后转 GGUF 交给 Ollama（需 llama.cpp）:
    #   python llama.cpp/convert_hf_to_gguf.py finetune/merged --outfile fiscal-qwen.gguf
    #   ollama create fiscal-qwen -f Modelfile    # Modelfile: FROM ./fiscal-qwen.gguf

注意：本机未实跑训练（torch 为 CPU 版），此脚本未做端到端验证。
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='基座模型名或本地路径')
    ap.add_argument('--adapter', required=True, help='train_lora.py 输出的 adapter 目录')
    ap.add_argument('--out', required=True, help='合并后模型输出目录')
    args = ap.parse_args()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        sys.exit('缺少依赖（需 pip install torch transformers peft accelerate）: %s' % e)

    if not os.path.isdir(args.adapter):
        sys.exit('adapter 目录不存在: %s（先跑 train_lora.py）' % args.adapter)

    print('[*] 加载基座: %s' % args.base)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True)
    print('[*] 加载 adapter: %s' % args.adapter)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print('[OK] 已合并导出: %s' % args.out)
    print('     下一步：convert_hf_to_gguf.py 转 GGUF → ollama create <name> -f Modelfile')
    print('     再把 serving/qa_service.py 的 OLLAMA_MODEL 换成新模型名即可接回')


if __name__ == '__main__':
    main()
