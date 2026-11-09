# 微调方案（LoRA on Qwen2.5）

> 本目录交付：**数据集构造脚本 + 训练脚本 + 合并导出脚本 + 方案设计**。
> 受本机环境限制（torch 为 CPU 版、ModelScope 大文件下载实测 0.18MB/s），未在本机实跑训练，
> 脚本与数据可在有 GPU 的机器上执行。

## 一、为什么微调

当前系统（RAGFlow 检索 + deepseek-r1:7b 生成 + 提示词约束）在 49 条评测上最新一轮为 **98.0%**（48/49；修复引用/拒答逻辑前为 87.8%~93.9%），仍有三个问题：

| 问题 | 现象 | 微调能否解决 |
|---|---|---|
| 输出冗长 | 推理模型会先输出思考过程，答案被稀释 | ✅ SFT 可把"直接给结论"固化进模型 |
| 格式不稳 | 有时漏掉条款号、有时把提示混进依据 | ✅ 训练样本统一为「结论/依据/提示」并带 `[1]` 引用 |
| 拒答边界不稳 | 偶尔对库外问题给模糊回答 | ✅ 混入 10 条拒答样本（无引用），强化"没有就说没有" |

> 注意：**知识不靠微调灌入**，政策条文仍由 RAGFlow 检索提供；微调只解决"怎么说、什么时候不说"。

## 二、数据构造（`build_dataset.py`）

**关键：训练样本必须与线上 prompt 结构一致（train-serve 对齐）**

线上 `serving/qa_service.py` 的 user 消息是「知识库内容（带 `[n]` 来源）+ 用户问题 + 7 条硬性要求」，
并要求答案带 `[n]` 引用、无依据时拒答。因此这里每条样本都构造为三段式 `messages`：

```
system：与线上同一份 prompts/system_prompt.md（去掉给 Dify UI 用的开场白段）
user  ：知识库内容：\n[1] 来源：财政政策知识库\n<该条的「依据」原文>\n\n用户问题：<问题>\n\n【硬性要求】...
assistant：**结论**：...\n**依据**：... [1]\n**提示**：...
```

样本来源：
1. **知识库 59 条 QA** → 上下文取该条「依据」原文，答案带 `[1]` 引用（59 条带引用）
2. **10 条库外问题** → 上下文为 `（无）`，答案明确"知识库中没有查到"，**不带引用**（与服务端拒答判定一致）

统计：69 条（59 QA + 10 拒答），按 85/15 切训练/验证，见 `data/finetune/stats.json`。

数据格式（jsonl，同时保留 `instruction/output` 兼容旧脚本）：
```json
{"messages": [
   {"role": "system", "content": "<线上 system prompt>"},
   {"role": "user", "content": "知识库内容：\n[1] 来源：财政政策知识库\n《...》第八条：...\n\n用户问题：预留份额比例是多少？\n\n【硬性要求】..."},
   {"role": "assistant", "content": "**结论**：30% 以上...\n**依据**：《政府采购促进中小企业发展管理办法》第八条：... [1]"}
 ], "instruction": "...", "output": "..."}
```

> ⚠️ 评测污染提醒：`eval/eval_set.md` 中有 14 题与知识库 QA 同题、部分负样本与训练拒答同源。
> 训练前后对比时应从评测集剔除同题，或把结果拆成「已见题 / 未见题」两组分别报告。

## 三、训练配置（`train_lora.py`）

| 项 | 取值 | 理由 |
|---|---|---|
| 基座 | Qwen2.5-1.5B-Instruct | 8G 显存下 bf16 LoRA 可训；3B 勉强，7B 不可行 |
| 方法 | LoRA（r=8, alpha=16, q/k/v/o_proj） | 显存友好，不动全参 |
| 精度 | bf16（不用 fp16/QLoRA） | RTX 5060 是 Blackwell(sm_120)，bitsandbytes 支持不稳 |
| 优化器 | adamw_torch（**不用** paged_adamw_8bit） | 后者依赖 bitsandbytes，与上面的选型一致 |
| warmup | warmup_steps（按总步数 10% 计算） | transformers 5.x 已移除 warmup_ratio 参数 |
| batch | 1 × 梯度累积 8 | 8G 显存约束 |
| lr | 2e-4，cosine | LoRA 常用区间 |
| epoch | 3 | 69 条小数据，过多会过拟合 |
| max_len | 1024（prompt 超长会显式报错，不会静默全 mask） | 政策条文较长，已被上下文预算截断 |
| loss | 只在答案段计算（prompt 部分 label=-100） | 让模型学"怎么答"而非"怎么问" |

## 四、依赖（必须钉版本，否则跑不起来）

```bash
# GPU 机器（RTX 5060 是 sm_120，需要 CUDA 12.8 轮子 → torch>=2.7）
pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=4.46,<6" "peft>=0.13" "datasets>=2.20" "accelerate>=1.1"

# 说明：
# - transformers 4.44/4.45 只有 evaluation_strategy，本脚本用 eval_strategy 会 TypeError
# - transformers 5.x 已移除 warmup_ratio（本脚本改用 warmup_steps）
# - accelerate 低于 1.1.0 时 TrainingArguments 直接 ImportError
```

## 五、跑训练（GPU 机器）

```bash
# 1) 构造数据（任意机器）
python finetune/build_dataset.py

# 2) 训练（脚本内部用绝对路径定位数据，不依赖当前目录）
python finetune/train_lora.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --data  data/finetune/train.jsonl \
    --dev   data/finetune/dev.jsonl \
    --out   finetune/output \
    --epochs 3 --batch 1 --grad_accum 8

# 3) 合并导出（把 adapter 合并回基座）
python finetune/merge_export.py \
    --base Qwen/Qwen2.5-1.5B-Instruct \
    --adapter finetune/output \
    --out finetune/merged

# 4) 转 GGUF 交给 Ollama
python llama.cpp/convert_hf_to_gguf.py finetune/merged --outfile fiscal-qwen.gguf
ollama create fiscal-qwen -f Modelfile     # Modelfile 内容：FROM ./fiscal-qwen.gguf
```

> 本机（无 GPU 环境）只能验证到"数据构造 + 脚本语法"；CPU 试通需要手动装
> `peft / datasets / accelerate` 并用 0.5B 模型，且极慢，不建议。

## 六、接回线上服务

`serving/qa_service.py` 调的是 Ollama `/api/chat`，所以最简路径是：
1. `merge_export.py` 合并 → 转 GGUF → `ollama create`
2. 把环境变量 `OLLAMA_MODEL` 换成新模型名（接口与调用方式都不变）
3. 重跑 `python scripts/eval_judge.py` 对比「微调前 / 微调后」（注意上面的评测污染提醒）

若要走 vLLM：起 OpenAI 兼容服务后，把 `OLLAMA_BASE` 指向它并确认 `/api/chat` 兼容性
（vLLM 默认是 `/v1/chat/completions`，需要改 `serving/qa_service.py` 的调用路径）。

## 七、已知限制

- 本机未实跑训练；脚本在 transformers 5.15.1 下通过了参数兼容性检查（`eval_strategy` 存在、
  `warmup_ratio` 已移除、`adamw_torch` 合法），但未验证收敛与效果。
- `data/finetune/*.jsonl` 为构造产物，已在 `.gitignore` 中排除，可随时重建。
