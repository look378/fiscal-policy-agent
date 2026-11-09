# 财政政策 AI 智能体（Fiscal Policy AI Agent）

> 面向财政/政务场景的政策问答与运行预警系统：**RAGFlow 做文档解析与检索**，自研编排层做**引用强制与无依据拒答**，本地模型生成，配套**实时监控看板与 LLM 异常预警**，容器化部署并支持 Kubernetes 编排。

`Python` · `RAGFlow` · `Dify 0.15.3` · `Ollama / deepseek-r1` · `FastAPI` · `Prometheus` · `Grafana` · `Kubernetes(kind)` · `LLM-as-judge`

> 本仓库是**完整可复现的工程实现**：从政策原文清洗、知识库构建、检索问答、引用/拒答约束、评测体系，到监控看板与异常预警，含全部源码、评测数据与三份代码审查报告。

## 1. 解决什么问题

| 痛点 | 现状 | 本项目做法 |
|---|---|---|
| 政策文件分散、检索难 | 政策散落在多个网站/PDF，人工翻找 | RAGFlow 解析 5 份政策原文 + 结构化知识库，条款级检索 |
| 大模型编造政策条款 | 通用模型容易"张冠李戴"，乱给比例和金额 | 引用强制 + 数字逐字校验 + 无依据拒答 |
| 政策问答质量不可度量 | 上线后没人知道答得对不对 | 49 条回归评测集 + LLM 裁判 + 数字硬校验 |
| 异常无人发现 | 拒答率升高、延迟劣化靠用户投诉 | Prometheus 指标 + Grafana 看板 + LLM 归因预警 |

## 2. 当前实测结果

| 指标 | 数值 |
|---|---|
| 知识库 | 6 个文档（59 条 QA + 5 份政策原文），199 个 chunk，bge-m3 向量化 |
| 评测集 | 49 条（8 条负样本专测"会不会乱编"） |
| **通过率** | **48/49 = 98.0%**（LLM 裁判 + 数字硬校验复核后） |
| 正样本 | 41/41 = 100% |
| 负样本正确拒答 | 7/8 = 87.5% |
| 平均响应 | 18.2 秒（本地 deepseek-r1:7b，CPU 嵌入） |
| Bad case | 逐条归因与修复状态见 [`eval/badcase_log.md`](eval/badcase_log.md) |

> 说明：修复引用/拒答逻辑后最新一轮为 **48/49**；修复前多轮在 87.8%~93.9% 之间波动，主要差异在负样本误判与引用判定。单轮结果仍受采样影响，建议以 3 轮均值为准。
> 评测口径：LLM 裁判先判，再用「期望数字必须出现在回答里」的硬规则复核，
> 避免裁判放水（初版裁判把答错的 #12 判成了通过，被这套复核抓出来）。

## 3. 架构

```
财政政策原文(PDF/网页)
        │
        ├──────────────────────────┐
        ▼                          ▼
┌────────────────────┐   ┌───────────────────────────┐
│ RAGFlow（Docker）  │   │ Dify 0.15.3（Docker）      │
│ ├ 文档解析/切块     │   │ ├ 知识库（6 文档已索引）    │
│ ├ bge-m3 向量化    │   │ ├ 对话应用（绑定知识库）    │
│ └ 混合检索 API     │   │ └ 编排/会话/引用            │
└─────────┬──────────┘   └─────────────┬─────────────┘
          │                            │
          ▼                            ▼
┌──────────────────────────┐   本地 Ollama (deepseek-r1:7b)
│ 自研问答编排层 FastAPI    │   本地 bge-m3 embedding 服务
│ ├ 检索（top_k=8）         │
│ ├ 引用强制 + 数字校验      │
│ ├ 无依据拒答（三层）       │
│ └ OpenAI 兼容接口         │
└─────────────┬────────────┘
              ▼
   Prometheus（19090）──▶ Grafana 看板（13000）
              │
              ▼
   LLM 异常预警服务（8902）
   （规则判定 → LLM 生成原因+处置建议 → /alerts）

   Kubernetes（kind）：Deployment / Service / HPA
```

## 3.1 在线服务一览

| 服务 | 地址 | 说明 |
|---|---|---|
| RAGFlow | http://localhost/（API :9380） | 文档解析 + 混合检索 |
| Dify | http://localhost:8080 | 知识库 + 对话应用（本地演示账号：admin@fiscal.local / `Fiscal@2026`） |
| 自研问答服务 | http://127.0.0.1:8901/api/ask | 引用强制 + 拒答 + Prometheus 指标（内置试用页 `GET /`） |
| 本地 embedding | http://127.0.0.1:8900/v1 | bge-m3（OpenAI 兼容） |
| 异常预警服务 | http://127.0.0.1:8902/alerts | 规则 → LLM 归因 |
| Prometheus | http://localhost:19090 | 指标存储（15s 抓取，7 条告警规则） |
| Grafana | http://localhost:13000 | 看板「财政政策智能体 · 运行监控」（本地演示账号：admin / `fiscal2026`） |
| Ollama | http://localhost:11434 | deepseek-r1:7b 本地推理 |

> 以上账号/口令均为**本地 Docker 演示默认值**，请勿用于公网部署；生产环境用环境变量或 Secret 覆盖（见 `.env.example`）。


## 4. 目录结构

```
财政AI智能体/
├── data/
│   ├── policies/            # 政策原文（网页抓取，*.txt 不入库）
│   ├── policies_clean/       # 清洗后按条款结构化（入库）
│   ├── knowledge_base.md     # 结构化知识库（59 条 QA）
│   └── finetune/             # LoRA 微调数据集（*.jsonl 不入库，可重建）
├── eval/
│   ├── eval_set.md           # 49 条回归评测集
│   ├── eval_final.md         # 评测结果（裁判 + 数字硬校验）
│   ├── eval_judge.json       # 逐题原始回答
│   └── badcase_log.md        # bad case 归因与修复记录
├── prompts/system_prompt.md  # 系统提示词（服务端与 Dify 共用）
├── scripts/                  # 抓取/清洗/导入/评测/回归/Dify 配置脚本
├── serving/
│   ├── qa_service.py         # 问答编排服务（含试用页 / SSE / metrics）
│   ├── embed_server.py       # 本地 bge-m3 embedding 服务
│   ├── refusal_markers.py    # 拒答词表（服务端与评测共用）
│   └── Dockerfile / Dockerfile.embed
├── finetune/
│   ├── build_dataset.py      # 知识库 → 与线上对齐的 SFT 数据集
│   ├── train_lora.py         # Qwen2.5 LoRA（bf16）
│   ├── merge_export.py       # 合并 adapter / 导出
│   └── README.md             # 微调方案与依赖说明
├── monitoring/
│   ├── prometheus.yml        # 抓取配置
│   ├── prometheus-rules.yml  # 7 条告警规则
│   ├── docker-compose.yml    # Prometheus + Grafana（+ 可选容器化告警服务）
│   ├── grafana/              # 数据源/看板自动配置（10 面板）
│   └── alert_service/        # LLM 异常预警（规则→归因→落盘）
├── k8s/manifests.yaml        # Namespace/Secret/ConfigMap/Deployment/Service/HPA/PDB
├── dify015/                  # Dify 0.15.3 部署配置（.env 与 volumes 不入库）
├── ragflow/                  # RAGFlow 部署配置（.env 与 logs 不入库）
├── docs/                     # 技术解析 / 三份审查报告 / 试用指南
├── .env.example              # 环境变量模板
├── LICENSE
└── README.md
```

> **仓库不含**：模型权重、容器数据卷（`*/volumes/`、`monitoring/data/`）、运行日志、`*.env`、`.dify*_token`、`参考资料/`（第三方 Helm 模板）与 `archive/`（历史归档）。这些都能按下文步骤重建。

## 5. 关键技术决策（面试可讲）

| 决策 | 原因 |
|---|---|
| RAGFlow 只做解析+检索，问答编排自研 | RAGFlow 内置对话流程黑盒、不好做引用硬约束；自研层能精确控制提示词、数字校验与拒答 |
| 不用推理模型直接问答 | deepseek-r1 会把 token 花在思考上且易编数字，必须配硬约束与上下文 |
| LLM 裁判 + 数字硬校验双轨评测 | 纯关键词会误判（实测把正确答案判失败），纯裁判会放水（实测把错答案判通过） |
| 知识库既有 QA 也有原文 | QA 保证回答质量，原文保证条款级召回与引用溯源 |
| LoRA 用 bf16 不用 QLoRA | RTX 5060 是 Blackwell(sm_120)，bitsandbytes 支持不稳 |
| 微调只解决"格式与纪律" | 政策知识靠检索注入，不靠模型记忆（避免知识过期与幻觉） |

## 6. 快速启动

先准备环境变量（不设置时使用本机演示默认值）：
`cp .env.example .env`，或 PowerShell 里 `$env:RAGFLOW_KEY="ragflow-..."`。

> 两套第三方栈的环境文件：`ragflow/.env` 与 `dify015/.env` 不入库，需分别从各自目录的
> `.env.example` 复制并修改密码（已有数据卷时沿用原密码）。

```bash
# 0) 一键启动（推荐）：RAGFlow + embedding + 问答 + 预警 + 监控
powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1
#   可指定 bge-m3 模型目录： ... -BgePath <目录>；跳过监控： -SkipMonitoring

# 手动分步：
# 1) RAGFlow（工作目录 ragflow/）
docker compose -p fiscal-ragflow up -d

# 2) 本地 embedding 与问答服务
python serving/embed_server.py --model-path <bge-m3目录> --port 8900
python serving/qa_service.py --port 8901          # 容器内需加 --host 0.0.0.0

# 3) 导入知识库（首次；文件路径相对仓库根）
python scripts/import_kb.py --key <RAGFLOW_KEY> \
    --files data/knowledge_base.md data/policies_clean/01_政府采购促进中小企业发展管理办法_财库2020-46号.md

# 4) 跑评测
python scripts/eval_judge.py      # LLM 裁判 → eval/eval_judge.json
python scripts/eval_verify.py     # 加数字硬校验复核 → eval/eval_final.md

# 5) 监控与预警
cd monitoring && docker compose up -d
python monitoring/alert_service/alert_service.py --port 8902

# 6) Dify（0.15.3，工作目录 dify015/）
docker compose -p dify015 up -d
python scripts/setup_dify015_models.py     # 注册本地模型
python scripts/setup_dify015_app.py       # 建知识库 + 索引文档 + 建应用
python scripts/bind_dify_dataset.py        # 绑定知识库（含 enabled / 引用开关）
python scripts/test_dify_app.py "预留份额比例是多少？"   # 断言检索引用非空

# 7) 容器化与 K8s
powershell -ExecutionPolicy Bypass -File scripts\build_images.ps1
powershell -ExecutionPolicy Bypass -File scripts\deploy_kind.ps1   # 需先安装 kind
```

## 7. 已知限制与踩坑

- **Dify 版本选择**：先部署了 1.18/1.1.3（插件架构），模型供应商插件在 daemon 层装成功但 API 层识别不到，**降级到 0.15.3（内置供应商）**后一次跑通；
- 本机网络对 Docker Hub 不稳定（DNS 污染 `registry-1.docker.io → 31.13.69.169`），依赖国内镜像站 + 多次重试；
- 微调脚本与数据已就绪，但本机 torch 为 CPU 版、大文件下载限速（0.18MB/s），**未实跑训练**；
- 生成模型为本地 deepseek-r1:7b（推理模型）：**端到端**约 15~25 秒/题（含检索、生成、引用），其中纯生成约 12~18 秒；embedding 走 CPU（bge-m3），首问较慢；
- RAGFlow 与 Dify 两套 RAG 并存，知识库内容相同但切块/索引独立（便于对比两种平台的检索效果）。
- 内置试用页（http://127.0.0.1:8901/）仅本机演示用：无鉴权、无速率限制；OpenAI 兼容接口的 `stream=true` 是**单 chunk 伪流式**（生成完再推送）。
- 监控告警**不推送**：仓库未部署 Alertmanager、Grafana 也未配置 Contact Point，Prometheus 规则（7 条）的 firing 只在 Prometheus UI/`/api/v1/alerts` 可见；自研预警服务的告警在 `127.0.0.1:8902/alerts` 与 `logs/` 中。
- **密钥管理**：仓库不含任何真实部署密钥（`.env`、`.dify*_token`、容器卷均不入库）。`dify015/.env.example` 与 `dify015/docker-compose.yaml` 中的 `SECRET_KEY` 是 Dify 官方示例值，生产环境必须用 `openssl rand -base64 42` 重新生成并写入 `.env`；RAGFlow 的 API Key 通过环境变量或 K8s Secret 注入。

## 8. 免责声明

个人学习与作品展示用途，不构成财政、法律或审计意见；政策条款以官方最新发布为准。

## 9. 文档导航

| 文档 | 内容 |
|---|---|
| [`docs/技术解析.md`](docs/技术解析.md) | 架构取舍、检索/生成/评测/监控四层设计、关键指标口径 |
| [`docs/试用指南.md`](docs/试用指南.md) | 各前端页面入口、账号、建议试用顺序 |
| [`docs/新增代码审查.md`](docs/新增代码审查.md) | 第三轮审查：新增页面/SSE/告警/脚本的 3 个独立复审结论与修复 |
| [`docs/代码审查报告-v2.md`](docs/代码审查报告-v2.md) | 第二轮独立全面审查（5 BLOCKER / 30 HIGH）与修复状态 |
| [`docs/代码审查报告.md`](docs/代码审查报告.md) | 第一轮审查：8 BLOCKER + 工程化问题清单 |
| [`eval/badcase_log.md`](eval/badcase_log.md) | 逐条 bad case 归因、修复动作与回归结果 |
| [`finetune/README.md`](finetune/README.md) | 微调数据构造（与线上 prompt 对齐）、依赖版本、导出步骤 |

## 10. License

[MIT](LICENSE) © 2026 look378
