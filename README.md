# 财政政策智能体 · Fiscal Policy AI Agent

> 财政/政务场景的政策问答与运行监控：RAGFlow 负责政策原文解析与条款检索，自研编排层负责引用强制、
> 数字校验与无依据拒答，本地模型生成；配套评测体系、运行看板与 LLM 异常预警。

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![RAGFlow](https://img.shields.io/badge/RAGFlow-v0.24-FE6A39)](https://ragflow.io/)
[![Dify](https://img.shields.io/badge/Dify-0.15.3-1C64F2)](https://dify.ai/)
[![Ollama](https://img.shields.io/badge/Ollama-deepseek--r1:7b-000000)](https://ollama.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-dashboard-F46800?logo=grafana&logoColor=white)](https://grafana.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-kind-326CE5?logo=kubernetes&logoColor=white)](https://kind.sigs.k8s.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 一、简介

政策问答最容易出的问题不是"检索不到"，而是**答得太像真的**：模型给出一个比例、一个金额、一个期限，
听起来有出处，实际上是把别的条款、别的地区、甚至训练记忆里的数字串了过来。政务/财政场景里，
一条错的比例会直接误导申报。

这个项目把"依据"做成硬约束，而不是靠提示词祈祷：

1. **引用可核对**。每条回答都要求标注 `[n]`，接口只返回"答案里确实引用、且确实进入模型上下文"的片段；
   模型没标引用时返回空引用列表，不伪造来源。
2. **没依据就拒答**。三层拦截：检索相似度闸门（top1 < 0.33）→ 文本拒答（强拒答词 + 无有效引用）→ 服务异常兜底。
   拒答与回答分别计数，能看出"是知识库没覆盖，还是模型不肯答"。
3. **效果可度量、运行可观测**。49 条回归集（含 8 条库外负样本）用 LLM 裁判 + 数字硬校验双轨评测；
   6 类 Prometheus 指标 + 10 面板看板 + LLM 异常归因预警。

> 当前实测：**48/49 = 98.0%**（正样本 41/41，负样本正确拒答 7/8，平均端到端 18.2s）。
> 评测口径与已知局限见下文，未达 100% 的原因已逐条记录在 [`eval/badcase_log.md`](eval/badcase_log.md)。

---

## 二、功能一览

### 2.1 政策知识库构建

从公开政策原文（网页/PDF 文本）清洗出条款级 Markdown，并补充 59 条结构化 QA：

| 层 | 内容 | 用途 |
|---|---|---|
| 原文层 | 5 份财政政策（政府采购促进中小企业发展、政府购买服务、政府采购法实施条例、专项资金、直达资金） | 条款级召回与引用溯源 |
| QA 层 | 59 条「问题 → 结论 / 依据 / 提示」 | 保证常见问题的回答质量与口径一致 |

清洗脚本按「第X条」切分、剥离站点导航与页脚（ICP 备案、主办单位等），通知类文档用正文锚点定位；
结果落 `data/policies_clean/`，索引到 RAGFlow 后共 6 个文档 / 103 个 chunk（bge-m3，1024 维）。

### 2.2 检索与问答编排

自研 FastAPI 服务（`serving/qa_service.py`），一次问答的编排：

| 步骤 | 做法 | 参数 |
|---|---|---|
| 检索 | 调 RAGFlow 混合检索，只对网络/5xx 重试（4xx 直接失败） | `top_k=8`、相似度阈值 0.2 |
| 上下文装配 | 单块截断 + 总长预算（按实际来源头长度计），只把真正进入 prompt 的片段记为 `used` | 单块 600 字 / 总 3000 字 |
| 生成 | 调本地 Ollama，推理模型空回答时放大预算重试一次 | `deepseek-r1:7b`、`num_ctx=8192` |
| 引用判定 | 解析 `[n]` / `【n】` / `[1,2]` / `[1-3]`，过滤越界编号，只保留进过上下文的片段 | — |
| 拒答判定 | 相似度闸门 + 强拒答词（开头 120 字内）+ 无有效引用 | 闸门 0.33 |

接口：`POST /api/ask`、`POST /v1/chat/completions`（OpenAI 兼容，支持 SSE）、`GET /v1/models`、
`GET /livez`（存活）、`GET /health`（就绪，依赖不可用返回 503）、`GET /metrics`，以及内置试用页 `GET /`。

### 2.3 引用与拒答的三层边界

```
检索 → [闸门] top1 < 0.33 → 直接拒答（约 0.5~0.9s，不调用模型）
            ↓ 通过
        生成 → [文本] 开头含强拒答词且无有效引用 → 记为文本拒答
            ↓ 未拒答
        返回 → citations 只含"被引用 且 进过上下文"的片段；无引用则空 + citation_verified=false
```

指标上 `refused_total = 闸门拒答 + 文本拒答`，并单列 `refused_gate_total` / `refused_text_total`，
避免"模型变得爱拒答"和"知识库召回变差"混在一起看不出来。

### 2.4 评测体系（49 条）

| 项 | 说明 |
|---|---|
| 正样本 | 41 条，覆盖中小企业认定、预留份额、价格评审、采购程序、政府购买服务、专项资金、直达资金 |
| 负样本 | 8 条库外问题（地方补贴金额、未收录法条、年度额度等），期望模型明确说"没查到" |
| 裁判 | LLM 裁判先判（temperature=0），再叠确定性硬校验 |
| 硬校验 | 期望数字必须出现（带边界，`200万元` 不匹配 `1200万元`）；负样本要求"裁判通过 + 无引用 + 无编造数字" |
| 三态 | PASS / FAIL / SKIP（服务异常或裁判无法解析记 SKIP，不计入通过率） |

### 2.5 运行监控（Prometheus + Grafana）

指标：请求数、拒答数（闸门/文本分列）、错误数、累计延迟（成功与拒答都计）、completion/prompt token、
进入上下文的片段数。Grafana 10 个面板：QPS、错误率、拒答率、平均延迟、趋势、token 速率、服务存活、
告警服务检查/触发、待归因/落盘失败/归因失败。

Prometheus 侧另有 7 条规则（服务 down、target 缺失、错误率、延迟、预警服务停滞），带最小样本量门限（≥5），
避免低流量下"3 次请求 1 次失败"就触发。

### 2.6 LLM 异常预警

`monitoring/alert_service/`：从 Prometheus 拉指标 → 4 条规则（服务 down / 错误率 >10% / 拒答率 >40% /
平均延迟 >20s）→ 连续 2 次命中才触发（去抖）→ 告警先落盘，再异步调用本地 LLM 生成「可能原因 / 处置建议 /
是否需人工」，结果回写 JSONL 与 `/alerts`。重启时回放历史、恢复冷却窗口，遗留 pending 标记为 orphaned。

### 2.7 平台集成与交付

- **Dify 0.15.3**：知识库（hybrid_search / top_k=4）+ 对话应用绑定 + 引用开关，脚本会建库后回读断言，
  避免"配置被接口静默丢弃"的假成功；端到端冒烟断言检索引用非空。
- **容器化**：`fiscal-qa` / `fiscal-alert` 两个镜像，非 root 运行，健康检查指向 `/livez`。
- **Kubernetes（kind）**：Namespace / Secret / ConfigMap / 两个 Deployment / Service / HPA / PDB，
  含 `command+args` 启动、`fsGroup` 卷权限、readiness 与 liveness 分离。

---

## 三、架构

### 3.1 分层设计

```
数据层    政策原文清洗 → 条款级 Markdown + 结构化 QA（data/）
   ↓
检索层    RAGFlow（解析 / 切块 / bge-m3 向量化 / 混合检索）
   ↓
编排层    FastAPI：检索 → 上下文预算 → 生成 → 引用判定 → 拒答判定（serving/qa_service.py）
   ↓
能力层    Ollama deepseek-r1:7b（生成）· bge-m3 本地服务（向量化，OpenAI 兼容）
   ↓
观测层    Prometheus 指标 → Grafana 看板 · 预警服务（规则 + LLM 归因）
   ↓
交付层    Docker Compose / Kubernetes（kind）· Dify 平台对照实现
```

### 3.2 一次问答的主流程

```
POST /api/ask
  ├─ 参数校验（空/空白 → 422）
  ├─ RAGFlow 检索（top_k=8，失败按瞬态错误重试）
  ├─ 相似度闸门：top1 < 0.33 → 拒答（refused_by=gate）
  ├─ 上下文装配：used = 真正进入 prompt 的片段
  ├─ Ollama 生成（空回答 → 放大预算重试一次）
  ├─ 引用解析：只保留"被引用 且 在 used 中"的片段
  ├─ 文本拒答：开头 120 字含强拒答词 且 无有效引用 → refused_by=text
  └─ 返回 answer / citations / citation_verified / refused / retrieved / top_similarity / latency_ms
```

### 3.3 目录结构

```
财政AI智能体/
├── data/                     # 政策原文、清洗结果、知识库、微调数据构造
├── eval/                     # 49 条评测集、评测结果、逐题原始数据、badcase 台账
├── prompts/system_prompt.md  # 系统提示词（服务端与 Dify 共用）
├── scripts/                  # 抓取清洗、知识库导入、评测回归、Dify 配置、启动与部署
├── serving/                  # 问答编排服务、embedding 服务、拒答词表、两个 Dockerfile
├── finetune/                 # SFT 数据构造、LoRA 训练、合并导出、方案说明
├── monitoring/               # Prometheus 配置与规则、Grafana 看板、LLM 预警服务
├── k8s/manifests.yaml        # kind 部署清单（Secret 用占位符）
├── dify015/ ragflow/         # 两套平台部署配置（.env 与数据卷不入库）
├── docs/                     # 技术解析、试用指南、三份代码审查报告
├── .env.example              # 环境变量模板
├── .github/workflows/ci.yml  # CI：Python 编译 + YAML/JSON 校验 + 必需文件检查
└── README.md
```

---

## 四、快速开始

### 4.1 前置条件

- Windows（脚本按 PowerShell 5.1 兼容处理）或 Linux/macOS
- Docker Desktop（RAGFlow、Dify、Prometheus/Grafana 都用容器）
- Python 3.11+
- Ollama + `deepseek-r1:7b`；bge-m3 权重目录（本地 embedding 服务需要）

### 4.2 一键启动

```powershell
# 环境变量：不设置时使用本机演示默认值
copy .env.example .env

# RAGFlow + embedding + 问答 + 预警 + 监控
powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1

# 可选参数：-BgePath <bge-m3 目录>；-SkipMonitoring 跳过监控栈
```

### 4.3 分步启动

```bash
# 1) RAGFlow（工作目录 ragflow/，首次需 cp .env.example .env）
docker compose -p fiscal-ragflow up -d

# 2) 本地 embedding + 问答服务
python serving/embed_server.py --model-path <bge-m3目录> --port 8900
python serving/qa_service.py --port 8901        # 容器内需加 --host 0.0.0.0

# 3) 知识库首次导入（文件路径相对仓库根）
python scripts/import_kb.py --key <RAGFLOW_KEY> \
    --files "data/knowledge_base.md" "data/policies_clean/*.md"

# 4) 监控与预警
cd monitoring && docker compose up -d
python monitoring/alert_service/alert_service.py --port 8902

# 5) Dify 0.15.3（工作目录 dify015/，首次需 cp .env.example .env）
docker compose -p dify015 up -d
python scripts/setup_dify015_models.py
python scripts/setup_dify015_app.py
python scripts/bind_dify_dataset.py
python scripts/test_dify_app.py "预留份额比例是多少？"   # 断言引用非空

# 6) 容器镜像与 kind 集群
powershell -ExecutionPolicy Bypass -File scripts\build_images.ps1
powershell -ExecutionPolicy Bypass -File scripts\deploy_kind.ps1   # 需安装 kind
```

### 4.4 验证

```bash
# 存活 / 就绪 / 指标
curl http://127.0.0.1:8901/livez
curl http://127.0.0.1:8901/health
curl http://127.0.0.1:8901/metrics

# 问答冒烟（正常问题应带引用；库外问题应 1s 内拒答）
curl -s -X POST http://127.0.0.1:8901/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"中小企业参与政府采购，预留份额比例是多少？"}'
```

页面：自研试用页 http://127.0.0.1:8901/ ｜ Grafana http://localhost:13000 ｜ Dify http://localhost:8080

---

## 五、实测结果与评测口径

| 指标 | 数值 |
|---|---|
| 知识库 | 6 个文档（59 条 QA + 5 份政策原文）/ 103 个 chunk / bge-m3（1024 维） |
| 评测集 | 49 条（41 正 + 8 负） |
| 总通过率 | **48/49 = 98.0%** |
| 正样本 | 41/41 = 100% |
| 负样本正确拒答 | 7/8 = 87.5% |
| 平均端到端延迟 | 18.2s（本地 deepseek-r1:7b，CPU embedding） |
| 评测降级 / SKIP | 0 / 0 |

口径说明：

- 裁判与硬校验**分开各司其职**：裁判判断"意思对不对"，硬校验只做确定性检查（数字是否出现、负样本是否声明未找到且无编造）。
  纯关键词法会把"措辞不同但正确"判错（实测 30.6%）；纯 LLM 裁判会放水（实测 91.7% 虚高）。
- 硬校验用数字边界匹配，避免 `200万元` 被 `1200万元` 误命中；区间数字命中任一合法端点即算出现。
- 唯一失败题（#44 投诉处理期限）是知识库确实未收录该法条、模型倾向硬答，归因与修复建议见 badcase 台账。
- 单轮结果仍受采样影响；修复引用/拒答逻辑前多轮在 87.8%~93.9% 波动，建议以多轮均值为准。

---

## 六、技术栈

### 6.1 核心技术选型

| 层 | 选型 | 说明 |
|---|---|---|
| 文档解析/检索 | RAGFlow v0.24 | 解析、切块、bge-m3 向量化、混合检索 API |
| 向量化 | bge-m3（本地服务，OpenAI 兼容） | 1024 维，CPU 推理 |
| 生成 | Ollama `deepseek-r1:7b` | 本地推理，不依赖外部 API |
| 问答编排 | FastAPI（自研） | 检索、上下文预算、引用/拒答、指标 |
| 评测 | LLM-as-judge + 数字硬校验 + 负样本确定性 | 见第五节 |
| 监控 | Prometheus + Grafana | 指标 + 7 条规则 + 10 面板 |
| 异常预警 | 自研（规则 + LLM 归因） | 规则判定、去抖、异步归因、JSONL 回放 |
| 平台对照 | Dify 0.15.3 | 知识库/应用/引用链路对照 |
| 部署 | Docker Compose / Kubernetes(kind) | 非 root 镜像、健康检查、探针与 PDB |

### 6.2 关键取舍

| 取舍 | 原因 |
|---|---|
| RAGFlow 只做解析+检索，问答编排自研 | 平台内置对话流程不可控，做不了引用强制、数字校验与拒答分层 |
| 生成用推理模型 + 硬约束 | 7B 推理模型会编数字；靠"上下文预算 + 提示词硬约束 + 引用判定 + 拒答"兜底，而不是换更大模型 |
| 知识库 QA 与原文并存 | QA 保证常见问题答案口径，原文保证条款级召回与可核对引用 |
| 微调用 bf16 LoRA 而非 QLoRA | RTX 5060 是 Blackwell(sm_120)，bitsandbytes 支持不稳；训练只固化输出纪律，不灌知识 |
| 微调数据按线上 prompt 结构构造 | 训练样本带检索上下文与 `[n]` 引用，避免 train-serve 偏差 |
| 告警不接 Alertmanager | 本机部署无推送通道，规则状态在 Prometheus UI 与 `/alerts` 可见；接通知需另配 Alertmanager 或 Grafana Contact Point |

---

## 七、配置说明

主要环境变量（完整列表见 [`.env.example`](.env.example)）：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAGFLOW_BASE` / `RAGFLOW_KEY` / `RAGFLOW_DATASET` | `http://localhost:9380` / 内置演示值 / 数据集 ID | RAGFlow 检索入口 |
| `OLLAMA_BASE` / `OLLAMA_MODEL` | `http://127.0.0.1:11434` / `deepseek-r1:7b` | 本地生成 |
| `TOP_K` / `SIM_THRESHOLD` | `8` / `0.2` | 检索条数与初筛阈值 |
| `MIN_SIM_TO_ANSWER` | `0.33` | 相似度闸门，低于则直接拒答 |
| `MAX_TOKENS` / `NUM_CTX` | `1536` / `8192` | 生成预算与上下文窗口 |
| `CHUNK_CHARS` / `MAX_CTX_CHARS` | `600` / `3000` | 单块截断与总上下文预算 |
| `MODEL_PATH` | — | bge-m3 权重的本地路径（embedding 服务） |
| `PROM_URL` | `http://localhost:19090` | 预警服务拉取指标的 Prometheus |
| `MIN_SAMPLES` / `CONSECUTIVE` | `5` / `2` | 预警最小样本量与去抖次数 |
| `DIFY_BASE` / `DIFY_EMAIL` / `DIFY_PASSWORD` | `http://localhost:8080` / 演示账号 | Dify 配置脚本使用 |

> 仓库内所有默认账号与口令均为**本机演示值**，生产环境用环境变量或 Secret 覆盖；
> `dify015/.env.example` 中的 `SECRET_KEY` 是 Dify 官方示例，生产必须用 `openssl rand -base64 42` 重新生成。

---

## 八、测试与自检

| 检查 | 命令 | 期望 |
|---|---|---|
| Python 语法 | `python -m compileall serving scripts finetune monitoring/alert_service` | 全部通过（CI 已覆盖） |
| 配置合法性 | CI 校验全部 YAML/JSON | 通过 |
| 服务存活/就绪 | `GET /livez` / `GET /health` | 200 / 200（依赖不可用 503） |
| 全量评测 | `python scripts/eval_judge.py` → `python scripts/eval_verify.py` | 生成 `eval/eval_final.md`，exit 0 |
| 单题回归 | `python scripts/eval_regress.py --nos 1 7 44` | 通过 exit 0，失败 exit 1，服务异常 SKIP exit 2 |
| Dify 检索断言 | `python scripts/test_dify_app.py "..."` | 引用条数 > 0，否则 exit 2 |
| 告警服务 | `GET /alerts`、`GET /metrics` | 返回历史告警与 6 类指标 |

CI（GitHub Actions）在每次 push 时执行：Python 编译、YAML/JSON 校验、必需文件检查。历史运行见仓库 Actions 页。

---

## 九、文档索引

| 文档 | 内容 |
|---|---|
| [`docs/技术解析.md`](docs/技术解析.md) | 检索/生成/评测/监控四层设计、参数依据、指标口径 |
| [`docs/试用指南.md`](docs/试用指南.md) | 各页面入口、账号、建议试用顺序与常见问题 |
| [`docs/新增代码审查.md`](docs/新增代码审查.md) | 第三轮审查：新增页面/SSE/告警/脚本的复查结论与修复 |
| [`docs/代码审查报告-v2.md`](docs/代码审查报告-v2.md) | 第二轮独立全面审查（5 BLOCKER / 30 HIGH）与修复证据 |
| [`docs/代码审查报告.md`](docs/代码审查报告.md) | 第一轮审查：8 BLOCKER 与工程化问题清单 |
| [`eval/badcase_log.md`](eval/badcase_log.md) | bad case 归因、修复动作与回归结果 |
| [`finetune/README.md`](finetune/README.md) | 微调数据构造（对齐线上 prompt）、依赖版本、合并导出 |

---

## 十、已知限制（如实说明）

1. **生成慢**：本地 7B 推理模型端到端约 15~25s/题（CPU embedding），交互体验受限于算力，非检索链路问题。
2. **微调未实跑**：本机 torch 为 CPU 版、模型下载限速，训练脚本与数据已就绪但未完成训练与效果对比。
3. **评测为单轮**：当前 49 条为单轮问答；多轮追问与指代消解未覆盖。
4. **两套 RAG 并存**：RAGFlow 与 Dify 知识库内容相同但切块/索引独立，用于对照两种平台效果，不是生产级统一检索。
5. **SSE 为单 chunk 伪流式**：生成完成后一次推送，真流式需改造生成链路。
6. **无鉴权与限流**：服务默认只监听 127.0.0.1，容器/K8s 场景需在网关层加认证与限流。
7. **告警不推送**：未部署 Alertmanager，规则仅在 Prometheus UI / `/alerts` 可见。
8. **索引需随清洗结果重建**：`data/policies_clean/` 更新后需重新运行 `import_kb.py`，否则 RAGFlow 检索仍是旧内容。
9. **评测集与训练集的同题风险**：评测集有部分题与知识库 QA 同题，微调前后对比时应剔除同题或分组报告。

---

## 免责声明

本项目用于技术验证与学习，不构成财政、法律或审计意见；政策条款以官方最新发布为准。
示例政策文本来自公开渠道，仅用于检索与引用演示。

---

## License

[MIT](LICENSE) © 2026 look378
