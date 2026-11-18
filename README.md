# ServicePilot｜基于混合检索与分布式会话治理的多 Agent 企业服务咨询系统

[![CI](https://github.com/mhn0207/ServicePilot/actions/workflows/ci.yml/badge.svg)](https://github.com/mhn0207/ServicePilot/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Async-009688?logo=fastapi&logoColor=white)
![Vue 3](https://img.shields.io/badge/Vue-3.x-4FC08D?logo=vuedotjs&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-Lease_%26_MULTI-DC382D?logo=redis&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Hybrid_RAG-FF6F00)
![Tests](https://img.shields.io/badge/Tests-530%2B_Passed-success)

**ServicePilot** 是面向企业级技术排障、账单核验与订阅咨询场景的多 Agent 协同与 RAG 工程系统。项目围绕 **“如何让大模型在企业咨询场景中既可控、可追溯，又具备高并发会话一致性与可量化评测基准”** 展开设计，涵盖多角色编排、混合检索流水线、分布式会话事务、阶段级预算熔断及四层离线/在线评测体系。

---

## 核心工程亮点 (Key Engineering Highlights)

1. **防幻觉与幂等保护的多 Agent 编排 (`AgentOrchestrator`)**
   - **三路融合意图识别**：并行融合 LLM 语义理解（0.7）、Embedding 向量匹配（0.2）与零延迟正则规则（0.1），支持粗粒度意图组（Group）聚合投票与细粒度规则精化（Pattern Refinement），结合上下文感知实体抽取（自动剥离金额/电话干扰，精准识别错误码、订单号与否定撤回金额）。
   - **结构化主从路由与在线动态降权**：区分单角色、主从并行协作（`run_parallel` + `ResponseComposer`）与确定性人工交接（`EscalationAgent` 零 LLM 兜底）；结合 5 分钟滑窗成功率、平均延迟与监控惩罚项（`monitor_penalty`）动态调整路由评分并支持单请求熔断探测恢复（Recovery Probe）。
   - **副作用防重放与检索缓存继承**：各并行分支隔离工具尝试状态并自动汇总；非幂等工具执行前强制持久化 `tool_outcome_unknown` 事务标记；主 Agent 失败降级至 `GeneralAgent` 时自动复用当前请求已完成的 RAG 检索结果（`reuse_retrieval`），避免重复消耗检索预算。

2. **分布式会话租约与三层记忆架构 (`MemoryManager`)**
   - **会话级并发串行化与自动续租**：通过进程内 `KeyedLocks` 与 Redis 分布式租约（`redis_lease`，后台以 `lease_s / 3` 周期自动续租，失锁立即取消当前协程）保证同一会话严格串行。
   - **整轮原子提交与幂等回执**：用户消息、助手回复、多轮槽位状态（`conversation_state`）与请求回执（`turn_receipt`）通过 Redis `MULTI/EXEC` 事务原子落盘；支持基于 `client_request_id` 的安全重试与冲突拦截（409）。
   - **三层记忆与乐观锁防丢消息**：工作记忆（Redis List）超过阈值时后台异步触发 LLM 摘要压缩，采用 `WATCH/MULTI` 乐观锁校验快照，确保压缩期间新到达的并发消息绝不被覆盖；历史对话归档至 ChromaDB 情景记忆（幂等 Content-Hash Upsert），长效偏好由 `ProfileUpdater` 合并节流更新至用户画像库。

3. **带版本溯源的混合 RAG 引擎与全链路预算控制 (`KnowledgeBase` & `BudgetPolicy`)**
   - **混合召回 + RRF 融合 + 模型重排**：结合 ChromaDB 向量召回与词面（BM25 / Terms）独立召回，支持 Query 改写、多路并发召回与 LLM Rerank；使用双层版本号与 In-Flight Future 复用抵御缓存击穿。
   - **内容寻址与乐观并发文档维护**：切片自动生成包含文档身份、正文 SHA-256 版本与片段序号的 `chunk_id`；管理端支持预览并按 `expected_version` 原子删除整篇文档（版本冲突返回 409），跨知识变更自动失效旧检索响应。
   - **集中式 `BudgetPolicy` 与提示词缓存优化**：单次对话硬超时 50s、总预算 45s，集中校验“意图 + 主 Agent + 兜底 ≤ 请求预算”约束，细粒度切分主 Agent（28s）、可选 RAG（剩余 ≥18s 才开启改写/重排）与降级兜底（剩余 ≥11s）；System Prompt 按“工具 → 角色契约 → Skills”固定前缀排列，支持显式缓存断点。

4. **四层可复现评测与代码溯源体系 (`evaluation/`)**
   - 覆盖 **证据级检索（Hit@K / Evidence Recall@K / MRR@K）**、**独立策略消融对比（A 向量 / B 混合 RRF / C 改写融合 / D 模型重排）**、**成对回答质量（`retrieved_context` vs `oracle_context` 事实覆盖与 Grounding 校验）** 与 **Agent 离线编排验收（`agent-dev-v1`）**，所有报告绑定源码 SHA-256 清单与知识快照哈希，且基线仅允许显式晋升。

---

## 系统架构与请求生命周期 (System Architecture)

```text
+-----------------------------------------------------------------------------------+
|                              Vue 3 工作台 (Frontend)                               |
|   会话恢复与幂等重试  |  知识库版本化维护  |  工具调用轨迹审查  |  RAG 离线验收看板   |
+-----------------------------------------+-----------------------------------------+
                                          | HTTP / Bearer Token (/api/servicepilot)
                                          v
+-----------------------------------------------------------------------------------+
|                        FastAPI 网关与执行治理层 (API Layer)                         |
|  Bearer / Metrics 鉴权隔离  |  50s 硬超时 & 45s 阶段预算  |  Prometheus 指标采集    |
+-------------------+---------------------+---------------------+-------------------+
                    |                     |                     |
                    v                     v                     v
+-------------------------+   +-----------------------+   +-------------------------+
|   分布式会话与三层记忆    |   |   三路融合意图与槽位   |   |   多 Agent 路由与编排器  |
|  - KeyedLocks + Redis锁 |   |  - LLM (0.7)          |   |  - General / Technical  |
|  - turn_receipt 幂等回执 |   |  - Embedding (0.2)    |   |  - Billing / Escalation |
|  - WATCH/MULTI 后台压缩  |   |  - Pattern (0.1)      |   |  - 滑窗评分与熔断探测     |
|  - 情景记忆 & 用户画像   |   |  - 实体提取与否定撤回  |   |  - 并行协作 & Composer  |
+-------------------------+   +-----------------------+   +------------+------------+
                                                                       |
                                                                       v
+-----------------------------------------------------------------------------------+
|                      工具生态与混合检索层 (Tooling & Hybrid RAG)                    |
|  - 混合检索：ChromaDB 向量召回 + 词面召回 + RRF 融合 + 查询改写 + LLM Rerank        |
|  - 领域工具：故障码诊断、排障计划、账单字段核验、金额比对、人工交接摘要              |
|  - 标准协议：内置 FastMCP HTTP Server (/mcp/) + 外部 MCP Client + 热加载 Skills     |
+-----------------------------------------------------------------------------------+
```

> **边界与已知限制**：
> - 内置业务工具提供字段核验、金额比对、排障计划和交接摘要。可选 SQLite 账单沙盒提供合成流水查询和持久化退款核验单，实际支付和退款均由外部业务系统处理。见 [账单沙盒演示](backend/README.md#账单沙盒演示)。
> - 离线评测使用固定响应以保证零成本可复现，验证链路与协议契约；冻结语料包含 `suite.v2`、`public_zh.v1`、`public_zh_hard.v1` 与 `service_zh.v1`，大规模候选池下的真实模型收益与强制检索词表覆盖仍持续演进。

---

## 四层评测体系与实验基准 (Evaluation Framework)

| 评测层次 | 核心关注点与指标 | CLI 入口 (`backend/`) |
| --- | --- | --- |
| **1. 证据级检索** | 版本化片段 `chunk_id`、知识快照与多证据组标注；计算 `Hit@K`、`Evidence Recall@K`、`MRR@K` | `python -X utf8 scripts/evaluate_retrieval_evidence.py` |
| **2. 独立策略消融** | 严格控制变量对比 **A**（纯向量）、**B**（向量+词面 RRF）、**C**（多查询改写融合）、**D**（C+模型重排） | `python -X utf8 scripts/evaluate_retrieval_strategies.py --validate-suite` |
| **3. 成对回答质量** | 对照 `retrieved_context` 与诊断对照 `oracle_context`，分离检索缺失与生成幻觉，逐项核验事实与禁项 | `python -X utf8 scripts/evaluate_answer_quality.py --validate-annotations` |
| **4. Agent 离线验收** | 驱动真实 `AgentOrchestrator` 与工具循环（固定模型响应 + synthetic 检索），验证多轮槽位更正与协作路由 | `python -X utf8 scripts/evaluate_agents.py --validate-annotations` |

详细的实验协议、冻结语料划分及本地实验记录见 **[后端文档](backend/README.md)**。

---

## 快速开始 (Quick Start)

需要安装 Docker / Docker Compose，并准备兼容 Anthropic Messages API（支持 Tool Calling）的模型服务凭据。在仓库根目录打开 PowerShell：

```powershell
if (!(Test-Path backend/.env)) { Copy-Item backend/.env.example backend/.env }
# 编辑 backend/.env，填入 ANTHROPIC_API_KEY、ANTHROPIC_MODEL 与 ANTHROPIC_BASE_URL
Set-Location backend

docker compose -p servicepilot-local -f docker-compose.yml -f docker-compose.web.yml up -d --build --wait --wait-timeout 180
```

| 服务入口 | 默认地址 | 说明 |
| --- | --- | --- |
| **Web 工作台** | http://localhost:5174 | 聊天、知识文档管理、工具轨迹审查、RAG 离线验收 |
| **Swagger API 文档** | http://localhost:8000/docs | FastAPI OpenAPI 交互式文档 |
| **存储就绪探针** | http://localhost:8000/health | 2 秒预算内并发探测 Redis 与 ChromaDB 集合状态 |
| **Prometheus** | http://localhost:9090 | 指标采集与监控面板 |

- **默认开发配置**：默认启用 `SERVICEPILOT_ENV=development`（兼容 `ZHIYING_ENV`），空库启动时自动导入 `data/demo_docs/sample_knowledge.json` 演示政策，并默认关闭外部 MCP 连接（`MCP_EXTERNAL_ENABLED=false`）。
- **工作台体验路径**：可依次体验 **401 登录故障排障**、**订单号与金额否定更正**、**复合问题并行协同（登录报错 + 重复扣款）**、**知识导入与按版本删除**，以及 **RAG · Agent 离线验收**。
- **日常启停**：停止或重启容器时请勿附加 `down -v`，以免清除本地持久化的 Redis / ChromaDB 数据卷。

---

## 本地开发与自动化测试 (Development & Testing)

```bash
# 1. 后端测试（在 backend/ 目录下使用 Python 3.12 虚拟环境）
python -X utf8 -m pip install -r requirements-dev.txt
python -X utf8 -m pytest tests -q

# 2. 前端测试与构建（在 frontend/ 目录下，Node 22）
npm ci
npm test
npm run build
```

以上命令在 PowerShell、macOS 和 Linux 下相同。推送到 `main` 及每个 Pull Request 会由 GitHub Actions 执行同一组检查（`.github/workflows/ci.yml`）；测试不读取 `.env`，不需要模型 Key。配置及验证范围见 [后端说明](backend/README.md)，前端开发见 [工作台说明](frontend/README.md)。

---

## 目录结构 (Repository Layout)

```text
backend/
  agents/                多 Agent 编排器、角色契约与领域工具定义
  api/                   FastAPI 路由、鉴权中间件与生命周期管理
  core/                  三路意图识别、多轮会话状态、分布式锁与请求阶段预算
  memory/                Redis 工作记忆与 ChromaDB 情景记忆/用户画像管理
  tooling/               混合检索知识库、文档版本维护、MCP Client/Server
  evaluation/            四层评测协议、源码溯源打包、冻结语料与离线 Fixture
  scripts/               检索证据、策略消融、回答质量与 Agent 评测 CLI
  skills/                可热加载的领域业务规则 (Markdown Skills)
  tests/                 后端单元、集成与回归测试套件
frontend/
  src/                   Vue 3 工作台、请求代次状态机与 API 客户端适配
  tests/                 前端接口契约、会话恢复与并发状态隔离测试
```

> **兼容性说明**：系统全面支持 `SERVICEPILOT_*` 环境变量、`/api/servicepilot` 网关前缀与 `servicepilot://` MCP 资源标识，同时向下兼容历史 `ZHIYING_*` 与 `/api/zhiying` 协议标识。

---

## 演进说明 (Project Evolution)

维护者：mhn。本项目为个人独立维护与演进的企业服务多 Agent 工程实践项目，早期基础框架参考自开源项目 [Biscuit-AI531/EchoMind](https://github.com/Biscuit-AI531/EchoMind)（初始整理基线 `b1adff4`），并在个人工作区（原内部代号 `ZhiYing`）上完成了多 Agent 角色契约重构、分布式会话租约与整轮事务提交、全链路阶段预算、混合检索版本化管理及四层离线/在线评测体系的深度改造，统一命名为 **ServicePilot**。整理时未在源工作区发现项目级 LICENSE / NOTICE，未擅自新增授权许可证；依赖组件按各自开源许可证使用。
