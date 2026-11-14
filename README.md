# ServicePilot｜基于 RAG 的多 Agent 企业服务咨询系统

[![CI](https://github.com/mhn0207/ServicePilot/actions/workflows/ci.yml/badge.svg)](https://github.com/mhn0207/ServicePilot/actions/workflows/ci.yml)

面向企业服务咨询的多 Agent AI 应用，覆盖技术故障、账单争议与订阅咨询，提供知识检索、排查建议和人工交接摘要。

个人实践项目，在 ZhiYing / EchoMind 相关代码基础上持续迭代，重点关注 Agent 编排、会话可靠性与 RAG 评测。

**技术栈：** Python 3.12 · FastAPI · Anthropic SDK · Redis · ChromaDB · MCP · Vue 3 · Docker Compose

## 核心能力

- **多 Agent 协作**：融合 LLM、向量匹配与关键词规则识别意图、抽取实体，路由至通用、技术和账单角色；支持复合问题并行处理、结果汇总与失败降级。
- **RAG 与会话管理**：支持向量与词面混合检索、查询改写、并行召回和模型重排；使用 Redis 保存工作记忆，ChromaDB 保存知识、历史摘要及画像。通过会话租约、请求回执与整轮事务提交处理并发和重试，支持实体更正与近期消息恢复。
- **工具与运行观察**：接入 Anthropic Tool Calling、标准 MCP 和 3 个可热加载 Skills，提供参数校验、角色权限、超时熔断、执行预算、工具轨迹及 Prometheus 指标；支持管理员查看知识片段并按内容版本删除文档。

业务工具提供字段核验、金额比较和处理建议，不连接真实订单、支付、退款或工单系统。

## 设计取舍

- **确定性路由**：主 Agent 与协作 Agent 由意图、关键词和实体打分决定，不交给模型自由选择。行为可回归测试、可解释，代价是覆盖面取决于词表。
- **统一的执行预算**：一次聊天的全部时间参数集中在 `core/request_budget.py` 的 `BudgetPolicy`，导入时校验“意图 + 主 Agent + 兜底 ≤ 请求预算”等约束；预算不足时跳过可选步骤而不是整体超时。
- **兜底复用检索**：专属 Agent 失败后由通用 Agent 接手，直接复用本次请求已取得的检索结果；已尝试过有副作用的工具则不再兜底。
- **离线评测用固定响应**：回归与协议验收不调用真实模型，结果可复现且零成本；它验证的是链路和协议，不代表回答质量。
- **基线只能显式晋升**：评测完成不会自动覆盖基线，避免退化被悄悄吸收。

## 请求流程

```text
Vue 工作台 → FastAPI 鉴权 → 会话锁与回执检查
           → 读取记忆 → 意图识别与实体合并 → Agent 路由
           → 工具与知识检索 → 回复汇总
           → Redis 提交消息、状态和回执 → 返回
```

## 评测设计

| 层次 | 关注点 |
| --- | --- |
| 证据级检索 | 版本化片段 ID、知识快照和证据组标注；计算 Hit@K、Evidence Recall@K、MRR@K |
| 独立策略对比 | 对比向量、混合、查询改写与模型重排；记录配置、实际执行路径、耗时和用量 |
| 成对回答质量 | 对照实际检索上下文与标注证据上下文，分别检查事实覆盖、禁止结论和证据支持 |
| Agent 离线验收 | 使用真实编排器与固定模型响应，检查多轮实体、角色路由、工具调用及返回证据 |

数据集、标注与 fixture 随代码保存；运行报告记录数据和评分身份。离线模拟用于回归与协议验收，真实模型效果另行评估。目前真实实验规模较小，尚未建立稳定的回答质量或多 Agent 收益结论。数据划分、已有实验结果及命令见 [后端说明](backend/README.md)。

## 快速开始

需要 Docker / Docker Compose，以及兼容 Anthropic Messages 和工具调用的模型服务。从仓库根目录执行 PowerShell：

```powershell
if (!(Test-Path backend/.env)) { Copy-Item backend/.env.example backend/.env }
# 编辑 backend/.env，填写模型 Key、模型名与服务地址。
Set-Location backend

docker compose -p servicepilot-local -f docker-compose.yml -f docker-compose.web.yml up -d --build --wait --wait-timeout 180
```

| 入口 | 默认地址 |
| --- | --- |
| 工作台 | http://localhost:5174 |
| API 文档 | http://localhost:8000/docs |
| 存储健康检查 | http://localhost:8000/health |
| Prometheus | http://localhost:9090 |

默认使用本机 development 配置、公开演示知识并关闭外部 MCP。首次构建需要联网安装依赖与下载 embedding 模型；已有服务占用端口时需调整配置。对外部署时配置 production 与访问 Token；模型 Key 仅保存在后端。健康检查检测存储就绪状态。

在工作台可依次体验：401 故障咨询、订单与金额更正、知识导入与片段查看，以及 **RAG · Agent 离线验收**。聊天和知识搜索可能产生模型调用费用，离线验收使用固定响应；管理员功能需要对应权限。

日常管理沿用同一项目名和两份 Compose 文件。修改源码后重新执行构建命令；日常停止或重启不要使用 `down -v`，以免删除数据卷。

## 开发与验证

```bash
# backend/：使用 Python 3.12 和本项目虚拟环境
python -X utf8 -m pip install -r requirements-dev.txt
python -X utf8 -m pytest tests -q

# frontend/：Node 22
npm ci
npm test
npm run build
```

以上命令在 PowerShell、macOS 和 Linux 下相同。推送到 `main` 及每个 Pull Request 会由 GitHub Actions 执行同一组检查（`.github/workflows/ci.yml`）；测试不读取 `.env`，不需要模型 Key。

测试覆盖编排、会话重试、实体更正、知识维护和评测协议；Docker 验证覆盖服务启动、HTTP 链路、存储故障与重启保留。配置及验证范围见 [后端说明](backend/README.md)，前端开发见 [工作台说明](frontend/README.md)。

```text
backend/
  agents/                Agent 与工具契约
  api/                   FastAPI 接口
  core/                  意图、会话状态、权限与执行预算
  memory/                Redis / ChromaDB 记忆管理
  tooling/               知识库、工具管理与 MCP
  evaluation/            评测协议、冻结语料与 fixture
  scripts/               评测命令入口
  skills/                业务规则
  tests/                 自动化测试
frontend/                Vue 工作台与接口适配
```

真实配置、数据库、会话、画像与运行报告不纳入版本管理。接口保留 `/api/zhiying`、`ZHIYING_*` 等现有协议标识。

## 已知限制

- 业务工具只做字段核验和建议，不连接真实订单、支付或工单系统；人工升级只生成交接摘要。
- 演示知识库只有 6 篇短文，评测语料的候选片段也很少（中文集各 12 片）。现有检索题在简单词面匹配下就接近满分，因此不同检索策略之间还拉不开差距。
- Agent 离线验收仅 7 个用例且使用固定响应，尚无多 Agent 相对单 Agent 的收益结论。
- 强制检索的判定依赖有限的主题词表，换一种说法可能漏判。

## 来源与授权

维护者：mhn。本版本由个人 ZhiYing / ZhiYingFrontend 工作区整理，后端原 Git 来源为 [Biscuit-AI531/EchoMind](https://github.com/Biscuit-AI531/EchoMind)，整理前 HEAD 为 `b1adff4`。当前 Git 从个人整理基线继续维护，来源代码与后续迭代共同构成本项目。

整理时未在源工作区发现项目级 LICENSE / NOTICE，未擅自新增授权许可证；公开分发前需核实上游授权。依赖软件按各自许可证使用。
