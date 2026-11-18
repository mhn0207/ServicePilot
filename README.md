# ServicePilot｜带工具调用与评测的企业客服 Agent

[![CI](https://github.com/mhn0207/ServicePilot/actions/workflows/ci.yml/badge.svg)](https://github.com/mhn0207/ServicePilot/actions/workflows/ci.yml)

ServicePilot 用 FastAPI、Vue、Redis 和 ChromaDB 处理技术排障与账单核验。单域请求由对应角色处理，明确的跨域请求并行调用技术与账单 Agent。系统记录工具参数和结果，用 Redis 会话回执处理重试。

## 一条请求的真实验收

> 登录时报 401，同时订单 ORD-20261024 疑似重复扣款，请分别说明如何处理。

1. **技术分支**在知识库检索 401 证据，给出检查 Token 有效期、认证方式和请求环境的步骤。
2. **账单分支**查 SQLite 沙盒订单。答复保留两条核实结果，`CHG-DEMO-01` 的 **199 元已入账**，`CHG-DEMO-02` 的 **199 元预授权冻结（未入账）**。
3. **重发请求**时保留同一会话、消息和 `client_request_id`。接口返回原回执，会话历史仍只有一问一答。

[复跑这条案例](backend/README.md#最短真实验收) · [体验工作台账单沙盒](backend/README.md#账单沙盒演示)

## 单 Agent 与多 Agent 的实测对照

两套 Docker 服务走实际 `POST /chat`，使用同一个 `deepseek-flash`、演示知识库和请求预算。用例包括单域、跨域、订单改口、证据不足和工具失败，每套 10 个用例、11 轮。

| 指标 | 单 Agent | 多 Agent |
| --- | ---: | ---: |
| 自动验收通过 | 10/11 | 11/11 |
| 复核后通过 | 11/11 | 11/11 |
| HTTP 延迟 P50 | 6.88 秒 | 6.35 秒 |
| HTTP 延迟 P95 | 10.51 秒 | 11.39 秒 |
| 供应商报告输入 / 输出 Token | 40,229 / 5,549 | 60,229 / 7,802 |

人工复核确认了单 Agent 答案中分开表述的币种、金额与状态。这轮多 Agent 的中位延迟较低，尾延迟和 Token 更高，复核后的通过数持平。系统在明确的跨域请求上启用协作。样本量为每组 11 轮，P95 对应该组最慢的一轮。

[中文用例与评分规则](backend/evaluation/corpora/chat-paired.zh.v2.json) · [完整配对实验与复核步骤](backend/README.md#中文聊天配对实验)

## 工程实现

- **会话与工具可靠性**。Redis 会话租约、原子提交与请求回执，配合 SQLite 订单唯一约束保护账单核验副作用。工具熔断、恢复探测和请求预算控制故障与超时。
- **可追溯检索**。Chroma 向量和词面 BM25 混合排序，知识片段按内容版本定位。知识变更后，Redis 共享版本使其他 API 进程识别旧缓存。
- **分层评测**。证据召回、检索策略消融、实际证据与标准证据的回答对照，以及 Agent 工具与多轮状态验收。

[后端架构与接口](backend/README.md#请求链路) · [评测入口](backend/README.md#评测入口)

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
