# ServicePilot｜带工具调用与评测的企业客服 Agent

[![CI](https://github.com/mhn0207/ServicePilot/actions/workflows/ci.yml/badge.svg)](https://github.com/mhn0207/ServicePilot/actions/workflows/ci.yml)

ServicePilot 用 FastAPI、Vue、Redis 和 ChromaDB 处理技术排障与账单核验。单域请求由对应角色处理，明确的跨域请求并行调用技术与账单 Agent。系统记录工具参数和结果，用 Redis 会话回执处理重试。

## 一条请求的真实验收

> 登录时报 401，同时订单 ORD-20261024 疑似重复扣款，请分别说明如何处理。

1. **技术分支**在知识库检索 401 证据，给出检查 Token 有效期、认证方式和请求环境的步骤。
2. **账单分支**查 SQLite 沙盒订单。答复保留两条核实结果，`CHG-DEMO-01` 的 **199 元已入账**，`CHG-DEMO-02` 的 **199 元预授权冻结（未入账）**。
3. **重发请求**时保留同一会话、消息和 `client_request_id`。接口返回原回执，会话历史仍只有一问一答。

[复跑这条案例](backend/README.md#最短真实验收) · [体验工作台账单沙盒](backend/README.md#账单沙盒演示)

## 评测结果

| 评测 | 样本与运行方式 | 主要结果 |
| --- | --- | --- |
| [真实 API 并发压测](backend/README.md#真实-api-并发压测) | 经 Nginx 请求单个 FastAPI worker，模型为 `deepseek-flash`，1／4／8 并发共 72 轮 | 业务成功 **72/72**，预期工具断言 **108/108**。8 并发的 P95 **5.88 秒**，吞吐 **1.50 请求/秒** |
| [词面倒排索引对照](backend/README.md#词面倒排索引对照) | 真实 ONNX 和 ChromaDB HTTP，5,000 片合成扩容池，4 并发 | 证据 Recall@3 **71.74% → 93.48%**，检索 P95 **1,134 → 604 ms**。120 和 1,000 片的同并发测试中，索引提高召回并增加 P95 |
| [证据级检索](backend/README.md#检索证据) | 24 条中文固定结果，22 条有标注证据 | Hit@3 **27.27%**，证据 Recall@3 **25.00%**，MRR@3 **0.242**。固定结果故意包含漏召回和 1 次执行错误 |
| [检索策略消融](backend/README.md#真实检索策略实测) | 20 题冻结验证集，真实 MiniLM ONNX，改写和重排调用 `deepseek-flash` | 纯向量 / 混合召回 MRR@3 **0.944 / 0.972**。重排优化后完整执行 **5/20 → 20/20**，P95 **20.92 秒 → 1.93 秒** |
| [中文检索与候选池基准](backend/README.md#本地检索记录) | 短摘要、扩展池与 ChromaDB 回环 HTTP 基准 | 120 片扩展池 dev 中混合召回比纯向量多找回 **2/16** 题的证据。[历史词面候选基准](backend/README.md#业务词面候选基准)记录扩池后的截断和耗时 |
| [成对回答与 Judge](backend/README.md#回答与系统验证) | 40 题英语 dev 固定回放，另取 2 道验证题调用真实模型 | 固定回放的事实覆盖两分支均为 **77/84**。真实复测 **4/4** 个答案生成，**8/8** 次 Judge 调用通过协议校验，**1/4** 个分支仍有断言支持度未决 |
| [Agent 离线验收](backend/README.md#agent-与工作台) | 9 轮真实编排器，固定模型响应与模拟检索 | 角色覆盖 **9/9**，必需工具调用 **6/7**，返回证据 **4/7** |
| [意图路由短路](backend/README.md#意图路由基准) | 19 个用例，真实本地向量与固定延迟模型替身 | 启用短路后每轮意图模型调用 **19 → 11**，路由错误保持 **0/19** |
| [单 Agent 与多 Agent](backend/README.md#2026-11-25-docker-实测) | Docker 中用 `deepseek-flash` 调用真实 `POST /chat`，两组各 10 个用例、11 轮 | 复核后 **10/11 / 11/11**，工具断言均为 **15/15**。本次多 Agent 中位延迟从 **7.37 降至 5.51 秒**，供应商报告的输入和输出 Token 均更高 |
| [Docker 与服务可靠性](backend/README.md#2026-11-24-docker-实测) | 真实 Redis、ChromaDB、SQLite 和 Nginx | 相同请求 ID 的 8 个并发请求只提交 **1 轮**，换会话再次确认只保留 **1 张核验单**。跨进程知识缓存 **5/5**，Nginx 上游换 IP 恢复 **5/5** |

结果来自维护者的本地报告。各项使用不同语料和运行方式，分母与边界见[实验明细](backend/README.md#数据与已有实验结论)。仓库评测命令生成的报告默认保存在 `backend/data/eval/`。

## 工程实现

- **会话与工具可靠性**。Redis 会话租约、原子提交与请求回执，配合 SQLite 订单唯一约束保护账单核验副作用。工具熔断、恢复探测和请求预算控制故障与超时。
- **可追溯检索**。Chroma 向量和词面 BM25 混合排序，知识片段按内容版本定位。知识变更后，Redis 共享版本使其他 API 进程识别旧缓存。
- **分层评测**。证据召回、检索策略消融、实际证据与标准证据的回答对照，以及 Agent 工具与多轮状态验收。

[后端架构与接口](backend/README.md#请求链路) · [评测入口](backend/README.md#评测入口)

## 快速开始

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
| **统一网关** | http://localhost | Nginx 提供工作台与 `/api/servicepilot/`，兼容 `/api/zhiying/` |
| **Swagger API 文档** | http://localhost:8000/docs | FastAPI OpenAPI 交互式文档 |
| **存储就绪探针** | http://localhost:8000/health | 2 秒预算内并发探测 Redis 与 ChromaDB 集合状态 |
| **Prometheus** | http://localhost:9090 | 指标采集与监控面板 |

- **本地演示**。示例配置启用 `SERVICEPILOT_ENV=development`，空库会导入演示政策，外部 MCP 默认关闭。复合订单案例另需[启用账单沙盒](backend/README.md#账单沙盒演示)。
- **日常启停**。沿用上面的项目名和两个 Compose 文件，执行 `stop` 或 `restart` 时去掉 `up` 专用选项，即可保留本地数据卷。开启 API 鉴权后，按[采集配置](backend/README.md#prometheus-采集身份)设置独立指标令牌。

---

## 本地开发与自动化测试

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

## 目录结构

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
