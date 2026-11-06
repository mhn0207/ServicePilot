# ServicePilot 后端

Python 3.12 / FastAPI / Redis / ChromaDB / Anthropic SDK / MCP。

整体架构、部署入口、来源说明和四批次 RAG 评测计划见[根 README](../README.md)。

## 配置与启动

在本目录复制 `.env.example` 为 `.env`，填写 `ANTHROPIC_API_KEY`、`ANTHROPIC_MODEL` 和兼容 Anthropic Messages 的 `ANTHROPIC_BASE_URL`。真实凭据不得提交。`ZHIYING_ENV=development` 仅用于本机演示；production 要配置 `ZHIYING_API_TOKENS` 并使用独立凭据。

```powershell
docker compose -p servicepilot-local -f docker-compose.yml -f docker-compose.web.yml up -d --build
```

聊天 API 为 `POST /chat`，重试沿用 `client_request_id`；检索为 `POST /search?query=...&top_k=3`。`/health` 只验证应用与相关依赖状态，不能证明模型 Key、额度或模型名称可用。

`/eval/run` 提交现有意图与对话评测，返回 202 任务；`/eval/jobs/{job_id}` 查询进度。当前对话质量评测范围是 Orchestrator，不代表完整 HTTP 链路评测。

`/monitor` 查看运行统计，`/metrics` 提供 Prometheus 指标，`/trace/tool/{request_id}` 查询工具轨迹。`/skills/reload` 重载 Markdown 或 JSON 业务规则。外部 MCP 默认关闭，可按需配置 `config/mcp_servers.json` 并启用 `MCP_EXTERNAL_ENABLED`。

## 测试

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

已有测试使用模拟服务，不能代替真实 Redis / ChromaDB / 模型联调。`scripts/verify_*.py` 与在线检索脚本属于额外验证，使用前确认目标环境。

## 数据边界

只有 `data/demo_docs/sample_knowledge.json` 是仓库内示例知识。数据库、会话、画像、日志、历史评测和本机 `.env` 不提交。development 空库可自动导入示例；production 不自动导入。工具不执行真实退款或创建工单。
