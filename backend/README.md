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
python -X utf8 -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

已有测试使用模拟服务，不能代替真实 Redis / ChromaDB / 模型联调。`scripts/verify_*.py` 与在线检索脚本属于额外验证，使用前确认目标环境。

## 第一批：证据级检索评测

本批实现完成，真实模型效果尚未验证。旧 `scripts/evaluate_retrieval.py` 的参数、`evaluation/rag_cases.json` 的 12 条标题级样例保持原含义。新协议为 `rag-evidence-v1`，24 条开发用例位于 `evaluation/rag_evidence_cases.v1.json`，与旧标题指标分开使用。

所有命令从本目录执行。若使用本项目虚拟环境，将 `python` 换成 `.venv/Scripts/python`。Windows 安装依赖使用 `-X utf8`，避免含中文注释的 requirements 被按 GBK 解码。

```powershell
# 1. 从固定的 sample_knowledge.json 导出快照，先查看原文和 chunk_id，再人工标注
python -X utf8 scripts/evaluate_retrieval_evidence.py --export-snapshot

# 2. 默认重算仓库内固定模拟结果，不启动服务、不读 .env、不连接数据库或模型
python -X utf8 scripts/evaluate_retrieval_evidence.py

# 3. 运行相关验收；测试入口阻止 api.main 加载本机 .env
python -X utf8 -m pytest tests/test_retrieval_evidence.py tests/test_knowledge_base.py tests/test_hybrid_retrieval.py tests/test_tool_manager.py -q

# 4. 使用既有运行产物重新计算；替换 <run_id> 为实际目录名
python -X utf8 scripts/evaluate_retrieval_evidence.py --fixture data/eval/rag/<run_id>/executions.json
```

默认 fixture 是人工固定的模拟输出，包含空结果、重复、fallback、第三名命中、部分覆盖和一次故意模拟的执行错误。默认命令的预期退出码是 **1**，表示报告已生成且含执行错误；**0** 表示无执行错误（不表示全部命中），**2** 表示输入/协议/写入校验失败。模拟延迟的 0ms 是占位值，不能视为真实测量。

报告追加到被忽略的 `data/eval/rag/<run_id>/`，目录已存在则拒绝覆盖。每次保存 `snapshot.json`、`dataset.json`、可重算的 `executions.json` 和含逐条分数的 `report.json`。记录协议、知识内容哈希、数据集哈希、Git HEAD/分支/工作区状态、Top-K、UTC 起止时间、执行状态、返回片段和延迟；错误只保存固定错误码，不保存 Token、异常正文或未知资料正文。`--run-id` 可指定未使用过的目录名。

知识快照只来自 `data/demo_docs/sample_knowledge.json`，共 6 篇短文、6 个片段，没有扩大政策范围。入库和离线导出共用纯切片函数。Chroma 原存储 ID 保持不变，检索结果新增 `chunk_id = document_id:content_version:chunk_index`，内容版本是原文 SHA-256；同一语料重建稳定，内容变化会使旧证据 ID 失效。缺失身份/版本元数据的旧片段不会伪造 ID，需在独立示例环境中重新导入正确原文。

每个证据组表示一个需要覆盖的依据，组内 ID 为可替代片段；`source_quote` 保存人工核对的原文摘录。协议校验要求摘录存在于每个替代片段中，未知/过期标注 ID、缺失必需标注、重复用例、同源组跨 split 均拒绝。`turns` 预留多轮结构，但本批执行器只接受单轮。`required_facts`、`forbidden_claims` 只是预留标注，本批不评分回答。

| 指标 | 定义 |
| --- | --- |
| Hit@K | 前 K 个实际排名中至少覆盖一组证据为 1，否则为 0 |
| Evidence Recall@K | 已覆盖证据组数 / 标注的必需证据组数 |
| MRR@K | 第一个相关片段实际排名的倒数；未命中为 0 |

同组命中两个替代片段只算一组；一个片段可覆盖多组。重复片段不增加覆盖，也不删除排名位置后再补足 K。fallback 占据返回位置但不算证据。只有标注证据的 answerable/partial 用例进入正向指标；unanswerable 必须为 not_applicable，分数是 null。请求错误在 eligible 主指标中按未命中计，同时呈现全体执行错误率和成功执行子集；空分母为 null。返回的非 fallback 片段必须与快照的 ID、标题、正文匹配，否则记为 snapshot_mismatch，不能凭正确标题通过。

只有显式 `--online` 才请求现有默认 `/search`，鉴权只读取进程环境变量 `ZHIYING_EVAL_TOKEN`，不会加载 `.env` 或把 Token 写入报告。请自行将令牌安全注入进程环境，并指向独立、只导入本示例快照的 ServicePilot 环境：

```powershell
python -X utf8 scripts/evaluate_retrieval_evidence.py --online --base-url http://127.0.0.1:8000 --top-k 3
```

该模式可能调用查询改写与重排模型，并沿用现有缓存；第一批不绕过缓存，不新增策略开关。逐条保留服务返回的 cached/reranked 标记，但没有完整模型配置、阶段耗时或 token 对比。客户端只校验实际返回片段，不能证明远端整个数据库恰好等于快照；现有 `/search` 也不暴露底层 ToolResult.success，HTTP 200 空结果无法进一步区分内部失败与普通空召回。这些边界需在第二批的独立评测实例和执行轨迹中处理。本批未执行在线实验，未获得真实流水线质量基线。

## 数据边界

只有 `data/demo_docs/sample_knowledge.json` 是仓库内示例知识。数据库、会话、画像、日志、历史评测和本机 `.env` 不提交。development 空库可自动导入示例；production 不自动导入。工具不执行真实退款或创建工单。
