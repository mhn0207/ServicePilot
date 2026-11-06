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

## 第二批：独立检索策略实验

实现和离线验收完成，真实收益待实验。没有回答生成/Judge、Agent 多轮评测或新生产 API。第一批文件、默认命令、原 12/24 条回归用例及其故意退出 1 的 fixture 均保留。

静态资料在 `evaluation/corpora/`：`service_rules.v2.json` 原文、`snapshot.v2.json` 完整快照、`sources.v2.json` 合成声明与原始分节、`cases.v2.json` 静态证据标注、`suite.v2.json` 分组/冻结清单。24 篇英语合成规则用既有 500 字切片得到 72 片；每篇三个不同条件或流程片段，不靠重复填充。文档涉及认证/权限/租户、错误处理、支付开通、冻结/扣款、续费/退款、发票/交接，以及身份生命周期、争议、汇款、区域迁移等。无外部转载或私人数据，不进入线上演示 seed；这是英语检索实验，不能推断中文效果。

40 条 dev（36 eligible，4 无答案）、20 条 validation（18 eligible，2 无答案），另含部分可答与跨文档证据。分组按问题来源及政策族整组划分，复合问题与相关来源放在同一 split。validation 使用新增业务场景，没有把 v1 看过的问法改写为验证数据。先查看完整快照再人工标注，数据在实验前冻结。修改原文、分组或标注必须重新复核并另立版本，不能只刷新哈希。

| 策略 | 实际步骤 |
| --- | --- |
| A | 仅向量召回 → 最终 Top-K |
| B | 向量 + 词面召回 → 名次融合 → 最终 Top-K |
| C | 原查询及改写查询分别混合召回 → 跨查询名次融合 → 最终 Top-K |
| D | C 的实际候选 → 模型重排 → 最终 Top-K |

默认 `top_k=3`、`candidate_k=20`、`vector_k=20`、`lexical_k=20`、最多两个额外改写，`cache_mode=bypass`。预算须满足 `top_k <= candidate_k <= vector_k`，混合策略还需 `candidate_k <= lexical_k`。向量/词面各按分支预算取候选；单查询融合后截断到 candidate_k，跨查询联合池最多为查询数乘 candidate_k。最终所有策略返回相同 Top-K，候选不足则保留实际数量及原因。C/D 多出的查询和候选是实际额外成本。

词面分词与 BM25 标题加权排序由 `tooling/retrieval_ranking.py` 供生产和评测共用，线上默认搜索行为保持兼容。评测将分支名次用 RRF（`1/(60+rank)`）等权融合，再对不同查询结果做相同名次融合，不相加不可比的原始相似度。各阶段同分按 chunk_id 排序。模型重排只看到查询、候选索引、标题前 160 字和正文前 400 字，ID 映射保留在程序中；有效索引去重、遗漏补齐，越界或输出截断则降级到原排序。

所有命令从本目录执行，`--run-id` 指定的目录必须未使用。默认单策略 B、dev、模拟依赖；矩阵也是 dev。

```powershell
# 原第一批回归：预期退出 1，故意保留一个 fixture_error
.venv/Scripts/python -X utf8 scripts/evaluate_retrieval_evidence.py

# 扩充语料：导出快照，再核对静态标注/整组划分/冻结身份
.venv/Scripts/python -X utf8 scripts/evaluate_retrieval_strategies.py --suite evaluation/corpora/suite.v2.json --corpus evaluation/corpora/service_rules.v2.json --export-snapshot
.venv/Scripts/python -X utf8 scripts/evaluate_retrieval_strategies.py --validate-suite

# 模拟依赖单策略和 dev 矩阵；无模型加载或网络调用
.venv/Scripts/python -X utf8 scripts/evaluate_retrieval_strategies.py --strategy A --run-id example-dev-A
.venv/Scripts/python -X utf8 scripts/evaluate_retrieval_strategies.py --matrix --cache-mode bypass --run-id example-dev-matrix

# 重放矩阵中的单策略记录；没有新的检索/模型调用
.venv/Scripts/python -X utf8 scripts/evaluate_retrieval_strategies.py --fixture data/eval/rag/experiments/example-dev-matrix/D --run-id example-replay

# 显式选两个已完成、身份兼容报告进行比较；不覆盖/晋升基线
.venv/Scripts/python -X utf8 scripts/evaluate_retrieval_strategies.py --compare data/eval/rag/experiments/example-dev-matrix/A/report.json data/eval/rag/experiments/example-dev-matrix/B/report.json --run-id example-comparison

# 显式验证候选，附固定 A/B；D 只是模拟验收示例，不代表获胜方案
.venv/Scripts/python -X utf8 scripts/evaluate_retrieval_strategies.py --split validation --candidate D --with-baselines --run-id example-validation-D

# 相关测试及后端完整回归（全部离线，不读真实 .env）
.venv/Scripts/python -X utf8 -m pytest tests/test_retrieval_strategies.py tests/test_retrieval_evidence.py tests/test_hybrid_retrieval.py tests/test_knowledge_base.py tests/test_tool_manager.py -q
.venv/Scripts/python -X utf8 -m pytest tests -q
```

模拟向量是 64 维特征哈希 + cosine，用于执行召回代码；模拟改写是固定查询变换、模拟重排用词面排序。它们不是语义 embedding 或真实模型质量。`synthetic`、`real` 和 `fixture` 在报告模式中区分；重放保留原数据的模式并另记 `observation=replayed`，其耗时/用量来自原记录，不是重放耗时。不可把模拟与真实报告混合比较。

真实入口仅供另行明确授权后执行，本批未运行以下命令。只读取显式进程环境，不加载 `.env`：C/D 需要 `SERVICEPILOT_EVAL_API_KEY`、`SERVICEPILOT_EVAL_MODEL`，可选 `SERVICEPILOT_EVAL_BASE_URL`；通过安全方式注入，不把密钥写进命令或报告。向量模式须提供已缓存 all-MiniLM-L6-v2 目录（包含 `onnx/` 的全部模型文件）；缺文件会拒绝，下载入口被禁止，也不会切换模型。

```powershell
# 将路径替换为已准备好的评测模型目录；以下命令未执行
.venv/Scripts/python -X utf8 scripts/evaluate_retrieval_strategies.py --mode real --strategy A --onnx-cache 'D:/models/eval-minilm' --run-id real-dev-A
.venv/Scripts/python -X utf8 scripts/evaluate_retrieval_strategies.py --mode real --strategy B --onnx-cache 'D:/models/eval-minilm' --run-id real-dev-B
.venv/Scripts/python -X utf8 scripts/evaluate_retrieval_strategies.py --mode real --strategy C --onnx-cache 'D:/models/eval-minilm' --enable-provider --run-id real-dev-C
.venv/Scripts/python -X utf8 scripts/evaluate_retrieval_strategies.py --mode real --strategy D --onnx-cache 'D:/models/eval-minilm' --enable-provider --run-id real-dev-D
```

真实工厂只创建新的内存 Chroma collection，显式注入本地 embedding；使用全库 get 校验 ID、正文、标题、数量与冻结快照后才评测，结束删除自己的 collection。所有同步 Chroma 操作在线程中执行。没有默认 KnowledgeBase 初始化、远端连接/本地回退、自动 seed、用户会话或线上 ToolManager；模型直接使用独立 SDK 客户端，避开会更新模块全局统计的生产调用封装。SDK 内部重试关闭，统计是实际逻辑调用；未返回的 token 保留 null，没有调用才汇总为 0。费用因没有核实价格保留 null。

两层结果缓存只存在于评测实例内。bypass 不读不写；use 的键包含规范化策略、知识哈希、依赖身份、查询和相关上下文，配置或知识变更不复用旧项。降级结果不写流水线缓存；缓存命中的步骤标记 skipped。单次 CLI 为冷实例，不提供跨进程缓存或独立缓存性能实验；测试显式重复查询验证两层行为。索引构建/模型加载时间在 setup 单列，不当作结果缓存预热。

输出追加到被忽略的 `data/eval/rag/experiments/<run_id>/`。每个策略有完整快照、完整/过滤数据集、suite、执行结果、轨迹和 `report.json`；矩阵另有比较与完成清单。实验协议 `rag-experiment-v2` 包装第一批 `rag-evidence-v1` 评分结果，不改默认流水线报告。保存完整 `source_dataset_hash`、过滤后 `evaluated_dataset_hash`、case_set、知识/冻结哈希和每次 run/config 身份。比较要求知识、标注、split、Top-K、模式、embedding 与非研究预算一致；策略开关本身允许不同。缺关键样本/未完成/身份不匹配时拒绝，有执行错误时不判回归通过；基线不会自动晋升。

主指标保留执行失败的 eligible 分母，另报成功 eligible 子集。共同成功子集给出明确 case_id 与两个分母（全部共同成功数、其中 eligible 数）。逐例列出 Recall 主判据的胜/平/负和 Hit/Recall/MRR 变化及退化证据。每例保存计划配置、实际步骤、改写列表、分支预算结果、去重/截断数量、降级原因、阶段耗时、缓存读写和实际模型观测。合法空结果、底层请求失败、改写失败回原查询、候选不足跳过重排、重排异常降级分别记录。完整执行率排除失败、降级及流水线缓存命中；无需重排的合法跳过可算完成。

P50/P95 使用 nearest rank，记录样本数，计时是单次顺序执行的请求墙钟时间（含实际结果缓存路径、不含 setup）；小样本不代表生产延迟。本批没有真实效果/费用/性能基线，也没有重复真实实验，未报告波动或真实优胜方案。测试另覆盖真实内存 Chroma 引擎配注入假 embedding 的全库一致性，这仍不是实际模型联调。

## 数据边界

`data/demo_docs/sample_knowledge.json` 是线上默认演示知识；`evaluation/corpora/` 是显式评测专用静态资料，不能自动 seed 到线上库。数据库、会话、画像、日志、历史评测和本机 `.env` 不提交。development 空库可自动导入原示例；production 不自动导入。工具不执行真实退款或创建工单。
