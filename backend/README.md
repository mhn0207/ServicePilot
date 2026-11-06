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

词面分支的 `lexical_k` 是相关性排序后的返回上限。模拟与 Chroma 适配器都先读取当前冻结快照内的全部匹配片段（当前最多72片），执行词面排序后取前 K，不再先对存储顺序使用 `[:k]` 或 `get(limit=k)`。这是小规模离线基准的全匹配评分，不承诺大型语料的扫描开销；扫描范围与返回预算不同。新依赖身份保存 `lexical=all-matches-bm25-before-top-k-v2`，缺此字段的历史报告按 `match-prefix-limit-v1` 识别，可校验/重放，但检索与回答比较均拒绝新旧算法混比。算法身份也参与实例缓存键。线上 `KnowledgeBase.search` 和 ToolManager 行为未改。

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

## 第三批：单轮回答质量

实现和离线验收完成，真实收益待实验。新协议独立于检索报告与旧四维对话 Judge：`rag-answer-dataset-v1`、`rag-answer-fixture-v1`、`rag-answer-report-v1`、`rag-answer-comparison-v1`。没有修改旧 `/eval`、`/chat`、异步任务、前端或在线模型统计。

`evaluation/corpora/answers.v4.json` 为默认回答标注：原 60 个英语问题、40 dev / 20 validation、118 项必答事实、146 项可选补充、133 项结构化禁止项，保持原 group_id/split，包含 3 个 partial 和 6 个 unanswerable。每个事实带稳定 ID、正文摘录和快照片段 ID；未知内容、可答范围、下一步与参考均独立记录。完整 source_dataset_hash、split 的 source_evaluated_dataset_hash、实际子集 evaluated_dataset_hash/case_set、知识与两个冻结身份都保留，不能混合 dev/validation。v1–v3 静态标注及对应 fixture 全部保留，不原地修改旧哈希。v2 拆分复合事实并补虚构动作/日期禁项，v3 修正 only 措辞，v4 在只观察固定离线向量后按原问题逐例复核必答范围；没有依据真实模型输出改答案，没有改动第二批冻结知识和检索标注。

v4 的 `facts` 只列当前问题不可缺少的结论，`optional_facts` 保留原事实/ID/证据作为可选补充，`scope_rationale` 给出每例取舍。例如 v2-020 不再要求讲已开票更正，v2-041 不再要求讲API令牌重发；v2-023 明确询问交接清单与秘密移除，仍要求全部清单项。可选事实不进入正确性 Judge 的预期ID/参考或 oracle 构建，省略不扣事实覆盖分；若回答额外断言仍须经过完整 grounding 与禁止项检查。未删除原264项事实，没有通过放宽禁止结论来抬分；实际评分合理性仍待独立人工校准。

默认标注哈希 `c88ffa6bada4e55079e532d92750d93b334f773abc74aba44698968fd1024e3b`，回答冻结哈希 `dde0a9d74393463a3a6115a1544d5c068ddb27183e79512aba6775e4219110ca`。静态校验重建全部原文和所有split，核对每条问题/检索用例哈希，并检查必答与可选两组的ID去重和全部引用。没有金额、价格、时效或合同条款时明确未知，partial仍须回答已知核心规则。

```powershell
# 所有命令均从 backend 执行；默认不会读取 .env、调用模型或连接用户库
.venv/Scripts/python -X utf8 scripts/evaluate_answer_quality.py --validate-annotations
.venv/Scripts/python -X utf8 scripts/evaluate_answer_quality.py --run-id answer-dev
.venv/Scripts/python -X utf8 scripts/evaluate_answer_quality.py --case v2-001 --run-id answer-one

# 消费第二批完整单策略报告；与当前 suite/split/Top-K/实际片段严格核对
# 固定 fixture 只认识预先录入的精确请求，其他上下文须提供相应fixture或显式真实依赖
# 默认v4 fixture按修复后的B词面算法制作，上面的example-dev-matrix应由当前代码生成
.venv/Scripts/python -X utf8 scripts/evaluate_answer_quality.py --source-report data/eval/rag/experiments/example-dev-matrix/B/report.json --run-id answer-from-B

# validation 明确指定一个候选；B为工程默认，不扫描策略矩阵
.venv/Scripts/python -X utf8 scripts/evaluate_answer_quality.py --split validation --candidate B --case v2-058 --run-id answer-validation-one

# 纯fixture评分记录重放/重新校验汇总，无生成、Judge或检索新请求
.venv/Scripts/python -X utf8 scripts/evaluate_answer_quality.py --fixture data/eval/rag/answers/answer-dev --run-id answer-replay
.venv/Scripts/python -X utf8 scripts/evaluate_answer_quality.py --compare data/eval/rag/answers/answer-dev/report.json data/eval/rag/answers/answer-replay/report.json --run-id answer-compare

# 显式选择历史标注，重放既有v3回答报告；不重新运行旧算法或调用模型
.venv/Scripts/python -X utf8 scripts/evaluate_answer_quality.py --annotations evaluation/corpora/answers.v3.json --fixture data/eval/rag/answers/batch3-final-dev --run-id answer-v3-replay

# 必要相关验收和完整回归
.venv/Scripts/python -X utf8 -m pytest tests/test_answer_quality.py tests/test_retrieval_evidence.py tests/test_retrieval_strategies.py -q
.venv/Scripts/python -X utf8 -m pytest tests -q
```

单策略来源可以是第二批 `ExperimentReport`，或 `retrieval_source(..., runner=显式执行器, config=策略, mode=实际模式)` 注入；继续复用第二批执行状态/轨迹及第一批检索评分器。默认 B + SyntheticIndex，所有阶段分别记录依赖身份；synthetic 检索 + fixture 生成/Judge 明确标记 mixed，不能伪装全真实基线。固定 fixture 是预录入答案和逐项判断，通过实际请求哈希查表；它不做语义理解，也不根据运行中的参考临时打分。

两个分支使用相同生成规则、temperature（默认 0）、max_tokens（默认 2048），彼此无会话历史。生成输入仅问题和本分支实际上下文。上下文采用规范 JSON，保留片段 ID/标题/完整正文；默认上限 16000 字符，包括 JSON 格式开销，可用 `--context-chars` 调整。超预算整分支明确拒绝，无隐式截断；`context_shown=false` 时保存的是未展示的候选文本，没有发送给生成器。oracle 按 fact_id 顺序、每事实最小替代 chunk_id、跨事实稳定去重构建，不从检索输出或模型答案选择。oracle 不硬截 Top-K，记录实际片段数、字符数、文本/哈希及预算，因此不是公平的检索方案。无正向依据的 unanswerable 的 oracle=N/A；retrieved 仍评估诚实处理。检索错误不冒充合法空集，不调用该分支生成，不被 oracle 成功掩盖；合法空召回仍可请求限制说明。

正确性 Judge 与 grounding Judge 是两次独立调用，默认各 8192 输出 token，可用 `--judge-tokens` 设置。前者看标注、逐事实 covered/not_covered/contradicted/uncertain、逐禁止项 violation/no_violation/uncertain 和范围处理 pass/fail/uncertain；后者不接收参考或 oracle 标注，只看本分支实际上下文、实际回答及确定性的句子审计单位，逐断言返回 supported/unsupported/uncertain。每个单位必须被覆盖；需要标注 extraction=complete/partial/unknown。引用的回答 span 和证据 quote 必须真实存在；未知片段、未展示证据、重复断言/ID、遗漏事实、无证据 supported、非法 JSON/重复键/NaN/布尔偏移、空响应和未完成输出均不能评分通过。确定性检查只核对结构/引用/完整性，语义蕴含、否定、断言抽取是否充分仍需真实 Judge 与人工校准；stub 输入隔离测试不证明真实模型抗注入。

句子审计单位同时识别标点和 LF/CRLF/CR 换行，未带句末标点的段落也必须检查，避免漏审整行业务断言。新报告的提示词/评分身份增加 `sentence-units-v2`；历史只有三项提示词哈希的报告按原单位规则重放，保留历史评分口径，不能冒充已修复的完整性验收，也不能与新规则报告直接比较。固定fixture仍按实际请求哈希匹配，不改写旧fixture哈希。

| 报告项 | 分子、分母与限制 |
| --- | --- |
| generation_success_rate | 正常完成生成数 / 全部适用分支；检索失败和预算拒绝也留在分母 |
| judge_coverage | 两个 Judge 完整有效且无 uncertain/不完整抽取的任务数 / 全部适用分支；另列 valid/generated |
| end_to_end_complete | 完整有效评判任务数 / 全部适用任务；完成不等于内容正确 |
| end_to_end_success | 事实全覆盖、无禁项、范围处理通过、所有事实断言有据且无未决的任务数 / 全部适用任务 |
| fact_coverage_micro_lower | scored 子集 covered 事实数 / 该子集全部必要事实数，uncertain 不当 covered；macro 是有事实样本覆盖比例的均值 |
| forbidden_violations | violation 数 / scored 子集所有禁项，另列 uncertain；关键违规逐例单列，不被均分抵消 |
| grounding_supported_lower | 抽取完整且可评分子集中 supported 事实断言数 / 该子集事实断言数；uncertain 不当 supported；partial/unknown 抽取不产生整段比例 |

scored 是两个 Judge 均通过结构/证据校验的子集，允许保留 uncertain 用于保守覆盖统计；valid 进一步排除未决。两个集合均明确列任务 ID、完整分母、缺失/未决数量。Judge 异常不填 0 或满分，不混入有效质量均值。零事实、零断言、零禁止项为 null/N/A；没有可核验断言不默认为 100% 有据。成对报告列共同生成/共同有效交集及分母、每例上下文/失败/差异，同时保留全部主分母。比较固定同一回答标注、知识、问题集合/split、Judge/生成配置、上下文政策和阶段模式，研究变量为检索配置；A/B/C/D 本身差异不造成误拒绝，其他模型混杂变量拒绝。缺样本/未完成拒绝，评分不完整或有关键未决则 inconclusive；没有自动通过或晋升基线。

输出只追加到 ignored 的 `data/eval/rag/answers/<run_id>/`，已存在即拒绝。保存 `report.json`、快照、标注、来源检索报告、`review.json`。报告记录 Git HEAD/工作区、来源 run/config/trace、各阶段依赖与实际调用/请求哈希/参数/耗时/供应商用量，重放保留原模式、原调用观测并另记 observation/replay_of/replayed_at，不是本次实测。供应商没返回 token 为 null，没调用才汇总为 0；解析失败也保留已返回用量。SDK 与业务重试都为 0，没有估算不可见尝试和费用。review 清单覆盖高分、低分、uncertain、关键违规，包含问题、实际上下文、回答、判定和待填 human_verdict/disagreement/reviewer；没有替用户完成复核。

退出码：**2** 为输入/配置/身份/报告/目录校验失败，**1** 为报告写入成功但存在阶段错误或未决，**0** 为没有阶段错误/未决；均不代表所有回答质量通过。默认 dev fixture 故意包含 uncertain，因此预期 1。

真实小样本入口已提供，以下命令本批没有执行，需用户另行明确授权。只从安全注入的进程环境读取 `SERVICEPILOT_ANSWER_GENERATION_MODEL/API_KEY/BASE_URL` 和 `SERVICEPILOT_ANSWER_JUDGE_MODEL/API_KEY/BASE_URL`；BASE_URL 可省略，不读取真实 `.env`，报告只保留模型 ID 和端点哈希。独立 SDK 直接调用，绕开线上全局统计/缓存/熔断，DeepSeek 端点显式关闭隐式 thinking。真实模式必须 `--enable-provider` 且给出 `--limit` 或 `--case`；默认要求第二批真实来源报告，真实生成消费模拟检索时必须额外 `--allow-mixed`，始终标为 mixed，不能与全真实报告判优。

```powershell
# 先通过第二批独立环境准备真实 B 报告，再经另行授权执行本条；最多1例、每适用分支1次生成+2次Judge
.venv/Scripts/python -X utf8 scripts/evaluate_answer_quality.py --mode real --enable-provider --limit 1 --source-report data/eval/rag/experiments/real-dev-B/B/report.json --generation-tokens 2048 --judge-tokens 8192 --run-id real-answer-one

# 显式混合实验入口示例，真实生成/Judge + 默认模拟检索；不是全真实基线，本批未执行
.venv/Scripts/python -X utf8 scripts/evaluate_answer_quality.py --mode real --enable-provider --allow-mixed --case v2-035 --run-id mixed-answer-one
```

英语合成资料不代表生产政策；没有真实效果、费用、重复实验波动、中文收益或真实样本人工校准。本批仅单轮独立 RAG，第四批可复用身份、执行记录、上下文与逐项评分接口，但没有实现多轮 Agent 或展示闭环。

## 数据边界

`data/demo_docs/sample_knowledge.json` 是线上默认演示知识；`evaluation/corpora/` 是显式评测专用静态资料，不能自动 seed 到线上库。数据库、会话、画像、日志、历史评测和本机 `.env` 不提交。development 空库可自动导入原示例；production 不自动导入。工具不执行真实退款或创建工单。

## 第四批：Agent 离线执行与页面

新增固定 `agent-dev-v1` 开发协议样例。外部模型使用脚本化 SDK 响应，检索使用第二批 StrategyRunner 的 B 策略、Top-K=3 和内存特征哈希向量；这是实验适配器，不是业务默认搜索算法。真实 AgentOrchestrator 负责角色路由、工具白名单、工具循环及 Composer。中文样例只引用原有六篇演示资料，不能解释为中文质量基准。

从 backend 使用本项目 Python 3.12 环境：

```powershell
.venv/Scripts/python -X utf8 scripts/evaluate_agents.py
.venv/Scripts/python -X utf8 -m pytest tests/test_agent_evaluation.py tests/test_agent_api.py tests/test_chat_reliability.py -q
```

追加报告保存在 ignored 的 `data/eval/rag/agents/<run_id>/`。检索调用覆盖、角色、实体污染、返回/实际展示证据与回答检查分别呈现分子分母；无适用项为 N/A，未完成的语义 Judge 保留未决，不用零分或总分掩盖。报告保存实际生成输入，Composer 仅看到子 Agent 文本时不宣称其直接看到了原始证据。模拟依赖调用计数不是供应商用量；没有调用真实模型或证明真实收益。

`POST /eval/rag/run` 只接受 `{"suite":"agent-dev-v1"}`（也可省略 body），返回 202、job_id 和 run_id；复用 `GET /eval/jobs/{job_id}` 查询有界串行队列。`GET /eval/rag/runs` 与 `GET /eval/rag/runs/{run_id}` 读取终态报告；使用固定目录，拒绝路径穿越及链接逃逸。配置 Bearer 鉴权时均沿用 admin 权限。进程重启不恢复运行中的任务，终态磁盘报告可继续读取。新入口不调用旧 `/eval/run` Judge 链路。

现有评测页新增 RAG 区块，支持明确启动、已知任务恢复、历史报告和逐轮展开。HTTP 聊天验收单独标记 `scope=http_chat_fixture`，用 fakeredis 与固定模型验证实际路由回执和实体隔离；它不代表真实 Redis/Chroma、生产并发或业务模型验收。真实小样本和人工校准需要后续授权。

第四批重放：`.venv/Scripts/python -X utf8 scripts/evaluate_agents.py --replay batch4-final-offline`，只读取已保存报告并重新汇总，不发模型/检索请求；Judge fixture 或评分代码身份改变时明确拒绝重放，历史原报告仍可读取。`--validate-annotations` 校验七场景九轮及知识身份。CLI 退出 0 表示执行终态完成，不表示检查全通过或质量回归通过。

## 真实验证后的定点修复

金额提取与会话合并现在识别明确撤回，例如“金额不是99元”“99元不属于这笔订单”；被撤回的金额不会重新写入新订单，同一订单也会清除对应旧值及来源。“不是99元，是199元”保留新金额；“没有收到99元退款”仍保留待核验金额。这是针对明确表达的确定性规则，不宣称通用自然语言否定理解。

提供 `search_knowledge_base` 工具时，包含退款/续费/重复扣款/企业合同/套餐/登录认证主题且带明确咨询表达的请求，首次模型调用使用指定检索工具的 `tool_choice`。模型仍生成实际查询，工具返回沿用原调用循环；后续回答不会重复强制检索。问候、单纯订单更正和不含业务政策主题的字段收集保持可直接处理。该规则以当前中文业务场景为范围，不将全体请求都视为必须检索。

退款与订阅主题也识别退费、退钱、退订、续订、订阅、自动扣费/扣款，以及相邻的资金退回/返还表达。例如“取消自动扣费后，已经支付的钱会自动退回吗”同样要求先检索；单纯更正自动扣费金额、快递退回等非资金问题不因这些词自动触发。规则仍要求明确咨询表达，不代表通用自然语言语义识别。

Agent 统计保留 `total` 为已开始的请求数，新增 `in_flight`（在途）与 `completed`（已结束）。成功率与平均耗时只以已结束请求为分母，避免监控把等待中的请求误报为失败；没有已结束样本时沿用成功率1、平均耗时0的空样本约定。这只是监控约定，不代表已验证可用率。失败、取消和不完整回答仍在结束后按原规则计入结果，近期窗口及恢复探测规则保留。

精简 Docker 镜像不需要安装 Git 才能运行 Agent 评测：Git 不可用时 HEAD/dirty 记 null，保留源码模块哈希。真实小样本验证与单元测试分开记录在 ignored 的 `.local-review/`；只允许 Flash 的实验不切换 Pro。

## 聊天请求时间预算

`/chat` 保留50秒外层保护，另设置45秒内部请求截止时间；锁等待和记忆读取消耗这段时间，模型阶段不会各自获得新的整轮预算。主Agent阶段最多28秒，并为汇总与提交预留时间；意图5秒、查询改写2秒、模型重排3秒、单次Agent生成最多15秒、Composer最多5秒，均进一步受剩余时间约束。没有聊天预算上下文的旧CLI/调用者沿用其原有上限。

改写/重排在剩余预算不足18秒时跳过，使用原查询/已有排序；由预算引起的降级结果不写普通共享缓存。请求预算到期不计入工具连续故障或打开熔断器；被取消的半开探测释放后可由后续请求重新探测。工具自身错误/超时仍按原故障规则处理。

专业Agent失败后，剩余不足11秒不再启动完整降级；允许降级时最多8秒并预留收尾时间。本轮成功知识检索可供有当前工具权限的降级角色复用，要求同一handler且知识缓存revision未变，保留原query/参数/返回和原tool_use_id关联。缓存只在当前Request内保存；不复用副作用工具，不跨用户/会话传播。

响应 `performance.budget_events` 记录阶段开始/成功/失败/超时/未调用、跳过原因、`phase_remaining_s` 与 `request_remaining_s`。低预算失败仍沿用503/可重试回执，失败轮不写入正常历史；50秒外层超时仍保留504保护，不编造成功或改变副作用工具的未知结果语义。取消本地等待不等于远端供应商已停止计算，真实耗时/费用仍以实际观测为准。
