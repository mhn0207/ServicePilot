# ServicePilot 后端

Python 3.12 / FastAPI / Redis / ChromaDB / Anthropic SDK / MCP。

本页描述当前接口、验证入口和实验边界。项目功能、最短演示与来源说明见 [根 README](../README.md)。

## 配置与运行

从本目录使用 `.env.example` 建立本机 `.env`，已有文件不覆盖。填写 `ANTHROPIC_API_KEY`、`ANTHROPIC_MODEL` 与兼容 Anthropic Messages 的 `ANTHROPIC_BASE_URL`。模型凭据不提交、不传给前端。

```powershell
docker compose -p servicepilot-local -f docker-compose.yml -f docker-compose.web.yml up -d --build --wait --wait-timeout 180
```

示例配置的 `SERVICEPILOT_ENV=development`（兼容 `ZHIYING_ENV`）用于本机演示，并允许空库导入 `data/demo_docs/sample_knowledge.json`。production 必须设置 `SERVICEPILOT_API_TOKENS`（或 `ZHIYING_API_TOKENS`）；ChromaDB 连接失败时中止，禁止回退到本地库。外部 MCP 由示例配置中的 `MCP_EXTERNAL_ENABLED=false` 关闭；直接使用进程环境启动时也应显式设置此值。

若容器调用模型需要宿主机代理，须单独配置容器可访问的代理地址。容器中的 `127.0.0.1` 指向容器自身，不能直接复用宿主机回环代理。不要为排查连接问题关闭证书校验。

## 请求链路

1. Vue 工作台通过 `/chat/stream` 接收 SSE 阶段事件和最终答复。FastAPI 绑定请求身份，读取会话状态和近期消息。
2. 编排器结合意图、实体和近期 Agent 表现选择专业角色。明确的跨域请求并行执行，账单流水从工具结果直接渲染，汇总漏掉分支时回退到分段输出。
3. ChromaDB 提供向量召回、情景记忆和画像。知识库还使用词面 BM25 排序，单域检索可按剩余预算执行查询改写和模型重排。
4. Redis 会话租约串行化同一会话。消息、槽位和请求回执在同一事务提交。SQLite 唯一约束保护沙盒核验单，重试同一业务操作可取回已有单号。

## 主要接口与身份

| 接口 | 用途 |
| --- | --- |
| `POST /chat` | 聊天；重试沿用 `client_request_id` |
| `POST /chat/stream` | 同一聊天事务，以 SSE 推送实际阶段及最终回复 |
| `GET /conversation/history?conv_id=...` | 当前用户、当前会话已提交的近期消息；不调用模型 |
| `POST /search?query=...&top_k=3` | 当前业务 RAG 流水线；会改写查询并可能重排 |
| `POST /knowledge/add`、`POST /knowledge/upload` | 导入 JSON、TXT 或 Markdown，上传限制 10 MB |
| `GET /knowledge/stats` | 知识片段数量 |
| `GET /knowledge/documents` | 管理员文档列表及待核验旧片段 |
| `GET /knowledge/documents/{document_id}` | 管理员查看版本与有序片段 |
| `DELETE /knowledge/documents/{document_id}?expected_version=...` | 管理员按预览版本删除整篇文档 |
| `GET /skills`、`POST /skills/reload` | 查看/重新加载业务规则 |
| `GET /trace/tool/{request_id}`、`GET /trace/tools` | 单次/最近工具轨迹 |
| `GET /monitor`、`GET /metrics` | 运行摘要及 Prometheus 指标 |
| `POST /eval/run`、`GET /eval/jobs/{job_id}` | 意图和对话评测的异步任务 |
| `POST /eval/baseline` | 管理员显式选择已完成报告作为基线 |
| `POST /eval/rag/run`、`GET /eval/rag/runs` | 固定 Agent 离线 suite 及历史报告 |
| `/mcp/` | 标准 MCP 知识 Tools / Resources |

启用鉴权后，聊天用户身份由 Token 映射决定，覆盖请求正文中的 `user_id`；管理、轨迹及评测接口需要管理员权限。知识库在当前应用范围内共享，不是独立租户知识库。

### 知识维护

上传 `.md` 自动使用 `markdown-v1`。`.txt`、JSON 导入和离线语料默认使用 `legacy`；JSON 可以在每篇文档中指定 `"chunk_strategy": "markdown-v1"`，在线入库与离线快照共用切块规则和证据 ID。Markdown 片段带章节路径，并优先保留段落、嵌套列表和围栏代码块。超长块按 500 字上限分割（含章节路径），超长代码优先在完整行之间切分并补齐围栏。两种模式的文档可并存，管理页会显示切块模式，可以按各自的文档 ID 查看和删除。

文档管理沿用导入时生成的文档身份摘要，不使用标题作为删除定位，也不接收磁盘路径。列表只读取元数据，正文在展开文档时读取；接口禁止缓存。管理员确认删除时提交所查看的内容版本；已更新或身份/片段不完整返回 409，不存在返回 404，非法身份/版本返回 422，存储异常返回通用 503。

删除清理该文档全部片段，不能撤销；文档中的知识可能影响所有聊天用户。缺少文档身份的旧片段单独只读展示，缺版本、混合版本或缺片段的文档禁止删除，不自动迁移或猜测归属。管理扫描上限为 5000 片，超限明确报错，不静默截断列表。

导入、上传与版本删除通过 Redis 租约协调，依赖同一 Redis 数据库和 Chroma 知识库。变更中绕过检索缓存，写入完成后用共享版本让其他 API 进程识别旧结果。取消 HTTP 等待时，已经启动的写入仍会继续收尾。删除接口失败后请刷新并核对文档版本。

`/health` 在两秒总预算内检查 Redis、记忆/画像及知识库正在使用的 Chroma collection。远端 Chroma 采用有一秒传输超时的独立异步 count 请求，API 版本跟随实际客户端；本地模式合并未完成探测，避免取消等待后重复创建阻塞线程。未初始化、依赖异常或超时返回 503，恢复后返回 200；不会调用模型、生成 embedding 或写业务数据，也不证明模型 Key、额度或模型名称可用。

### Prometheus 采集身份

无鉴权 development 模式默认使用空的 `config/prometheus.token.example`。开启 API 鉴权后，为采集生成独立随机令牌文件，不要复用模型 Key：

```powershell
# 从 backend 执行；文件已存在时拒绝覆盖。
python -c "from pathlib import Path; import secrets; Path('config/prometheus.token').open('x', encoding='utf-8').write(secrets.token_urlsafe(32) + '\n')"
```

在本机 `.env` 设置 `SERVICEPILOT_METRICS_TOKEN_PATH=./config/prometheus.token`，再使用相同 Compose 命令更新服务。新变量为空时兼容 `ZHIYING_METRICS_TOKEN_PATH`。Compose 将该文件作为 secret 同时提供给应用和 Prometheus。应用读取 `SERVICEPILOT_METRICS_TOKEN_FILE` / `ZHIYING_METRICS_TOKEN_FILE=/run/secrets/metrics_token`，Prometheus 使用 `authorization.credentials_file`。非 Compose 启动时，直接设置应用的 `SERVICEPILOT_METRICS_TOKEN_FILE`（或 `ZHIYING_METRICS_TOKEN_FILE`）为本机文件路径。

令牌至少 24 字符、不可含空白，只授权 `GET /metrics`，不能作为聊天、轨迹或管理身份。管理员 API Token 仍可读取指标，普通用户无此权限。凭据文件缺失或格式错误会拒绝采集；空文件不在启用 API 鉴权时提供访问能力。真实 `*.token` 同时被 Git 和后端镜像构建排除，不将凭据提交或写入前端。

## 聊天的提交与失败语义

同一用户和会话由进程内锁与 Redis 租约串行化，租约定期续租；失锁时取消持有者。成功轮次把用户消息、助手消息、业务状态和回执放入同一 Redis 事务。失败轮次保留回执，不提交成功历史。

- 成功请求重试返回原回执；同一 ID 携带不同消息返回 409。
- 可重试失败允许相同 ID 再次执行，因此失败重试不一定返回相同响应。
- 副作用工具执行前保存不可自动重放的待核验标记；当前内置业务工具本身不连接退款、订单或工单系统。
- 全轮超时与阶段预算分别控制；HTTP 超时不表示供应商或外部工具肯定未执行。

阶段预算的全部数值集中在 `core/request_budget.py` 的 `BudgetPolicy`：请求预算 45 秒（外层硬超时 50 秒），主 Agent 28 秒并为后续阶段预留 5 秒，剩余不足 11 秒不再启动兜底，不足 18 秒跳过查询改写与重排。修改任一数值都必须通过 `validate()`，否则服务在导入阶段失败；`tests/test_budget_policy.py` 覆盖这些约束。

system 提示词按“工具 → 角色契约 → Skills”排列，三者不含本轮对话内容，按前缀自动缓存的兼容服务可直接命中。使用 Anthropic 官方接口时可设置 `ZHIYING_PROMPT_CACHE_BREAKPOINTS=1`，在角色契约和 Skills 末尾放置显式缓存断点；默认关闭，请求内容不变。断点开启后的实际命中率尚未实测。

后台压缩用快照与 WATCH 提交，避免覆盖新增消息；画像更新按用户合并并节流。上述机制不是跨 Redis、ChromaDB 与外部工具的全局事务，也不等于任意故障下 exactly-once。

历史读取只返回 Redis 当前保留的最多 20 条用户/助手消息，按时间顺序排列，不返回内部 metadata、画像或工具轨迹；启用鉴权时身份由 Token 决定，传入其他 `user_id` 不改变读取对象，管理员也只读自己的会话。响应禁止缓存。工作记忆 TTL 为 24 小时，后台压缩可能只保留更少的近期消息，因此这不是完整聊天归档；不存在或已过期的工作记忆返回空列表。读取异常或超时返回 503，不把依赖故障伪装成空会话。

### 意图识别短路

首轮明确请求可启用 `SERVICEPILOT_INTENT_FAST_PATH=true`（兼容 `ZHIYING_INTENT_FAST_PATH`，默认关闭）。规则需匹配整句，向量 Top-1 需与规则同意图且相似度至少为 0.82。符合规则的请求给向量最多 50 毫秒的启动窗口，慢向量与意图 LLM 随后继续并行。带历史的请求直接走完整识别。短路只影响意图 LLM，业务 Agent 仍正常执行。融合权重按实际启用且可用的分支归一化，低于阈值时交给既有澄清和协作逻辑。

## 聊天阶段事件

`POST /chat/stream` 使用与 `POST /chat` 相同的请求体和访问 Token。客户端发送 `Accept: text/event-stream` 并读取结构化事件。每次实际执行按路径发出 `stage` 事件，例如 `lease_acquired`、`intent_recognized`、`agent_routing`、`tool_call`、`composing`、`committing`。工具事件只带名称与 Agent，回复正文随事务完成后的 `final` 返回，格式与 JSON 聊天接口一致。

鉴权与请求体验证按常规 HTTP 状态返回。连接建立后的处理失败以 `error` 返回逻辑状态、错误详情与是否可重试。断流后原轮次在既有超时和会话租约下继续处理。重试需保留同一消息、`client_request_id` 和会话参数，命中成功回执时收到 `replayed` 和原回复。工作台的「重试此请求」会保留这些参数。

## 账单沙盒演示

在本地演示环境的 `.env` 中设置 `SERVICEPILOT_SANDBOX_ENABLED=true`，并在 `SERVICEPILOT_API_TOKENS` 的 JSON 映射里为 `user_id` 为 `u1001` 的账号配置独立的随机 Token（至少 24 字符，角色为 `user` 或 `admin`）。启动服务，在工作台连接设置中填入该访问 Token。

1. 发送 `沙盒查账 ORD-20261024`，看到一笔 199 元已入账流水和一笔 199 元预授权冻结。也可以自然语言询问这笔订单，由账单 Agent 使用只读查询工具。
2. 发送 `确认提交退款核验 ORD-20261024`，记录返回的 `RFD-...` 单号。系统只创建待人工核验的沙盒记录。换一个会话或请求 ID 再确认，仍返回同一单号。
3. 发送 `查询退款核验单 RFD-...`（替换为完整单号），看到订单号和待人工核验状态。其他账号无法查询或提交该订单。

这三条精确命令沿用 `POST /chat` 的身份、会话与回执协议，由服务端执行，可用于固定演示。自然语言会走原有模型链路。SQLite 文件位于 `backend/data/sandbox/business.sqlite3`，Compose 使用 `sandbox-data` 数据卷，重启或重建容器后保留。沙盒默认关闭，仅使用合成数据。

## 离线测试

使用项目自己的虚拟环境；安装步骤仅在准备环境时执行：

```powershell
python -X utf8 -m pip install -r requirements-dev.txt
python -X utf8 -m pytest tests -q
```

`tests/conftest.py` 在 pytest 生命周期内禁用 dotenv 文件解析，覆盖 `api.main` 的 `load_dotenv` 及 MCP/Chroma 配置依赖的 `dotenv_values` 路径；测试配置通过 `monkeypatch.setenv` 等方式显式注入，临时假配置文件仍可创建。单元测试以固定模型、FakeRedis、MockTransport 或注入的 embedding 为主；不能由测试通过推导真实模型质量或真实存储在所有故障下的行为。`tests/verify_*.py` 与在线脚本是额外入口，执行前确认隔离环境。

网关重建回归可从 `backend/` 运行 `python -X utf8 tests/verify_gateway_recovery.py`。脚本在独立 Docker 网络中更换同名 HTTP 上游的 IP，检查网关进程持续运行且请求路径、查询参数及两个 API 前缀正常转发。Compose 网关使用 Docker 内置 DNS，Nginx 镜像需支持开源版的动态上游解析（1.27.3 起）。

自动化测试覆盖编排、会话重试与实体更正、知识维护、权限、存储探测和评测协议。独立 Docker 验证使用新数据卷，覆盖 HTTP 聊天、并发重试、用户隔离、Redis 故障恢复及重启后消息、知识和报告保留；模型固定响应的验证只检查工程行为。

## 评测入口

以下命令从 `backend/` 执行；Windows 可将 `python` 替换为 `.venv/Scripts/python`。默认命令不读 `.env`、不调用供应商模型。

### 检索证据

```powershell
python -X utf8 scripts/evaluate_retrieval_evidence.py
```

使用 6 篇中文演示知识、6 个片段和 24 条回归用例。固定结果中有 22 道证据题，Hit@3 为 **6/22（27.27%）**，证据 Recall@3 为 **25.00%**，MRR@3 为 **0.2424**。fixture 故意放入一次执行错误，预期退出码为 **1**。退出码 0 表示无执行错误，命中情况以分项指标为准。退出码 2 表示输入或协议校验失败。

切片与在线入库共用 `tooling/knowledge_chunks.py`。证据 ID 包含文档身份、原文 SHA-256 版本和片段序号；评分核对实际 ID、标题和正文，拒绝未知/过期证据。旧 Chroma 存储 ID 不变；缺元数据的旧片段不猜造证据 ID。

| 指标 | 含义 |
| --- | --- |
| Hit@K | 前 K 名至少覆盖一组标注证据 |
| Evidence Recall@K | 覆盖的必需证据组数 / 全部必需证据组数 |
| MRR@K | 首个有效证据片段实际名次的倒数 |

两个证据组只覆盖一个时，Hit=1、Recall=0.5。重复片段不增加覆盖，也不删除排名后补足 K；fallback 不算证据。错误保留在主指标分母，并另报错误率和成功子集；无答案题不进入正向召回分母。

旧 `scripts/evaluate_retrieval.py` 的 12 条标题级冒烟保留原含义。证据级在线入口需显式 `--online --base-url ...`，只从进程环境读取 `SERVICEPILOT_EVAL_TOKEN`（兼容 `ZHIYING_EVAL_TOKEN`）；它调用业务 `/search`，可能消耗改写和重排额度，不是纯向量基线。

### 独立策略对比

```powershell
python -X utf8 scripts/evaluate_retrieval_strategies.py --validate-suite
python -X utf8 scripts/evaluate_retrieval_strategies.py --matrix --run-id example-dev-matrix
python -X utf8 scripts/evaluate_retrieval_strategies.py --split validation --candidate B --with-baselines --run-id example-validation-B
```

| 策略 | 步骤 |
| --- | --- |
| A | 向量召回 |
| B | 向量 + 词面召回，RRF 融合 |
| C | 原查询与改写查询分别混合召回，再跨查询融合 |
| D | C 的候选加模型重排 |

默认 Top-K=3，candidate/vector/lexical 预算各 20，cache=bypass。显式区分分支候选、单查询融合候选和最终返回数量；不相加不同查询的原始相似度。词面排序使用有限冻结语料内的全匹配片段，再取前 K；算法身份为 `all-matches-bm25-before-top-k-v2`，旧算法报告不能直接混比。

该执行器独立于线上 ToolManager，使用自己的依赖、缓存、统计和知识快照。**实验 B 与业务默认混合流水线不是完全相同的实现**，实验分数不能直接当作 `/chat` 收益。模拟向量是 64 维特征哈希，模拟改写/重排是固定规则，不是语义模型效果。

validation 禁止默认矩阵，需显式固定候选。比较要求知识、标注、split、Top-K、模式与依赖身份兼容；错误、降级、缓存命中和完整执行率分别报告，不自动选优或晋升。

真实 A/B 需要 `--mode real --onnx-cache <已有模型缓存目录>`；不会下载模型或使用业务库。C/D 还需要 `--enable-provider` 及进程环境中的 `SERVICEPILOT_EVAL_API_KEY`、`SERVICEPILOT_EVAL_MODEL`，可选 `SERVICEPILOT_EVAL_BASE_URL`。真实模式可能收费，应先限定样本和配置。

真实 D 在所有候选中请求 Top-K 索引，每项发送前 80 字标题与前 200 字正文。DeepSeek 的改写和重排默认关闭思考。轨迹分别记录截断、JSON 解析和输出结构校验失败。A/B 可与不同提示词版本的模型策略比较，因为 A/B 不调用模型。两组均使用模型时，比较器要求模型与提示词身份一致。

### 成对回答质量

```powershell
python -X utf8 scripts/evaluate_answer_quality.py --validate-annotations
python -X utf8 scripts/evaluate_answer_quality.py --run-id example-answer-dev
```

默认使用 B 模拟检索和 v4 固定生成/Judge fixture，报告明确记录混合依赖。retrieved_context 与 oracle_context 独立生成；生成器看不到参考、事实 ID、禁项或另一分支回答。oracle 是诊断对照，不是部署方案。

正确性 Judge 看参考并逐事实/禁项判断；grounding Judge 只看实际展示材料，检查答案断言的依据。引用、回答 span、覆盖和身份可确定性校验，语义蕴含与充分抽取仍需真实 Judge 和人工校准。错误与 uncertain 保留，N/A 不填成 0，不合成一个混合粒度总准确率。

默认 fixture 故意保留漏答、矛盾、违规和未决；退出 1 表示执行/评价有错误或未决，不等同于未生成报告。退出 0 也不表示质量全部合格。当前正式回答协议为英语单轮，不能冒充中文多轮回答评分。

真实回答入口需要显式 `--mode real --enable-provider` 与 `--limit` 或 `--case`，生成/Judge 凭据使用 `SERVICEPILOT_ANSWER_GENERATION_*` / `SERVICEPILOT_ANSWER_JUDGE_*` 进程环境。真实生成消费模拟检索时另需 `--allow-mixed`；不会加载真实 `.env`。

真实 Judge 的引用偏移有误时，只有逐字引文在答案中恰好出现一次才重新定位。引用归属、必需字段和证据覆盖仍按协议验证。报告分别记录截断、JSON 解析、输出结构和证据校验错误，保留未决项。

### Agent 与工作台

```powershell
python -X utf8 scripts/evaluate_agents.py --validate-annotations
python -X utf8 scripts/evaluate_agents.py --run-id example-agent-dev
```

固定 `agent-dev-v1` 使用真实 Orchestrator 和工具循环，搭配固定模型与 synthetic 检索。检查多轮实体、角色、检索调用及证据，记录 `scope=orchestrator`；不代表完整 `/chat` 或真实模型质量。

本地固定回放覆盖 9 轮。角色覆盖 **9/9**，必需工具调用 **6/7**，调用执行成功 **5/6**，返回证据 **4/7**，必需事实 **6/7**，新实体更正 **4/4**，旧实体清理 **2/2**。固定漏调和工具失败用例用于验证评分分母与失败拦截。

前端 RAG 区块通过异步任务接口运行同一 suite。报告追加到 `data/eval/rag/agents/`，历史终态可跨重启读取；进程内运行中任务不恢复。固定 Judge 判定只做 fixture 验收，不是独立语义模型或人工校准。

所有运行产物写入被忽略的 `data/eval/`，目录拒绝覆盖。报告保留数据/知识/依赖/评分身份，部分入口保存源码白名单归档；不把密钥、业务数据或 `.env` 纳入归档。旧静态标注和 fixture 保留用于兼容性回归与重放。

## 数据与已有实验结论

| 数据 | 规模与划分 | 当前证据边界 |
| --- | --- | --- |
| 中文演示 | 6 篇 / 6 片；24 条证据回归 | 线上开发 seed，不是业务真实政策 |
| 英语合成规则 `suite.v2.json` | 24 篇 / 72 片；40 dev、20 validation | 虚构企业规则；36/18 条正向题 |
| `public_zh.v1` | 12 篇短摘要 / 12 片；16 dev、8 validation | GitHub、阿里云与 Stripe 来源；验证正向题 7 条，按供应商分组 |
| `public_zh_hard.v1` | Redis 资料整理 6 篇 / 12 片；两个 split 各 15 题 | 24 条正向题；来源与语义标注待人工核验，未运行正式策略/生成实验 |
| `public_zh_expanded.v1` | 84 篇 / 120 片；沿用 `public_zh.v1` 的 16 dev、8 validation | 新增 72 片合成业务背景与 36 片定向干扰，独立人工复核待完成 |

冻结身份校验只保证数据一致，不证明来源现行有效、语义标注正确或人工独立性。中文资料与初始题目由 AI 整理；`public_zh.v1` 的展示标注获维护者认可，不等于逐一审计源网页或校准模型答案。hard 集人工字段仍为空，不覆盖旧集。

### 本地检索记录

以下为维护者本地报告摘要，原始运行产物未随仓库分发。固定 MiniLM 缓存、Top-K=3、预算各 20、bypass。扩展池的两个 split 各运行两轮，两轮逐题评分一致。

| 数据 / split | 正向分母 | A / B 证据 Recall@3 | 判断 |
| --- | --- | --- | --- |
| 英语 dev | 36 | 1.0 / 1.0 | 未证明 B 的主指标收益 |
| 英语 validation | 18 | 1.0 / 1.0 | 按预先规则保留较简单 A |
| 中文短摘要 dev | 16 | 0.6875 / 1.0 | 差异集中在少量资料主题 |
| 中文短摘要 validation | 7 | 0.9286 / 1.0 | 一题造成主要差异，不推导稳定或大库收益 |
| 中文扩展池 dev | 16 | 0.5625 / 0.6875 | B 新找回两题的证据 |
| 中文扩展池 validation | 7 | 0.9286 / 0.9286 | 主指标持平 |

用项目虚拟环境运行 `scripts/freeze_public_zh_expanded.py` 可以核对衍生冻结文件，加 `--write` 可以重新生成。策略 CLI 的 `--suite` 指向 `evaluation/corpora/public_zh_expanded.v1/suite.json` 即可复跑。先核对漏掉的正例是否进入融合前候选，再决定是否测 C/D。报告仅支持本地小样本离线检索结论。

### 真实检索策略实测

2026-11-19 在同一份英语冻结 validation 上逐题执行，20 题中 18 题有标注证据。参数为 Top-3、候选预算各 20、结果缓存绕过。向量使用真实 MiniLM ONNX，C 和 D 的模型为 `deepseek-flash`。延迟为单次顺序执行的请求侧 P95，建索引耗时单独记录。

| 策略 | 证据 Recall@3 | MRR@3 | P95 | 完整执行 |
| --- | ---: | ---: | ---: | ---: |
| A · 纯向量 | 1.000 | 0.944 | 46 ms | 20/20 |
| B · 混合召回 | 1.000 | 0.972 | 79 ms | 20/20 |
| C · 查询改写，旧提示协议 | 1.000 | 0.972 | 2.56 秒 | 20/20 |
| D · 旧重排协议 | 1.000 | 1.000 | 20.92 秒 | 5/20 |
| D · 精简候选与 Top-K 输出 | 1.000 | 1.000 | 1.93 秒 | 20/20 |

旧 D 有 15 题触及输出上限并回退。D 两次运行合计输出 Token 分别为 **42,541** 与 **572**，均含改写与重排。C 行是旧协议结果，比较新版 C 与新版 D 时应重跑 C。这些结果用于评估排序、延迟和协议完成率，生产语料上的收益仍待验证。

在 `backend/` 配置真实模型环境变量后，用本机已缓存的 ONNX 目录替换示例路径。每次运行使用新的 run ID。

```powershell
$onnxCache = 'C:/path/to/all-MiniLM-L6-v2'
python -X utf8 scripts/evaluate_retrieval_strategies.py --split validation --candidate D --with-baselines --mode real --onnx-cache $onnxCache --enable-provider --run-id example-real-validation-D
```

将候选改为 `C` 并更换运行 ID 可复跑查询改写。报告位于 `data/eval/rag/experiments/<run-id>/`。

### 业务词面候选基准

2026-11-18 本地复测直接执行 `KnowledgeBase.search()`，使用回环 HTTP 的 Chroma 0.5.23（单 worker）和已缓存的 MiniLM ONNX。24 条冻结查询固定向量候选 20、词面候选 200、Top-3。每组预热 1 轮、计时 3 轮，共 72 次请求，其中 69 次有标注证据。计时绕过应用结果缓存、查询改写和模型重排，从工作线程开始计算。120 片来自中文扩展冻结快照，较大池加入同一批合成干扰项。

| 片段数 | 并发 | 词面获取 p50 / p95（ms） | 排序 p50 / p95（ms） | 完整检索 p50 / p95（ms） | 词面触顶 | 混合证据 Recall@3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 120 | 1 | 23.1 / 31.0 | 13.7 / 22.7 | 77.6 / 95.6 | 0/72 | 95.65% |
| 120 | 4 | 62.5 / 109.9 | 15.4 / 26.6 | 213.3 / 275.2 | 0/72 | 95.65% |
| 1,000 | 1 | 58.1 / 76.8 | 20.0 / 22.2 | 117.5 / 139.5 | 63/72 | 54.35% |
| 1,000 | 4 | 100.3 / 149.2 | 19.9 / 30.2 | 247.4 / 304.8 | 63/72 | 54.35% |
| 5,000 | 1 | 195.3 / 233.6 | 17.1 / 19.4 | 253.5 / 292.5 | 72/72 | 21.74% |
| 5,000 | 4 | 288.1 / 390.9 | 17.7 / 26.4 | 404.3 / 530.9 | 72/72 | 21.74% |

取消词面 200 条上限后，1,000 和 5,000 片池分别找回 16 和 25 个原先同时缺于两路候选的证据组。同期分别有 1 和 3 个证据组已进入候选，却未进入 Top-3。按预先确定的 ID 读取同一批候选时，四组 p95 为 19–111 ms，原谓词为 258–546 ms。对照逐项核对 ID、正文和元数据，计时省略了确定候选 ID 的成本。重开装载完成的同一数据库后，触顶和召回率一致，计时有波动。

以上是 200 条预截断仍存在时的历史基准。当前实现对 Chroma 返回的词面匹配集合计算 BM25 和融合分数，再截取 Top-K。常见查询命中大量片段时，传输量和排序成本会增加。原始计时和对照产物保存在维护者本地 `data/eval/`。

工作台的知识新增、上传和版本删除通过同一 Redis 版本协调。多个 API 进程连接同一个 Redis 数据库和 Chroma 知识库时，读取共享版本后再命中进程缓存。变更进行中或 Redis 版本不可用时直接查询知识库。

### 回答与系统验证

2026-11-19 使用英语 dev 的 40 题固定回放。标准证据分支仅适用于 36 道有答案的题。

| 固定回放指标 | 检索证据 | 标准证据 |
| --- | ---: | ---: |
| 生成成功 | 40/40 | 36/36 |
| Judge 完整评分 | 39/40 | 35/36 |
| 标注事实覆盖 | 77/84 | 77/84 |
| 断言证据支持 | 68/80 | 76/79 |

另用真实 B 的检索报告，对 validation 的 `v2-041` 与 `v2-042` 运行 `deepseek-flash` 生成和 Judge。两题各有两个分支，4 个答案均生成成功。修复偏移处理后，8 次 Judge 调用均通过协议校验。检索证据分支的断言支持为 **11/12**，标准证据分支为 **7/7**，其中检索证据分支有 1 条断言支持度未决。样本用于验证真实协议和诊断，未做独立 Judge 校准。

```powershell
python -X utf8 scripts/evaluate_answer_quality.py --split validation --source-report data/eval/rag/experiments/example-real-validation-D/B/report.json --mode real --enable-provider --case v2-041 --case v2-042 --run-id example-real-answer
```

Docker 验收覆盖相同请求 ID 的 8 个并发重试只提交一轮、服务重启后 Redis 回执与历史、Chroma 文档及 SQLite 核验单均保留，以及两个 API 进程之间的知识缓存失效。SSE 在沙盒核验工具执行阶段断流后，用同一 ID 重试拿到原回执，SQLite 新增 1 张核验单，Redis 保留 2 条消息。后端容器换 IP 后，原 Nginx 进程恢复新旧 API 前缀的转发。配置新令牌变量后，Prometheus 的应用采集目标为 `up`。重建回归可用[离线测试](#离线测试)中的脚本复跑。

### 意图路由基准

2026-11-18 的本地冷缓存基准包含 19 个用例，预热 1 轮，计时 3 轮。向量分支使用真实本地 n-gram，LLM 分支使用延迟 40 ms 的固定响应替身。

| 本地配置 | 每轮意图模型调用 | 每轮路由错误 | P50 / P95 |
| --- | ---: | ---: | ---: |
| 对齐基线 | 19 | 3/19 | 42.65 / 56.77 ms |
| 权重归一化，短路关闭 | 19 | 0/19 | 48.07 / 58.41 ms |
| 权重归一化，短路开启 | 11 | 0/19 | 33.04 / 57.60 ms |

语义和路由边界可用 `python -X utf8 -m pytest tests/test_intent_routing.py -q` 回归。耗时来自本地替身基准，记录在维护者的 `data/eval/intent-routing-20261118/`。

### 中文聊天配对实验

#### 最短真实验收

准备 Docker、Python 3.12 虚拟环境和支持工具调用的 Anthropic Messages 兼容模型。在 `backend/` 中，按 [.env.example](.env.example) 创建 `.env`，填入 `ANTHROPIC_API_KEY`、`ANTHROPIC_MODEL` 和适用的 `ANTHROPIC_BASE_URL`。下面的命令会调用真实模型并消耗额度，输出路径需使用新文件名。

```powershell
python -X utf8 -m pip install -r requirements.txt
docker build --target production -t servicepilot-chat-pairs:local .
python -X utf8 scripts/evaluate_chat_pairs.py --image servicepilot-chat-pairs:local --provider-env .env --output data/eval/login-and-ledger.json --case login-and-ledger
```

检查报告中 `routed` 行的 `roles` 包含 `technical` 与 `billing`，工具轨迹包含对 `ORD-20261024` 的账单查询，回复同时保留 401 排查建议、199 元已入账和 199 元预授权冻结。`same_receipt` 和 `one_committed_turn` 都应为 `true`。

Docker 构建会校验 ONNX 归档并复用 BuildKit 缓存。下载受阻时，可以给生产构建追加 `--build-context "onnx_input=PATH"`，其中 `PATH` 是包含原始 `onnx.tar.gz` 的目录，例如 `$env:USERPROFILE/.cache/chroma/onnx_models/all-MiniLM-L6-v2`。同一个模型归档适用于 Docker 和本机 Chroma 客户端。

#### 全量实验与共享缓存验证

```powershell
python -X utf8 scripts/evaluate_chat_pairs.py --image servicepilot-chat-pairs:local --provider-env .env --output data/eval/chat-paired.zh.v2.json --verify-shared-knowledge
```

脚本创建并清理独立容器和网络，两组均走实际 `POST /chat`，使用同一模型、合成账单、演示知识库和请求预算。单 Agent 可使用两域只读工具的并集，多 Agent 沿用角色白名单。实验关闭外部 MCP 与在线意图向量，工具失败用例注入 SQLite 读取错误。

v2 有 10 个用例、11 轮，逐轮记录工具参数、回答、事实规则、HTTP 延迟和供应商报告的 Token。跨域角色直接运行混合召回，单域保留查询改写与模型重排。加 `--verify-shared-knowledge` 时，脚本启动额外 API 进程。实测在读端预热缓存后，从写端更新和删除文档，读端读到了新版，删除后旧片段消失。P95 在这一样本量下等于最慢一轮。

#### 最新 Docker 实测

2026-11-24 使用提交 `0551342` 构建生产镜像，两组均调用 `deepseek-flash`，本地报告 ID 为 `servicepilot-pair-e265c690`。以上全量命令会生成同格式的新报告。

| 指标 | 单 Agent | 多 Agent |
| --- | ---: | ---: |
| 自动评分通过 | 7/11 | 10/11 |
| 逐条复核后通过 | 11/11 | 11/11 |
| 复核后事实覆盖 | 22/22 | 22/22 |
| 预期工具调用 | 15/15 | 15/15 |
| HTTP 中位延迟 / 最慢一轮 | 7.61 / 13.42 秒 | 7.20 / 12.24 秒 |
| 供应商报告输入 / 输出 Token | 40,737 / 6,105 | 56,626 / 8,315 |

按答复原文复核了 8 个漏判的事实，涉及订单摘要中的共享币种、「Token 有效期」和账单查询失败的不同表述。复核后两组通过率持平。本次多 Agent 的中位延迟低 0.41 秒，供应商报告的输入和输出 Token 分别多约 39% 和 36%。延迟和消耗只反映各 11 轮的这一次运行。

- **复合请求**。多 Agent 调用技术和账单两个角色，保留 401 排查、199 元已入账和 199 元预授权冻结。重发同一请求后，两组都返回原回执且各只保留一问一答。
- **默认生产入口**。独立实例收到 8 个相同请求 ID 的并发请求时只提交一轮；换会话和请求 ID 再确认同一订单时，SQLite 仍只有一张退款核验单。同 ID 换消息返回 `409`，其他用户读不到该会话历史。重启 API 容器后重新获取映射端口，仍可取回原回执，SQLite 行数仍为 1。
- **跨进程缓存与网关**。`--verify-shared-knowledge` 的 5 项检查全部通过。用真实 Nginx 和测试 HTTP 上游执行 `python -X utf8 tests/verify_gateway_recovery.py --backend-image servicepilot-chat-pairs:local`，5 种配置均在上游换 IP 后恢复转发，网关进程持续运行。

本轮覆盖后端 HTTP、容器重启、共享知识缓存和网关转发。浏览器工作台及长时间压测待验证。

#### 历史对照与人工复核

2026-11-19 完整真实 API 运行使用 `deepseek-flash`，本地报告 ID 为 `servicepilot-pair-e1c6cf74`。

| 指标 | 单 Agent | 多 Agent |
| --- | ---: | ---: |
| 当时的自动评分 | 11/11 | 10/11 |
| 按修订规则重评分 | 11/11 | 11/11 |
| 预期工具调用 | 15/15 | 15/15 |
| HTTP 延迟 P50 / P95 | 6.71 / 10.06 秒 | 6.26 / 11.57 秒 |
| 供应商报告输入 / 输出 Token | 39,646 / 5,488 | 53,003 / 8,551 |

同一订单的共享币种与金额表述造成了一次多 Agent 评分漏判。规则修订后单独重跑 `--case correction`，两组各 **2/2 轮通过**。完整对照中，多 Agent 的中位延迟较短，尾延迟和供应商报告的 Token 较高。

对规则遗漏的改述，可用 `--review-report 原报告.json --reviews 复核记录.json --output 复核报告.json` 在离线模式附加人工判断。复核记录包含原报告的 `run_id`、`suite_sha256` 和 `reviews` 数组。每条判断带 `arm`、`case`、`turn`、`fact`、`covered`、`reviewer`、`reason` 与答复原文摘录 `quote`。复核报告保留原自动判定及其汇总。

待扩大验证的范围包括真实 C/D 在更大语料中的收益、不同任务分布下的单 Agent／多 Agent 表现，以及多 worker 长时间并发写入和依赖故障恢复。业务检索判定使用有限规则，知识和身份映射面向本项目演示场景。
