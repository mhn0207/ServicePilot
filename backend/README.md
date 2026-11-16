# ServicePilot 后端

Python 3.12 / FastAPI / Redis / ChromaDB / Anthropic SDK / MCP。

本页描述当前接口、验证入口和实验边界。项目功能、最短演示与来源说明见 [根 README](../README.md)。

## 配置与运行

从本目录使用 `.env.example` 建立本机 `.env`，已有文件不覆盖。填写 `ANTHROPIC_API_KEY`、`ANTHROPIC_MODEL` 与兼容 Anthropic Messages 的 `ANTHROPIC_BASE_URL`。模型凭据不提交、不传给前端。

```powershell
docker compose -p servicepilot-local -f docker-compose.yml -f docker-compose.web.yml up -d --build --wait --wait-timeout 180
```

示例配置的 `ZHIYING_ENV=development` 用于本机演示，并允许空库导入 `data/demo_docs/sample_knowledge.json`。production 必须设置 `ZHIYING_API_TOKENS`；ChromaDB 连接失败时中止，禁止回退到本地库。外部 MCP 由示例配置中的 `MCP_EXTERNAL_ENABLED=false` 关闭；直接使用进程环境启动时也应显式设置此值。

若容器调用模型需要宿主机代理，须单独配置容器可访问的代理地址。容器中的 `127.0.0.1` 指向容器自身，不能直接复用宿主机回环代理。不要为排查连接问题关闭证书校验。

## 主要接口与身份

| 接口 | 用途 |
| --- | --- |
| `POST /chat` | 聊天；重试沿用 `client_request_id` |
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

文档管理沿用导入时生成的文档身份摘要，不使用标题作为删除定位，也不接收磁盘路径。列表只读取元数据，正文在展开文档时读取；接口禁止缓存。管理员确认删除时提交所查看的内容版本；已更新或身份/片段不完整返回 409，不存在返回 404，非法身份/版本返回 422，存储异常返回通用 503。

删除清理该文档全部片段，不能撤销；文档中的知识可能影响所有聊天用户。缺少文档身份的旧片段单独只读展示，缺版本、混合版本或缺片段的文档禁止删除，不自动迁移或猜测归属。管理扫描上限为 5000 片，超限明确报错，不静默截断列表。

导入与版本检查/删除共用当前进程的写锁；这不是多 worker 的全局事务。删除工作线程结束后，不论成功或异常，都会在事件循环清除业务与旧评测工具缓存；即使 HTTP 等待被取消，也在实际存储操作结束后清理。HTTP 失败可能表示结果未知，应刷新核对，不自动重放删除；已经取得证据的在途请求并不因此被撤回。

`/health` 在两秒总预算内检查 Redis、记忆/画像及知识库正在使用的 Chroma collection。远端 Chroma 采用有一秒传输超时的独立异步 count 请求，API 版本跟随实际客户端；本地模式合并未完成探测，避免取消等待后重复创建阻塞线程。未初始化、依赖异常或超时返回 503，恢复后返回 200；不会调用模型、生成 embedding 或写业务数据，也不证明模型 Key、额度或模型名称可用。

### Prometheus 采集身份

无鉴权 development 模式默认使用空的 `config/prometheus.token.example`。开启 API 鉴权后，为采集生成独立随机令牌文件，不要复用模型 Key：

```powershell
# 从 backend 执行；文件已存在时拒绝覆盖。
python -c "from pathlib import Path; import secrets; Path('config/prometheus.token').open('x', encoding='utf-8').write(secrets.token_urlsafe(32) + '\n')"
```

在本机 `.env` 设置 `ZHIYING_METRICS_TOKEN_PATH=./config/prometheus.token`，再使用相同 Compose 命令更新服务。Compose 将该文件作为 secret 同时提供给应用和 Prometheus；应用读取 `ZHIYING_METRICS_TOKEN_FILE=/run/secrets/metrics_token`，Prometheus 使用 `authorization.credentials_file`。非 Compose 启动时，直接设置应用的 `ZHIYING_METRICS_TOKEN_FILE` 为本机文件路径。

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

## 离线测试

使用项目自己的虚拟环境；安装步骤仅在准备环境时执行：

```powershell
python -X utf8 -m pip install -r requirements-dev.txt
python -X utf8 -m pytest tests -q
```

`tests/conftest.py` 在 pytest 生命周期内禁用 dotenv 文件解析，覆盖 `api.main` 的 `load_dotenv` 及 MCP/Chroma 配置依赖的 `dotenv_values` 路径；测试配置通过 `monkeypatch.setenv` 等方式显式注入，临时假配置文件仍可创建。单元测试以固定模型、FakeRedis、MockTransport 或注入的 embedding 为主；不能由测试通过推导真实模型质量或真实存储在所有故障下的行为。`tests/verify_*.py` 与在线脚本是额外入口，执行前确认隔离环境。

自动化测试覆盖编排、会话重试与实体更正、知识维护、权限、存储探测和评测协议。独立 Docker 验证使用新数据卷，覆盖 HTTP 聊天、并发重试、用户隔离、Redis 故障恢复及重启后消息、知识和报告保留；模型固定响应的验证只检查工程行为。

## 评测入口

以下命令从 `backend/` 执行；Windows 可将 `python` 替换为 `.venv/Scripts/python`。默认命令不读 `.env`、不调用供应商模型。

### 检索证据

```powershell
python -X utf8 scripts/evaluate_retrieval_evidence.py
```

使用 6 篇中文演示知识、6 个片段和 24 条回归用例。默认 fixture 故意含一次执行错误，预期退出码 **1**；0 表示无执行错误，2 表示输入/协议校验失败。退出 0 不等于全部命中。

切片与在线入库共用 `tooling/knowledge_chunks.py`。证据 ID 包含文档身份、原文 SHA-256 版本和片段序号；评分核对实际 ID、标题和正文，拒绝未知/过期证据。旧 Chroma 存储 ID 不变；缺元数据的旧片段不猜造证据 ID。

| 指标 | 含义 |
| --- | --- |
| Hit@K | 前 K 名至少覆盖一组标注证据 |
| Evidence Recall@K | 覆盖的必需证据组数 / 全部必需证据组数 |
| MRR@K | 首个有效证据片段实际名次的倒数 |

两个证据组只覆盖一个时，Hit=1、Recall=0.5。重复片段不增加覆盖，也不删除排名后补足 K；fallback 不算证据。错误保留在主指标分母，并另报错误率和成功子集；无答案题不进入正向召回分母。

旧 `scripts/evaluate_retrieval.py` 的 12 条标题级冒烟保留原含义。证据级在线入口需显式 `--online --base-url ...`，只从进程环境读取 `ZHIYING_EVAL_TOKEN`；它调用业务 `/search`，可能消耗改写和重排额度，不是纯向量基线。

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

### 成对回答质量

```powershell
python -X utf8 scripts/evaluate_answer_quality.py --validate-annotations
python -X utf8 scripts/evaluate_answer_quality.py --run-id example-answer-dev
```

默认使用 B 模拟检索和 v4 固定生成/Judge fixture，报告明确记录混合依赖。retrieved_context 与 oracle_context 独立生成；生成器看不到参考、事实 ID、禁项或另一分支回答。oracle 是诊断对照，不是部署方案。

正确性 Judge 看参考并逐事实/禁项判断；grounding Judge 只看实际展示材料，检查答案断言的依据。引用、回答 span、覆盖和身份可确定性校验，语义蕴含与充分抽取仍需真实 Judge 和人工校准。错误与 uncertain 保留，N/A 不填成 0，不合成一个混合粒度总准确率。

默认 fixture 故意保留漏答、矛盾、违规和未决；退出 1 表示执行/评价有错误或未决，不等同于未生成报告。退出 0 也不表示质量全部合格。当前正式回答协议为英语单轮，不能冒充中文多轮回答评分。

真实回答入口需要显式 `--mode real --enable-provider` 与 `--limit` 或 `--case`，生成/Judge 凭据使用 `SERVICEPILOT_ANSWER_GENERATION_*` / `SERVICEPILOT_ANSWER_JUDGE_*` 进程环境。真实生成消费模拟检索时另需 `--allow-mixed`；不会加载真实 `.env`。

### Agent 与工作台

```powershell
python -X utf8 scripts/evaluate_agents.py --validate-annotations
python -X utf8 scripts/evaluate_agents.py --run-id example-agent-dev
```

固定 `agent-dev-v1` 使用真实 Orchestrator 和工具循环，搭配固定模型与 synthetic 检索。检查多轮实体、角色、检索调用及证据，记录 `scope=orchestrator`；不代表完整 `/chat` 或真实模型质量。

前端 RAG 区块通过异步任务接口运行同一 suite。报告追加到 `data/eval/rag/agents/`，历史终态可跨重启读取；进程内运行中任务不恢复。固定 Judge 判定只做 fixture 验收，不是独立语义模型或人工校准。

所有运行产物写入被忽略的 `data/eval/`，目录拒绝覆盖。报告保留数据/知识/依赖/评分身份，部分入口保存源码白名单归档；不把密钥、业务数据或 `.env` 纳入归档。旧静态标注和 fixture 保留用于兼容性回归与重放。

## 数据与已有实验结论

| 数据 | 规模与划分 | 当前证据边界 |
| --- | --- | --- |
| 中文演示 | 6 篇 / 6 片；24 条证据回归 | 线上开发 seed，不是业务真实政策 |
| 英语合成规则 `suite.v2.json` | 24 篇 / 72 片；40 dev、20 validation | 虚构企业规则；36/18 条正向题 |
| `public_zh.v1` | 12 篇短摘要 / 12 片；16 dev、8 validation | GitHub、阿里云与 Stripe 来源；验证正向题 7 条，按供应商分组 |
| `public_zh_hard.v1` | Redis 资料整理 6 篇 / 12 片；两个 split 各 15 题 | 24 条正向题；来源与语义标注待人工核验，未运行正式策略/生成实验 |

冻结身份校验只保证数据一致，不证明来源现行有效、语义标注正确或人工独立性。中文资料与初始题目由 AI 整理；`public_zh.v1` 的展示标注获维护者认可，不等于逐一审计源网页或校准模型答案。hard 集人工字段仍为空，不覆盖旧集。

### 本地检索记录

以下为维护者本地报告摘要，原始运行产物未随仓库分发。固定 MiniLM 缓存、Top-K=3、预算各 20、bypass；每个 split 单次运行，不含建库，也不是生产尾延迟测试。

| 数据 / split | 正向分母 | A / B 证据 Recall@3 | 判断 |
| --- | --- | --- | --- |
| 英语 dev | 36 | 1.0 / 1.0 | 未证明 B 的主指标收益 |
| 英语 validation | 18 | 1.0 / 1.0 | 按预先规则保留较简单 A |
| 中文短摘要 dev | 16 | 0.6875 / 1.0 | 差异集中在少量资料主题 |
| 中文短摘要 validation | 7 | 0.9286 / 1.0 | 一题造成主要差异，不推导稳定或大库收益 |

中文候选池仅 12 片，主要体现 Top-3 排序，不能据此证明大规模候选召回。两类实验都没有据验证结果调生产默认策略。C/D 真实收益、完整中文回答指标和独立人工校准仍未建立。

### 回答与系统验证

已有小样本中文真实生成与聊天冒烟记录，用于观察检索、角色协作及额外断言等问题；尚未形成完整中文回答指标或独立 Judge 校准结论。

当前仍需进一步验证的范围包括：真实 C/D 策略收益、多 Agent 相对单 Agent 的收益，以及多 worker 下的知识写入与缓存一致性。业务检索判定使用有限规则，知识和身份映射面向本项目演示场景。
