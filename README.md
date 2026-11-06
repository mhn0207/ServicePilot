# ServicePilot

面向企业服务场景的多 Agent 协同应用。由服务协调、技术可靠性、收入与合规 Agent 处理咨询、排障与账单核验建议，支持人工交接摘要。

本仓库为 mhn 维护的个人迭代版本，统一收纳 Python 后端与 Vue 工作台。业务工具目前提供核验建议、金额比较和交接摘要；不执行真实订单查询、退款、工单创建或人工排队。

## 当前能力

- FastAPI HTTP API、主辅 Agent 路由、并行执行及结果汇总。
- Agent 按需调用 RAG；向量与词面混合检索、查询改写、并行召回和模型重排。
- Redis 会话消息与状态、ChromaDB 摘要与画像；会话锁、续租、重复请求回执和整轮事务提交。
- 工具参数与权限检查、超时、缓存、熔断和降级；标准 MCP 客户端与服务端。
- Markdown Skills 动态加载与热更新，保留运行所需的 SKILL.md。
- 请求级阶段耗时与供应商 token 统计、工具轨迹、Prometheus 指标和路由降权恢复。
- 异步意图与对话质量评测、Judge 失败分离及兼容基线回归比较。

上述为实现范围，不代表生产性能承诺。检索评测目前仅有 12 条冒烟样例；尚未完成下述 RAG 评测升级。

## 目录

```text
backend/              Python 后端、测试、部署与示例配置
  skills/            Markdown 业务规则
  data/demo_docs/    可公开的示例知识 JSON
frontend/      Vue 工作台与测试
```

本地数据库、会话、画像、模型凭据、历史评测、教程与简历材料均不纳入仓库。Markdown 跟踪根 README、前后端基础 README 与运行所需的 SKILL.md；教程、简历包装及历史更新说明不纳入。

## 本地启动

需要 Docker / Docker Compose。后端模型服务需兼容 Anthropic Messages 和工具调用；不绑定某个一定可用的模型 ID。

在仓库根目录执行 PowerShell：

```powershell
Copy-Item backend/.env.example backend/.env
# 编辑 backend/.env：填写 ANTHROPIC_API_KEY、ANTHROPIC_MODEL、ANTHROPIC_BASE_URL
Set-Location backend
docker compose -p servicepilot-local -f docker-compose.yml -f docker-compose.web.yml up -d --build
```

这是新项目的数据环境，不复用旧目录的数据库。默认使用 development 演示模式和示例知识；外部 MCP 默认关闭。Prometheus 经应用 `/metrics` 采集，无需额外监听端口。

- 工作台：<http://localhost:5174>
- Swagger：<http://localhost:8000/docs>
- 健康检查：<http://localhost:8000/health>
- Prometheus：<http://localhost:9090>

端口若已被原项目占用，先停止原项目或调整新配置。后续管理命令保持相同项目名与两个 Compose 文件，不混用前端目录的另一套独立部署配置。

单独开发前端时：

```powershell
Set-Location frontend
npm ci
npm run dev
```

开发地址为 <http://localhost:5173>，`/api/zhiying` 代理到本机 8000 端口。模型密钥只放后端。对外部署需要改为 production 并设置 ZHIYING_API_TOKENS 等身份配置，开发默认值不适合作为公开服务配置。

## 验证

Python 3.12，后端目录：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

前端目录：

```powershell
npm ci
npm test
npm run build
```

已有检索冒烟脚本需要运行中的后端，可能消耗模型额度，不会随单元测试自动执行：

```powershell
python scripts/evaluate_retrieval.py --base-url http://localhost:8000 --top-k 3
```

它以预期文档标题计算 Hit@K 与 MRR@K；结果不是片段证据覆盖率，也不是最终回答准确率。

## RAG 评测升级路线（计划，尚未实施）

### 第一批：检索评测基础

保留 12 条冒烟样例，新增约 24 条开发用例；先稳定片段标识，再标注证据组、知识快照和数据集版本。计算 Hit@3、证据组 Recall@3、MRR@3、执行错误率，输出逐条结果。只有实际包含证据的片段才命中，不能仅凭标题通过。

两个必需证据组只命中一个时，Hit 为 1、证据覆盖为 0.5。无答案用例不参与正向召回指标；请求错误按未命中计入主指标并另列错误率。此批先验证当前默认流水线；该流水线可能调用改写与重排模型，不使用回答模型或 Judge。

### 第二批：检索策略对比

先新增显式策略与缓存配置、独立评测实例，再运行 A 向量、B 混合、C 混合加改写、D 混合加改写与重排。默认保持现有线上行为。记录配置路径与实际路径，降级不可冒充完整执行；缓存键包含检索配置和知识版本，评测绕过流水线及底层结果缓存。

固定 Top-K、单次候选预算和输入；各方案返回最终 Top-K，不能直接比较不同截断阶段。记录额外召回次数、去重候选数、阶段耗时、模型调用与已报告 token。缓存加速单独比较。

逐步形成约 40 条开发用例、20 条冻结验证用例；同源变体整组划分。只有获胜候选进入验证集，避免反复对验证集调参。小样本展示计数与波动，不宣称稳定的生产准确率。先判断改写/重排的收益，再决定保留哪些策略。

### 第三批：回答质量

固定回答模型、提示词和生成配置，分别基于实际检索片段与人工正确片段作答，区分召回问题和生成问题。逐条检查必要事实、依据一致性、禁止结论及无答案处理；确定性检查优先，语义 Judge 辅助并人工抽查。

Judge 失败不填默认分、不计入有效质量均值；Agent 执行失败单独计入失败。记录 Judge 有效率，评分不完整不得判回归通过。多轮每轮有独立预期，不泄漏后续答案。

### 第四批：Agent 与展示闭环

先检查实际检索调用和查询质量，再评估多轮省略、切换信息和复合诉求。区分必需、可选和不适用检索；不能将未调用工具一律判错。关联 run_id、case_id、request_id、知识片段、工具明细和评分，复用异步任务接口与评测页面。

Orchestrator 用例与完整 /chat 用例分开报告；后者只取少量场景验证记忆、回执和压缩，使用独立用户与数据环境。评测实例不继承在线用户历史，不执行有副作用的外部工具。

### 共通验收与报告

- 保存数据集/知识快照/代码与工作区标记/检索配置/生成与 Judge 配置，报告能定位每个失败样本。
- 实验身份与评分协议身份分开：方案配置可不同，比较必须保持同一数据与评分协议；不直接套用现有对话 suite_id。
- 保存追加式运行报告，基线显式晋升；新运行不得自动覆盖基线。Judge 错误或兼容性变化时不判通过。
- 汇总分开呈现意图、检索、对话与 Judge 有效性，不把混合粒度的总通过率称为准确率。
- 保留严重错误的逐例检查；收益同时报告质量、耗时和 token，不设不可解释的综合总分。

实施顺序合理：先校准测量，再比较策略，然后评估回答，最后接入 Agent 与页面。每批先做必要的指标、边界及集成验证；尚未执行模型实验，当前没有新效果指标。

## 名称兼容

项目与页面名称为 ServicePilot，前后端目录为 backend 和 frontend。现有 `/api/zhiying`、`ZHIYING_*` 环境变量、运行时配置键和容器服务标识保留兼容，避免品牌调整引入协议与存储迁移。新环境请使用 `servicepilot-local` Compose 项目名。

## 来源与维护

维护者：mhn。本版本基于本机 ZhiYing / ZhiYingFrontend 工作区整理；后端原 Git 来源为 [Biscuit-AI531/EchoMind](https://github.com/Biscuit-AI531/EchoMind)，本次整理前 HEAD 为 b1adff4，工作区含未提交迭代。目录整理、文档提交规则整理与后续开发不构成全部代码原创声明。

本次未在源工作区发现项目级 LICENSE / NOTICE，未擅自新增授权许可证；后续公开分发前应核实上游授权。依赖软件按各自许可证使用。
