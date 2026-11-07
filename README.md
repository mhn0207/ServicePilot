# ServicePilot

面向企业服务咨询的 AI 应用：围绕技术故障、账单争议与订阅咨询，整理用户信息、检索知识、生成排查与核验建议，并提供人工交接摘要。

Python / FastAPI 后端与 Vue 3 工作台统一维护。项目基于 ZhiYing / EchoMind 相关代码学习与二次开发，来源与授权边界见文末。

## 当前能力

- **聊天与协作**：意图识别、业务实体继承和更正、主辅 Agent 路由、并行执行、结果汇总及降级。
- **RAG**：500 字规则切片、向量与词面混合检索、查询改写、并行召回、模型重排和结果缓存。
- **知识维护**：管理员查看文档与有序片段、按当前版本删除；旧版或不完整片段只读展示。无鉴权 development 仅用于本机演示。
- **会话可靠性**：Redis 会话锁与续租、重复请求回执、消息与状态整轮提交；ChromaDB 存储知识、历史摘要及画像。
- **刷新恢复**：恢复当前身份、当前会话已提交的近期消息；读取失败时提示重试或新建对话，避免空白页面继续使用隐藏上下文。
- **工具与规则**：工具参数校验、角色白名单、超时与熔断，标准 MCP 客户端/服务端，以及 Markdown Skills 加载。
- **观察与评测**：工具轨迹、阶段耗时、供应商用量、Prometheus；证据级检索、独立策略对比、成对回答和 Agent 离线验收。

业务工具只提供字段核验、金额比较、排障建议和交接摘要，**不执行真实订单查询、支付核验、退款、工单创建或人工排队**。多 Agent、RAG 和监控的存在不代表效果或生产性能已经得到充分验证。

## 请求流程

```text
工作台 → FastAPI 鉴权 → 会话锁/旧回执
       → 读取记忆 → 意图与实体 → 单角色或多角色执行
       → 按需调用知识/业务工具 → 汇总
       → Redis 提交消息、状态和回执 → 返回
```

成功请求使用相同 `client_request_id` 重试时返回原回执；可重试失败允许再次执行。可能产生副作用的工具在执行前保存待核验标记，避免将未知结果当成可安全重放。

## 本地启动

需要 Docker / Docker Compose。模型服务需兼容 Anthropic Messages 与工具调用。模型名称和可用额度以自己的供应商账户为准。

从仓库根目录执行 PowerShell；已有配置时不要覆盖：

```powershell
if (!(Test-Path backend/.env)) { Copy-Item backend/.env.example backend/.env }
# 在本机编辑 backend/.env，填写模型 Key、模型名与服务地址。
Set-Location backend
docker compose -p servicepilot-local -f docker-compose.yml -f docker-compose.web.yml up -d --build --wait --wait-timeout 180
```

| 入口 | 默认地址 |
| --- | --- |
| 工作台 | http://localhost:5174 |
| API 文档 | http://localhost:8000/docs |
| 存储就绪检查 | http://localhost:8000/health |
| Prometheus | http://localhost:9090 |

示例配置使用 development 模式和公开演示知识，关闭外部 MCP。默认端口限制在本机；对外部署前需另行配置 production、访问 Token 与网络边界。不要把模型 Key 填到前端。

启动、停止和查看日志保持相同项目名及两份 Compose 文件。前端目录的独立 Compose 是另一种方案，不与上述方案同时使用；本地已有服务占用端口时应先调整本项目配置，避免误操作其他项目。

日常重启不会重新构建代码；修改源码后使用上面的 `up -d --build`。从 `backend/` 执行：

```powershell
docker compose -p servicepilot-local -f docker-compose.yml -f docker-compose.web.yml ps
docker compose -p servicepilot-local -f docker-compose.yml -f docker-compose.web.yml logs --tail 80 zhiying frontend
docker compose -p servicepilot-local -f docker-compose.yml -f docker-compose.web.yml restart
```

重启后检查 `/health` 返回 200，并打开工作台确认当前会话近期消息与已导入知识仍可读取。`/health` 只检查存储就绪，不验证模型凭据或额度。不要用 `down -v` 作为日常重启，它会删除 Compose 命名卷中的数据。首次构建需要联网安装镜像内依赖并下载 embedding 模型。

## 最短演示

1. 提问“企业服务登录返回 401，应该如何排查？”，查看回复和运行详情。
2. 提供一笔订单及金额，再更换订单，观察实体继承与清除。
3. 在知识库页检索公开示例；以管理员身份添加一份演示文档，展开查看片段，再确认删除该测试文档，观察列表和数量变化。
4. 评测页运行 **RAG · Agent 离线验收**，查看逐轮工具与证据记录。

聊天、知识搜索以及旧四维评测可能调用模型并产生费用。RAG Agent 离线验收使用固定模型响应和模拟向量；它证明工程路径，不证明真实回答质量。管理员功能需要相应权限。

## 验证与实验结论

- Python 3.12；后端从 `backend/` 执行 `python -m pytest tests -q`。测试入口阻止加载本机 `.env`。
- 前端从 `frontend/` 执行 `npm test` 与 `npm run build`；所需依赖见各目录清单。
- 默认离线评测命令、退出码、真实依赖开关见 [后端说明](backend/README.md)。

维护者保留的本地小样本检索记录显示：英语合成集的向量 A 与混合 B 在主证据 Recall 上均达到 1.0，未证明 B 的主指标收益；中文 12 片候选池中 B 的排序/证据覆盖较好，但验证集差异只涉及少量问题。原始运行报告未随仓库分发，这些摘要不是可直接推广的效果承诺。

真实 Flash 小样本与本地聊天冒烟已经执行；尚无充分的回答准确率、独立 Judge 校准、多 Agent 对单 Agent 收益或大规模性能结论。已补充建议式检索、前端会话/失败边界、存储就绪检查及专用指标采集身份的修复。启用 API 鉴权后，按 [后端说明](backend/README.md) 配置 Prometheus 的令牌文件；默认空示例仅用于无鉴权本地开发。

2026-11-08 在独立 Compose 项目中完成前后端 Dockerfile 构建、六服务启动及整组重启，验证近期消息恢复、用户键隔离和新增知识保留；同时修复 Windows Shell 换行导致的前端重启失败，并增加页面健康探测。验收使用新数据卷及不可调用模型的测试配置，未读取原业务数据、未新增真实模型调用，不能据此验证供应商凭据或额度。

## 目录与本地文件

```text
backend/                 API、Agent、记忆、工具、评测、测试与 Docker 配置
  data/demo_docs/        可公开的演示知识
  evaluation/corpora/    冻结实验语料、来源与标注
  evaluation/fixtures/   固定模拟响应及回归样本
  skills/               运行所需业务规则
frontend/                Vue 工作台、API 适配及测试
```

源码、测试、依赖锁文件、示例配置与复现实验所需的静态数据保留在版本管理中。真实 `.env`、本地虚拟环境、依赖目录、数据库、会话、画像、运行报告、构建产物、截图及个人复盘不提交。旧版静态标注/fixture 仍被兼容性测试或报告重放使用，不作为临时垃圾删除。

前端开发与交互说明见 [工作台说明](frontend/README.md)。项目协议保留 `/api/zhiying`、`ZHIYING_*` 和现有服务标识兼容。

## 来源与授权

维护者：mhn。本版本基于本机 ZhiYing / ZhiYingFrontend 工作区整理；后端原 Git 来源为 [Biscuit-AI531/EchoMind](https://github.com/Biscuit-AI531/EchoMind)，整理前 HEAD 为 `b1adff4`，当时工作区包含未提交迭代。

当前本地 Git 从整理后的个人基线开始。目录整理和后续提交不构成全部代码原创声明，也不能单独证明个人独立完成所有设计。

整理时未在源工作区发现项目级 LICENSE / NOTICE，未擅自新增授权许可证；公开分发前需核实上游授权。依赖软件按各自许可证使用。
