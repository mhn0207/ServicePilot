# ServicePilot 工作台

Vue 3 / Vite 工作台，连接同仓库 `backend/`，提供聊天、知识导入与检索、运行详情和异步评测展示。

## 本地开发

先准备后端，再从本目录执行：

```powershell
npm ci
npm run dev
```

开发地址 http://localhost:5173；`/api/zhiying` 代理到本机 8000 端口并去除前缀。模型 Key 只放后端。连接设置中的访问 Token 用于 ServicePilot 身份验证，不是模型供应商 Key。

## 构建与验证

```powershell
npm test
npm run build
```

测试覆盖 API 字段适配、轮询、旧响应隔离、RAG 任务绑定、会话恢复及知识维护。前后端集成使用同仓库 Compose 配置，运行方式见 [项目说明](../README.md)。

## Docker

推荐从 `../backend/` 使用 `docker-compose.yml` 与 `docker-compose.web.yml` 统一启动，命令见 [根 README](../README.md)。本目录 Compose 是独立方案，与统一方案端口重叠，选择一种即可。

运行时 API 地址由 `PYTHON_API_URL` 注入，默认 `/api/zhiying`。不要把模型凭据写入 Vite 环境变量、`runtime-config.js` 或前端构建产物。

入口 Shell 脚本使用 LF 换行，镜像构建也会清除 CRLF，避免 Windows 工作区构建后出现 `exec /entrypoint.sh: no such file or directory`。前端镜像用页面 HTTP 探测确认 Nginx 可用；它不替代后端 `/health` 或模型连接检查。

## 页面与后端

| 页面能力 | 后端接口 |
| --- | --- |
| 聊天 | `POST /chat` |
| 刷新恢复近期消息 | `GET /conversation/history` |
| 工具详情 | `GET /trace/tool/{request_id}` |
| 知识检索 | `POST /search` |
| 添加/上传知识 | `POST /knowledge/add`、`POST /knowledge/upload` |
| 知识数与 Skills | `GET /knowledge/stats`、`GET /skills` |
| 文档列表、片段与按版本删除 | `GET /knowledge/documents`、`GET/DELETE /knowledge/documents/{document_id}` |
| 规则重载 | `POST /skills/reload` |
| 旧评测 | `POST /eval/run`、`GET /eval/jobs/{job_id}` |
| RAG 离线验收 | `POST /eval/rag/run`、`GET /eval/rag/runs` |

普通用户不能访问所有管理与轨迹接口，服务端执行最终权限检查。新 RAG 区块显示权限不足；部分旧区域仍会静默隐藏读取失败。

文档管理在读取成功后展示列表，展开可查看完整证据 ID、版本与片段正文；删除需要再次确认，使用预览的内容版本。409 提示刷新并重新查看，网络/存储异常提示先核对结果，不自动重试删除。操作后刷新列表和统计，清除旧检索展示；跨越知识变更的在途检索响应不能重新显示旧资料。

启用鉴权后普通用户收到明确权限提示，不显示文档删除及导入表单；无鉴权 development 用于本机管理演示。缺少身份或片段不完整的旧资料仅查看，不提供删除；本次没有增加文档编辑器、回滚、批量删除或完整审批。

## 评测边界

**RAG · Agent 离线验收**只运行固定 `agent-dev-v1`（实验 B / Top-K=3），使用 fixture/synthetic、`scope=orchestrator`。它显示分子分母、N/A、未决及逐轮证据，不能据此宣布真实质量通过。旧四维评测与聊天/搜索可能调用真实模型并收费。

RAG 任务按 suite、服务地址、用户及令牌摘要绑定，sessionStorage 不存原始令牌。刷新只恢复已知 job，不重新提交；首次 POST 结果未知时保留未知状态，需核对历史。404 提示过期；后端重启不恢复运行中任务。切换连接或任务后，旧响应不能覆盖当前结果。

## 会话与失败边界

主动开始新对话会生成新的会话 ID 并失效旧请求代次；此前未完成的相同首句不会成为新会话的重试。同一会话内同一失败操作重试仍保留原请求 ID。

页面加载及连接配置变更后，按当前身份与会话 ID 恢复后端保留的近期已提交消息。恢复中暂停发送；失败时提供“重试恢复”及“新对话”，恢复成功或主动开始新对话前不继续发送。访问令牌不保存在浏览器中，启用鉴权后刷新需重新填写令牌。切换身份/服务或开启新对话后，旧恢复响应不得覆盖新页面。

恢复只展示当前 Redis 工作记忆，不是完整历史档案；最多 20 条，压缩、24 小时过期后可能减少或为空。恢复的消息不重新调用模型，也不恢复原来的运行详情；刷新时仍在处理、尚未提交的请求不在快照内，需要完成后再次刷新。

检索接口即使返回 HTTP 200，也必须明确 `success=true` 才作为成功处理；`success=false` 显示“检索失败”，成功的空结果才显示“暂未找到相关知识”。切换连接会清除旧错误和结果，过期响应不能覆盖当前页面。
