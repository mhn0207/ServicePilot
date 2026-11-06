# ServicePilot 工作台

Vue 3 / Vite 前端，提供聊天、知识管理、工具轨迹和异步评测任务展示。

## 本地开发

先启动相邻 `ServicePilot` 后端，再在本目录执行：

```powershell
npm ci
npm run dev
```

访问 <http://localhost:5173>。默认 `/api/zhiying` 由 Vite 去除前缀后代理到 `http://localhost:8000`。开发代理已修正旧前缀匹配；真实模型 Key 只放后端。

## 验证

```powershell
npm test
npm run build
```

测试验证 API 契约、异步任务轮询与请求状态隔离；生产产物在 `dist/`，不提交。

## Docker

推荐从相邻 `ServicePilot` 目录使用两份 Compose 文件统一启动，见[根 README](../README.md)。前端目录的 Compose 是独立部署方案，与后端统一方案端口重叠，请选择一种；不同项目名的数据卷不自动共享。

容器运行时 `PYTHON_API_URL` 默认 `/api/zhiying`；本机开发默认代理由 `vite.config.js` 配置。聊天页面的连接设置可指定 API 地址。访问令牌仅用于后端身份验证，与模型 API Key 不同。

## Agent RAG 离线验收

评测页增加独立的 RAG 区块，原四维评测仍保留。新按钮只请求 `/eval/rag/run` 的固定 `agent-dev-v1` suite（B / Top-K=3），展示同一 run_id 的终态报告、分子分母/N/A/未决和逐场景逐轮事件。fixture/synthetic 只证明协议执行，真实质量与人工校准待验证。

任务绑定以评测种类、suite、服务地址、用户与令牌摘要区分，sessionStorage 不存原始令牌。刷新仅恢复已知 job，不自动重新提交；首次 POST 网络结果不确定时保留未知状态，需核对历史报告。任务 404 提示过期；进程重启不恢复运行中任务。切换连接或任务后，旧响应不能覆盖当前结果。启用后端鉴权时，普通用户看到权限不足提示，服务端仍执行 admin 校验。

`npm test` 检查轮询、连接代次、未知提交与绑定隔离；`npm run build` 只检查构建。浏览器验收应连接显式隔离的 fixture 服务，不能点击正在使用的真实后端上的旧四维评测按钮。
