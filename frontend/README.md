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
