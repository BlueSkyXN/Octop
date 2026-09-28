# 飞书修正：迁移到开源组件

## 基线与仓库

本次从 Octop `upstream/develop` 的 `4667bd7b` 创建
`feature/feishu-component-migration`。它与当时的 `upstream/main`（1.0.2b4）
代码相同，差异仅为发布版本及文档。旧 `fork/image`、`feature/feishu-stream-card`
保留原状；本分支保留原镜像发布工作流，但只推功能分支，不打 `img-v*` 标签，
不自动构建或部署生产镜像。迁移完成后记录 `fork/image@86226c65` 为 merge parent，
防止日后合回发布分支时把旧 monkey-patch 文件重新引入；其代码以本文逐项归属为准。

GitHub fork 保持官方名称；本地都在 `/Volumes/TP4000PRO/Program`：

| 官方组件 | Fork | 本地目录 | 本次变化 |
|---|---|---|---|
| TencentCloud/octop-harness | BlueSkyXN/octop-harness | octop-harness-self | 命令取消、执行预算公共接口、MCP 参数和流上下文 |
| TencentCloud/octop-gateway | BlueSkyXN/octop-gateway | octop-gateway-self | 飞书原生卡片、表情生命周期、安全提及识别、轻量 probe |
| TencentCloud/octop-memory | BlueSkyXN/octop-memory | octop-memory-self | 无相关修正，保持上游 |
| TencentCloud/octop-browser | BlueSkyXN/octop-browser | octop-browser-self | 无相关修正，保持上游 |

## 旧修正的去向

| 旧实现 | 新归属 / 处理 |
|---|---|
| `feishu_compat.ensure_feishu_ws_stop_fix` | 上游 Gateway 已有完整 WS teardown/watchdog，不再移植 |
| `FeishuHardenedChannel` 的 bot identity、`chat_type`、mention 接线 | 上游已原生提供；补齐未知身份不激活、排除 `@all`、恢复真人名字 |
| 子类替换 `group_context_manager.should_persist_media` | Gateway 飞书 adapter 原生 `_persist_media`；未提及机器人不下载被动群媒体 |
| 入队即加 Typing、主库按会话扫反应 | Gateway 在通过群策略后添加，当前轮 `try/finally` 删除；`merge_inbound` 显式保留消息 ID |
| `feishu_card.py` | Gateway 原生 CardKit helper，`FeishuConfig.stream_card` 控制；不依赖 Octop 包 |
| 仅主库的 Feishu credential probe 分支 | `BaseChannel.probe()` 公共钩子，Feishu HTTP token 校验；主库仍统一 `manager.probe_channel()` |
| `execute_guard` 替换两类 backend 方法和版本探测 | Harness 原生命令执行器及 `execution_scope`；无 monkey-patch，无内部方法替换，无旧包名版本条件 |
| `ExecuteBudgetClampMiddleware` | Harness 执行器按 scope 剩余预算缩短 timeout，也覆盖没有显式 timeout 的命令 |
| `turn_budget`、心跳、超时文案 | Octop 业务策略保留；固定单个 producer task 消费源流，使用 Harness 公共执行 scope |
| `/stop` 外挂进程表扫描 | 使用原有 manager cancel；Harness 异步 execute 取消直接清理本轮本地命令 |
| `gateway_args_model` 重复 Pydantic 生成器 | 修复 Harness `mcp_args_model`，主库直接复用，保留上游别名和 null 处理 |
| 连接器只取首个 text block | Octop 修复 wire/in-process 两条链：合法 MCP `image/data/mimeType`，LangChain 转图片块，保留所有文本 |
| `lark-cli` 消息资源下载 | Octop connector adapter 保留；限制临时目录、魔数与大小，图片块开关兼容原环境变量 |
| 前端流式卡片和预算设置 | Octop dashboard 源码；布尔值不经过 `String(false)`，新增中英文文案及开关回归测试 |
| fork 镜像 CI、tag 规则、服务器薄层配置 | 不属于组件库；原样保留镜像 CI 文件，旧 `fork/image` 指针不动，不打 tag、不发布 |

## 依赖与复现

主库 `pyproject.toml` 以完整 Git commit SHA 固定修改后的 Harness 与 Gateway，
`uv.lock` 同步。不是本机绝对路径依赖、不是浮动 feature 分支，也不依赖修改
`site-packages`。安装 wheel 时的依赖同样指向这些 Git commits。
这是 fork 专用组合；向官方提交时应先提交组件修正，等官方版本发布后换回
对应的 PyPI 最低版本要求，不能直接把 fork Git URL 带进官方发版。

```bash
uv sync --locked
make all
make build-frontend
```

跨库同时开发可在 `uv sync --locked` 后临时覆盖，且不要提交本地路径：

```bash
uv pip install --no-deps -e ../octop-harness-self -e ../octop-gateway-self
uv run --no-sync pytest -m 'not live'
```

## 行为和边界

- `stream_card` 默认关闭；已启用的旧配置继续有效。`response_mode=stream`
  才展示实时文本及工具过程；invoke 仍只发最终内容，但心跳不再被折叠丢弃。
- 预算默认 600 秒，允许 60–3600 秒；心跳默认 60 秒。
- 群策略仍遵循已有 `group_context` 配置，不静默改变开启状态。
- Typing 在实际开始处理后出现，而不是刚入队就出现，避免排队或被动群消息
  留下无法清理的表情。批量消息逐条添加并清理。
- CardKit 终态幂等；更新失败保留完整纯文本回退，ERROR 后 COMPLETED 不会
  把失败标成成功。工具提示服从开关；思考内容不混入最终正文。
- 命令进程组清理适用于 POSIX 本地 shell/bubblewrap；Windows 只终止直接进程。
  远程 sandbox 使用自身的 timeout/取消实现，不宣称已支持其进程树清理。
- 本次无真实飞书凭据调用、无真实 LLM 调用、无生产部署。macOS 上不能代替
  Linux 真 bwrap、Windows、远程数据库及真实 CardKit 验证。

## 提交组织

组件修正分别在 `feature/feishu-runtime-hardening`（Harness）和
`feature/feishu-native-hardening`（Gateway）；Octop 接线在
`feature/feishu-component-migration`。全量门禁及 Git hooks 必须通过后提交。
主库先提交新的组件接线，再以保留新树的 merge commit 记录旧补丁已全部处理，
不把旧代码机械合回来。给官方的 PR 应从干净 develop 挑选相关改动，不包含 fork
镜像工作流与 Git 依赖锁。

与迁移无关但阻挡 Harness 离线测试的两条 web-fetch 用例单独提交：
它们已有假 HTTP 客户端，却遗漏 DNS 隔离，已按相邻用例补 mock，未放宽生产 SSRF 检查。
