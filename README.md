# python-code

多业务 Python 工作区。目录按 **apps / packages / experiments / materials** 分层。

## 目录约定

```text
apps/           可运行业务
  message/      聊天室抓取 + 热点分析 (Flask :5000)
  tool/         视频下载 / ASR / 转录精炼 / 逻辑分析 (Flask :5001)
  market_charts/ A股指数与板块量能、资金流柱状图 (Flask :5002)
  cursor_chat/  Cursor SDK 聊天 (Flask :5055)
packages/       可复用库（lite_agent_sdk）
experiments/    实验与草稿（非生产）
materials/      大体积资料（证券书籍等，默认不提交）
```

## 环境

在仓库根目录操作。建议将 `apps` 与 `packages` 加入 `PYTHONPATH`：

```powershell
# Windows PowerShell
$env:PYTHONPATH = "apps;packages"
```

或安装为可编辑包（见 `pyproject.toml`）：

```powershell
pip install -e .
```

各业务依赖见对应目录下的 `requirements.txt`。

## 启动

### message（聊天室）

```powershell
cd apps\message
# 复制 config.example.json → config.local.json 并填入密钥
python message.py
```

分析某日消息：

```powershell
python analyze_with_deepseek.py --date 2026-07-01
```

### tool（下载 / 转写 / 投研）

```powershell
# 在仓库根，需 PYTHONPATH 含 apps 与 packages
python apps\tool\app.py
# 浏览器 http://127.0.0.1:5001
```

本地密钥（勿提交）：

- `apps/message/config.local.json`
- `apps/tool/wind.local.json`
- `apps/tool/wanxing.local.json`
- `apps/tool/api_key.local.json`、cookies 等

### market_charts（量能 / 资金流）

```powershell
$env:PYTHONPATH = "apps;packages"
pip install -r apps\market_charts\requirements.txt
python apps\market_charts\app.py
# 浏览器 http://127.0.0.1:5002
```

数据来自东财公开接口（经 AkShare / 直连），无需本地密钥。

### cursor_chat

```powershell
$env:PYTHONPATH = "apps;packages"
python -m cursor_chat          # CLI
python -m cursor_chat web      # http://127.0.0.1:5055
```

配置：复制 `apps/cursor_chat/config.example.json` → `config.local.json`。

## 说明

- `tool` 的 LLM 配置复用 `apps/message` 的 `config_loader` 与 `packages/lite_agent_sdk`
- `experiments/`、`materials/` 不参与生产流水线
- 运行产物（`processed/`、`analysis_results/`、`message_data/`）已在 `.gitignore` 中忽略
