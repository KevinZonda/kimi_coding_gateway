# Kimi Coding Plan Gateway

> 由 [yulate/kimi_coding_gateway](https://github.com/yulate/kimi_coding_gateway) 改进而来。

这是一个 API 网关，用于将 OpenAI 兼容的 API 请求转发到 Kimi Coding Plan API (`api.kimi.com/coding`)。
它解决了 Kimi Coding API 的访问限制问题，并提供了完整的 OpenAI 兼容性，支持 Tools Calling 和 MCP 协议。

## 项目结构

```text
kimi_coding_gateway/
├── app/
│   └── main.py          # 网关核心服务
├── tests/
│   ├── test_basic.py    # 基础对话测试
│   ├── test_tools.py    # 工具调用测试
│   └── ...
├── requirements.txt     # 依赖列表
├── .env.example         # 环境变量模版
└── run.sh               # 启动脚本
```

## 快速开始

### 1. 配置

复制模版文件并配置 API Key：
```bash
cp .env.example .env
# 编辑 .env 文件填入你的 KIMI_API_KEY
```
*(如果不想创建 .env 文件，网关也可以直接读取环境变量)*

如需开启调试日志（完整记录每次请求/响应 JSON 到 JSONL）：
```bash
GATEWAY_DEBUG=true
GATEWAY_DEBUG_JSONL_FILE=./logs/gateway_requests.jsonl
```

如需网关默认注入推理强度（不覆盖客户端显式传值）：
```bash
GATEWAY_REASONING_EFFORT=high
```

### 2. 启动服务

使用启动脚本（自动安装依赖）：
```bash
./run.sh
```

或者手动启动：
```bash
pip install -r requirements.txt
python3 app/main.py
```

### 3. 使用方法

服务默认监听页面 `0.0.0.0` (允许局域网访问)。

启动后，控制台会打印本机局域网 IP，例如：
```text
[INFO] 局域网访问地址: http://192.168.1.5:8765
```

> [!WARNING]
> 开启局域网访问 (0.0.0.0) 意味着同一网络下的任何人都可以连接此端口使用你的 API Key。请确保在可信网络环境中使用，或通过防火墙限制访问。

#### Python SDK 集成
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="ignored",  # 网关处理认证
)

response = client.chat.completions.create(
    model="kimi-for-coding",
    messages=[{"role": "user", "content": "你好"}]
)
```

#### MCP Agent (Claude Code / Cline) 集成
- **Base URL**: `http://127.0.0.1:8765/v1`
- **API Key**: 任意值
- **Model**: `kimi-for-coding`

## Debug JSONL 日志

开启 `GATEWAY_DEBUG=true` 后，网关会对每次请求追加一行 JSON 到 `GATEWAY_DEBUG_JSONL_FILE`。

每行会包含：
- `request_id`、`timestamp`、`method`、`path`、`target_url`
- `request_json`（原始入站请求 JSON）
- `forwarded_request_json`（网关修复后的转发 JSON）
- `status_code`
- `response_json`（非流式为完整响应 JSON；流式为 SSE 中每个 `data:` 的 JSON 事件列表）

## 测试

在服务运行的情况下，执行测试脚本：

```bash
python3 tests/test_basic.py
python3 tests/test_tools.py
```
