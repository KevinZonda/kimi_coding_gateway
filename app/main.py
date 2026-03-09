"""
Kimi Coding Plan API Gateway
将 OpenAI 兼容的请求转发到 Kimi Coding Plan API (api.kimi.com/coding)
"""
import os
import asyncio
import json
import httpx
from datetime import datetime, timezone
from uuid import uuid4
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from dotenv import load_dotenv
import copy

# 加载环境变量
load_dotenv()


app = FastAPI(title="Kimi Coding Plan Gateway")

# Kimi Coding API 配置
KIMI_BASE_URL = "https://api.kimi.com/coding/v1"
# 使用 or 运算符处理空字符串的情况 (当 docker-compose 传递空值时)
KIMI_API_KEY = os.getenv("KIMI_API_KEY") or "sk-kimi"

# 关键：模拟 Kimi CLI 的 User-Agent
KIMI_CLI_USER_AGENT = "KimiCLI/1.3"

# 本地网关配置
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8765"))

# 调试配置
DEBUG_MODE = os.getenv("GATEWAY_DEBUG", "false").lower() == "true"
DEBUG_JSONL_FILE = os.getenv("GATEWAY_DEBUG_JSONL_FILE", "gateway_requests.jsonl")
DEBUG_WRITE_LOCK = asyncio.Lock()
DEFAULT_REASONING_EFFORT = os.getenv("GATEWAY_REASONING_EFFORT", "high")


def parse_json_payload(body: bytes):
    """解析 JSON，失败时保留原始文本与错误信息。"""
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        return {
            "_parse_error": str(exc),
            "_raw": body.decode("utf-8", errors="replace"),
        }


def parse_stream_json_events(raw_stream: bytes):
    """解析 SSE data 行中的 JSON 事件。"""
    text = raw_stream.decode("utf-8", errors="replace")
    events = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError as exc:
            events.append({"_parse_error": str(exc), "_raw": payload})
    return events


def write_debug_line_sync(line: str):
    """同步写入一行 JSONL。"""
    directory = os.path.dirname(DEBUG_JSONL_FILE)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(DEBUG_JSONL_FILE, "a", encoding="utf-8") as file_obj:
        file_obj.write(line)
        file_obj.write("\n")


async def append_debug_jsonl(record: dict):
    """异步追加 JSONL 调试日志。"""
    if not DEBUG_MODE:
        return
    line = json.dumps(record, ensure_ascii=False)
    async with DEBUG_WRITE_LOCK:
        await asyncio.to_thread(write_debug_line_sync, line)


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_v1(path: str, request: Request):
    """代理 /v1/* 路径的请求到 Kimi Coding API"""
    target_url = f"{KIMI_BASE_URL}/{path}"
    return await forward_request(target_url, request)

def parse_reasoning_effect(body_data):
    data = body_data
    model_name = data.get("model", "")

    if "reasoning_effort" in data:
        return data["reasoning_effort"]
    if model_name.endswith("high"):
        return "high"
    if model_name.endswith("medium"):
        return "medium"
    if model_name.endswith("low"):
        return "low"
    if model_name.endswith("off"):
        return None
    return DEFAULT_REASONING_EFFORT

def process_request_body(body: bytes) -> tuple[bytes, bool]:
    """
    处理请求体：
    1. 检测 stream 参数
    2. 修复 Kimi API 对 reasoning_content 的严格检查
    """
    if not body:
        return body, False
        
    try:
        data = json.loads(body)
        stream = data.get("stream", False)

        if data.get("messages") and "reasoning_effort" not in data:
            data["reasoning_effort"] = parse_reasoning_effect(data)

        if 'thinking' not in data:
            data['thinking'] = {}
        
        data['thinking']['type'] = 'disabled' if data["reasoning_effort"] is None else "enabled"
        data['model'] = 'kimi-for-coding'
        
        # 修复逻辑：为 assistant 消息补充 reasoning_content
        messages = data.get("messages", [])
        for msg in messages:
            if msg.get("role") == "assistant":
                # Kimi API 严格要求 Thinking 模型必须包含 reasoning_content
                # 无论是普通回复还是 Tool Call，都可能需要
                msg["reasoning_content"] = msg.get("reasoning_content", "")
        print(data)
        return json.dumps(data).encode("utf-8"), stream
    except json.JSONDecodeError:
        return body, False


async def forward_request(target_url: str, request: Request):
    """转发请求到 Kimi Coding API"""
    # 获取原始 body
    original_body = await request.body()
    # 处理 body（自动修复 messages）
    body, stream = process_request_body(original_body)
    debug_context = {
        "request_id": uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": request.method,
        "path": request.url.path,
        "query": request.url.query,
        "target_url": target_url,
        "stream": stream,
        "request_json": parse_json_payload(original_body),
        "forwarded_request_json": parse_json_payload(body),
    }
    
    if stream:
        return await handle_stream_request(target_url, body, debug_context)
    else:
        return await handle_normal_request(target_url, body, debug_context)


async def stream_generator(target_url: str, body: bytes, debug_context: dict):
    """转发流式响应并在结束后落盘调试日志。"""
    status_code = None
    stream_error = None
    raw_stream = bytearray()
    async with httpx.AsyncClient(timeout=300.0) as client:
        req = client.build_request(
            method="POST",
            url=target_url,
            content=body,
            headers={
                "Authorization": f"Bearer {KIMI_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": KIMI_CLI_USER_AGENT,
            },
        )
        try:
            response = await client.send(req, stream=True)
            status_code = response.status_code
            if status_code != 200:
                content = await response.aread()
                raw_stream.extend(content)
                error_msg = {"error": {"message": f"Upstream error: {status_code}", "type": "upstream_error"}}
                yield f"data: {json.dumps(error_msg)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                await response.aclose()
                return
            try:
                async for chunk in response.aiter_bytes():
                    raw_stream.extend(chunk)
                    yield chunk
            finally:
                await response.aclose()
        except Exception as exc:
            stream_error = str(exc)
            yield f"data: {{\"error\": \"{stream_error}\"}}\n\n".encode()
        finally:
            response_json = (
                parse_stream_json_events(bytes(raw_stream))
                if status_code == 200
                else parse_json_payload(bytes(raw_stream))
            )
            debug_record = {
                **debug_context,
                "status_code": status_code,
                "response_json": response_json,
            }
            if stream_error:
                debug_record["upstream_exception"] = stream_error
            await append_debug_jsonl(debug_record)


async def handle_stream_request(target_url: str, body: bytes, debug_context: dict):
    """处理流式请求"""
    
    return StreamingResponse(
        stream_generator(target_url, body, debug_context),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


async def handle_normal_request(target_url: str, body: bytes, debug_context: dict):
    """处理非流式请求"""
    # 使用 httpx 低级别 API 来确保 headers 正确发送
    async with httpx.AsyncClient(timeout=300.0) as client:
        req = client.build_request(
            method=debug_context["method"],
            url=target_url,
            content=body,
            headers={
                "Authorization": f"Bearer {KIMI_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": KIMI_CLI_USER_AGENT,
            },
        )
        try:
            response = await client.send(req)
        except Exception as exc:
            await append_debug_jsonl({**debug_context, "upstream_exception": str(exc)})
            raise

        try:
            content = response.json()
        except json.JSONDecodeError as exc:
            content = {"_parse_error": str(exc), "_raw": response.text}
        
        if target_url == f"{KIMI_BASE_URL}/models" or target_url == f"{KIMI_BASE_URL}/models/":
            if 'data' in content:
                jo = None
                for m in content['data']:
                    i = m.get('id', "")
                    if i =='kimi-for-coding':
                        jo = m
                        break

                if jo is not None:
                    def append_model(eff):
                        code = eff.lower()
                        njo = copy.deepcopy(jo)
                        njo['id'] = f"kimi-for-coding:{code}"
                        njo['display_name'] = njo['display_name'] + f" ({eff} Thinking)"
                        content['data'].append(njo)

                    append_model("High")
                    append_model("Medium")
                    append_model("Low")
                    append_model("Off")
                    content['first_id'] = content['data'][0]['id']
                    content['last_id'] = content['data'][-1]['id']

        await append_debug_jsonl({
            **debug_context,
            "status_code": response.status_code,
            "response_json": content,
        })
        return JSONResponse(
            content=content,
            status_code=response.status_code,
        )


@app.get("/")
async def root():
    """健康检查与配置信息"""
    return {
        "status": "running",
        "service": "Kimi Coding Plan Gateway",
        "kimi_base_url": KIMI_BASE_URL,
        "gateway_url": f"http://{GATEWAY_HOST}:{GATEWAY_PORT}",
        "user_agent": KIMI_CLI_USER_AGENT,
        "debug_mode": DEBUG_MODE,
        "debug_jsonl_file": DEBUG_JSONL_FILE,
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

def get_local_ip():
    """获取本机局域网 IP"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def start_gateway():
    """启动网关服务"""
    print(f"[INFO] 启动 Kimi Coding Plan Gateway...")
    print(f"[INFO] 监听地址: http://{GATEWAY_HOST}:{GATEWAY_PORT}")
    
    if GATEWAY_HOST == "0.0.0.0":
        local_ip = get_local_ip()
        print(f"[INFO] 局域网访问地址: http://{local_ip}:{GATEWAY_PORT}")
        print(f"[WARN] 注意：允许外部访问可能导致 API Key 被局域网内其他人使用")
        
    uvicorn.run(app, host=GATEWAY_HOST, port=GATEWAY_PORT)

if __name__ == "__main__":
    start_gateway()
