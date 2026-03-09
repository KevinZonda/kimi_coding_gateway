#!/bin/bash

# 确保进入脚本所在目录
cd "$(dirname "$0")"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 检查是否安装了依赖
if ! python3 -c "import fastapi, uvicorn, httpx, dotenv" 2>/dev/null; then
    echo "[INFO] 检测到未安装依赖，正在安装..."
    pip install -r requirements.txt
fi

# 设置 Python 路径
export PYTHONPATH=$PYTHONPATH:.

# 启动服务
echo "[INFO] 启动 Kimi Gateway..."
python3 app/main.py
