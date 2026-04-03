#!/usr/bin/env bash
# AutoTestAgent — Linux / macOS 一键安装脚本
set -euo pipefail

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║       AutoTestAgent  Setup (Linux/Mac)       ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

# ── [1/3] 检查 Python ──────────────────────────────────────────
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"; break
    fi
done
[ -z "$PYTHON" ] && echo "[ERROR] 未找到 Python，请安装 Python 3.10+" && exit 1
echo "[1/3] $($PYTHON --version) OK"

# ── [2/3] 安装依赖 ─────────────────────────────────────────────
echo "[2/3] 安装依赖（可能需要几分钟）..."
$PYTHON -m pip install -r requirements.txt
echo "[2/3] 依赖安装完成"

# ── [3/3] 配置说明 ────────────────────────────────────────────
echo "[3/3] 默认使用 .env.example 配置；如需自定义请创建 .env（优先级更高）"

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║  安装完成！编辑 .env 后执行：                ║"
echo "  ║    python main.py --task '你的测试任务'       ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""
