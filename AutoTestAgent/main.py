from __future__ import annotations

"""AutoTestAgent 入口。

快速使用：
    python main.py --task "测试登录流程" --vision mock

环境配置：
    复制 .env.example 为 .env，填写 API Key 和游戏包名即可。
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from tools.core import configure_stdout
from core.types import TestStatus
from core.models import RunResult


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoTestAgent — 基于 LangGraph + VLM 的游戏自动化测试 Agent"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="测试任务描述，例如: '进入设置页并开启夜间模式'",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="",
        help="指定 .env 文件路径（默认：项目根目录 .env）",
    )
    parser.add_argument(
        "--vision",
        type=str,
        default="",
        help="覆盖配置中的 vision_type（omni_v2 | mock）",
    )
    parser.add_argument(
        "--llm-provider",
        dest="llm_provider",
        type=str,
        default="",
        help="覆盖配置中的 llm_provider（openai | anthropic | google）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="输出 DEBUG 级别日志",
    )
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = _parse_args()
    _setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # ── 加载配置 ─────────────────────────────────────────────────
    from config import load_config
    config = load_config(args.env or None)

    if args.vision:
        config.vision.vision_type = args.vision
    if args.llm_provider:
        config.llm.provider = args.llm_provider

    # ── 任务输入 ─────────────────────────────────────────────────
    task = args.task.strip()
    if not task:
        task = input("请输入测试任务描述 > ").strip()
    if not task:
        logger.error("任务描述不能为空")
        return 1

    # ── 构建 Agent ──────────────────────────────────────────────
    logger.info("正在初始化 Agent...")
    logger.info("  Vision Provider : %s", config.vision.vision_type)
    logger.info("  LLM   Provider  : %s / %s", config.llm.provider, config.llm.model_name)

    from core.agent import get_agent
    worker = get_agent(config)

    # ── 预热视觉模型 ─────────────────────────────────────────────
    logger.info("预热视觉 Provider...")
    worker.vision.warm_up()

    # ── 执行测试 ─────────────────────────────────────────────────
    logger.info("开始执行测试任务: %s", task)
    result = worker.run(task)

    # ── 输出结果 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  测试结果  : {result.status.upper()}")
    print(f"  执行步数  : {result.steps}")
    if result.status != TestStatus.PASS:
        print(f"  结束原因  : {result.reason or '未知'}")
    print("=" * 60)

    if result.history:
        print("\n  执行历史：")
        for s in result.history:
            print(f"    [{s.step:02d}] {s.action:12s} {s.params}  — {s.reasoning[:50]}")

    # ── 清理资源 ─────────────────────────────────────────────────
    worker.teardown()

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
