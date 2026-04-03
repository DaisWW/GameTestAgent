from __future__ import annotations

"""测试报告生成器。

将本轮运行的步骤历史、Bug 列表、导航图统计写成 Markdown 文件，
保存到 runs/<run_id>/report.md。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

from core.types import BugSeverity, TestStatus
from core.models import BugRecord, NavStats

if TYPE_CHECKING:
    from core.memory.working_memory import MemoryStep

logger = logging.getLogger(__name__)


def write_report(
    run_dir: str,
    task: str,
    status: str,
    steps: int,
    history: "List[MemoryStep]",
    nav_stats: NavStats,
    new_bugs: List[BugRecord],
    reason: str,
    game_package: str = "",
    run_id: str = "",
) -> Path:
    """生成 Markdown 测试报告，保存到 run_dir/report.md。

    Returns:
        保存路径。
    """
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    status_emoji = "✅ PASS" if status == TestStatus.PASS else "❌ FAIL"
    lines = [
        f"# 测试报告 — {run_id or out.name}",
        "",
        "| 项目 | 内容 |",
        "|------|------|",
        f"| 任务 | {task} |",
        f"| 结果 | {status_emoji} |",
        f"| 步数 | {steps} |",
        f"| 时间 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
        f"| 游戏 | {game_package} |",
        "",
        "## 执行步骤",
        "",
        "| 步 | 动作 | 元素 | 思考 | 页面变化 |",
        "|---|------|------|------|---------|",
    ]

    for s in history:
        action    = s.action
        params    = s.params
        reasoning = s.reasoning[:50]
        label     = s.element_label or str(params)
        changed   = "✓" if s.success else "—"
        elem_id   = params.get("id", "")
        lines.append(
            f"| {s.step} | `{action}` | [{elem_id}] {label} | {reasoning} | {changed} |"
        )

    # ── 本轮发现的 Bug ──────────────────────────────────────────
    lines += ["", "## 本轮发现的问题", ""]
    if new_bugs:
        sev_map = {BugSeverity.CRITICAL: "🔴", BugSeverity.MAJOR: "🟠", BugSeverity.MINOR: "🟡"}
        lines.append("| 严重度 | 类型 | 描述 | 页面 |")
        lines.append("|--------|------|------|------|")
        for b in new_bugs:
            sev  = sev_map.get(b.severity, "⚪")
            cat  = b.category
            desc = b.description[:60]
            ph   = b.page_hash[:8]
            lines.append(f"| {sev} | {cat} | {desc} | `{ph}` |")
    else:
        lines.append("本轮未发现新问题。")

    # ── 导航图统计 ──────────────────────────────────────────────
    lines += [
        "",
        "## 导航图统计",
        "",
        f"- 已知页面: **{nav_stats.pages}**",
        f"- 页面转移: **{nav_stats.transitions}**",
        f"- 累计探索元素: **{nav_stats.visited_elements}**",
        "",
    ]

    if reason and status != TestStatus.PASS:
        lines += ["## 结束原因", "", f"> {reason}", ""]

    report_path = out / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("测试报告已保存: %s", report_path)
    return report_path
