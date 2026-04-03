"""visualize_nav.py — 将 nav_graph.json 渲染为一张导航图 PNG。

用法：
    python tools/visualize_nav.py                          # 自动找 data/memory/nav_graph.json
    python tools/visualize_nav.py --graph path/to/nav_graph.json
    python tools/visualize_nav.py --graph path/to/nav_graph.json --out nav.png
    python tools/visualize_nav.py --graph path/to/nav_graph.json --screenshots path/to/screenshots/

节点颜色代表访问次数（越深越多），节点可嵌入对应页面截图缩略图。
边标签为 action:element_label（或 element_id）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_graph(graph_path: str):
    import networkx as nx
    data = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    g = nx.DiGraph()
    for node_id, attrs in data.get("nodes", {}).items():
        g.add_node(node_id, **attrs)
    for edge in data.get("edges", []):
        u, v = edge["from"], edge["to"]
        attrs = {k: val for k, val in edge.items() if k not in ("from", "to")}
        g.add_edge(u, v, **attrs)
    return g, data.get("visited_map", {})


def _short(h: str, n: int = 8) -> str:
    return h[:n]


def _set_cjk_font() -> None:
    """尝试设置支持 CJK 的字体，避免中文乱码。"""
    import matplotlib.pyplot as plt
    candidates = ["Microsoft YaHei", "SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Arial Unicode MS"]
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def render(graph_path: str, screenshots_dir: str | None, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _set_cjk_font()
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import networkx as nx
    import numpy as np
    from PIL import Image
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    g, visited_map = _load_graph(graph_path)

    if g.number_of_nodes() == 0:
        print("导航图为空，无节点可渲染。")
        return

    n = g.number_of_nodes()

    # ── 布局 ────────────────────────────────────────────────────
    try:
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    except Exception:
        try:
            pos = nx.planar_layout(g)
        except Exception:
            pos = nx.spring_layout(g, seed=42, k=5.0 / max(n ** 0.5, 1))

    visit_counts = [g.nodes[nd].get("visit_count", 0) for nd in g.nodes]
    max_vc = max(visit_counts) if max(visit_counts) > 0 else 1
    norm = mcolors.Normalize(vmin=0, vmax=max_vc)
    cmap = plt.cm.YlOrRd

    node_colors = [cmap(norm(g.nodes[nd].get("visit_count", 0))) for nd in g.nodes]
    node_labels = {nd: _short(nd) for nd in g.nodes}

    edge_labels = {}
    for u, v, data in g.edges(data=True):
        lbl = data.get("label") or ""
        action = data.get("action", "tap")
        edge_labels[(u, v)] = f"{action}:{lbl}" if lbl else action

    # ── 画布：按节点数自适应大小 ─────────────────────────────────
    fig_w = max(20, n * 2.8)
    fig_h = max(14, n * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_title(
        f"Navigation Graph  —  {n} pages / {g.number_of_edges()} transitions",
        fontsize=16, pad=20,
    )
    ax.axis("off")

    # ── 节点 ────────────────────────────────────────────────────
    nx.draw_networkx_nodes(
        g, pos, ax=ax,
        node_color=node_colors,
        node_size=3600,
        alpha=0.90,
        linewidths=1.5,
        edgecolors="#333333",
    )
    nx.draw_networkx_labels(
        g, pos, labels=node_labels, ax=ax,
        font_size=9, font_color="#111111", font_weight="bold",
    )

    # ── 边 ──────────────────────────────────────────────────────
    nx.draw_networkx_edges(
        g, pos, ax=ax,
        edge_color="#666666",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=22,
        width=1.8,
        connectionstyle="arc3,rad=0.12",
        min_source_margin=28,
        min_target_margin=28,
    )
    nx.draw_networkx_edge_labels(
        g, pos, edge_labels=edge_labels, ax=ax,
        font_size=8, font_color="#1a3a99",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", alpha=0.85),
        rotate=False,
    )

    # ── 缩略图（OffsetImage，锚定到数据坐标）────────────────────
    if screenshots_dir:
        ss_dir = Path(screenshots_dir)
        THUMB_PX = 96
        for node, (x, y) in pos.items():
            img_path = ss_dir / f"{node}.png"
            if not img_path.exists():
                continue
            try:
                thumb = Image.open(img_path).convert("RGB")
                thumb.thumbnail((THUMB_PX, THUMB_PX))
                oi = OffsetImage(np.array(thumb), zoom=1.0)
                oi.image.axes = ax
                ab = AnnotationBbox(
                    oi, (x, y),
                    xybox=(0, 52),
                    boxcoords="offset points",
                    frameon=True,
                    bboxprops=dict(edgecolor="#aaaaaa", linewidth=0.8, boxstyle="round,pad=0.1"),
                    pad=0.1,
                )
                ax.add_artist(ab)
            except Exception:
                pass

    # ── colorbar ────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.01)
    cbar.set_label("visit count", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # ── legend ──────────────────────────────────────────────────
    total_visited = sum(len(v) for v in visited_map.values())
    patch = mpatches.Patch(color="none", label=f"已访问元素总计: {total_visited}")
    ax.legend(handles=[patch], loc="upper left", fontsize=10, framealpha=0.8)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"导航图已保存: {out_path}")


def _find_default_graph() -> str | None:
    fixed = [
        Path("data/memory/nav_graph.json"),
        Path("memory/nav_graph.json"),
        Path("../data/memory/nav_graph.json"),
    ]
    for c in fixed:
        if c.exists():
            return str(c)
    # 递归搜索 data/ 下的第一个 nav_graph.json
    for c in sorted(Path("data").rglob("nav_graph.json")):
        return str(c)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="可视化 NavigationGraph 导航图")
    parser.add_argument("--graph", default="", help="nav_graph.json 路径")
    parser.add_argument("--screenshots", default="", help="页面截图目录（可选，嵌入缩略图）")
    parser.add_argument("--out", default="data/nav_graph.png", help="输出 PNG 路径（默认: data/nav_graph.png）")
    args = parser.parse_args()

    graph_path = args.graph or _find_default_graph()
    if not graph_path or not Path(graph_path).exists():
        print("找不到 nav_graph.json，请用 --graph 指定路径。")
        return 1

    screenshots_dir = args.screenshots or None
    if not screenshots_dir:
        auto_ss = Path(graph_path).parent / "screenshots"
        if auto_ss.is_dir():
            screenshots_dir = str(auto_ss)

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("缺少 matplotlib，请先安装：pip install matplotlib")
        return 1

    render(graph_path, screenshots_dir, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
