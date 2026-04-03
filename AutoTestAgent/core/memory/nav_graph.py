from __future__ import annotations

"""第二层：中层路径导航图 (Navigation Graph)

以页面感知哈希（ImageHash）为节点，以 Action 为有向边，
构建"已探索地图"，防止死循环并优先引导 Agent 点击未探索元素。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx

logger = logging.getLogger(__name__)


class NavigationGraph:
    """中层路径导航图（NetworkX DiGraph，可持久化到 JSON）。

    Args:
        graph_path: JSON 持久化路径，不为 None 时自动 load/save。
    """

    def __init__(self, graph_path: Optional[str] = None) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        self._visited_map: Dict[str, Set[int]] = {}
        self._graph_path = graph_path

        if graph_path and Path(graph_path).exists():
            self.load_json(graph_path)
            logger.info("导航图已从 %s 恢复，共 %d 个页面节点", graph_path, self._g.number_of_nodes())

    def register_page(
        self,
        page_hash: str,
        elements: Optional[List[Dict[str, Any]]] = None,
        screenshot_path: str = "",
    ) -> None:
        """注册或更新页面节点。

        Args:
            page_hash:       页面感知哈希。
            elements:        VisionProvider.detect() 返回的元素列表
                             每项格式: {id, bbox, label, type}。
            screenshot_path: 该页面截图的本地路径（首次捕获时填入）。
        """
        if not self._g.has_node(page_hash):
            self._g.add_node(page_hash, visit_count=0, elements=[], screenshot_path="")
            self._visited_map[page_hash] = set()
            logger.debug("注册新页面节点: %s", page_hash[:8])
        self._g.nodes[page_hash]["visit_count"] += 1
        if elements:
            self._g.nodes[page_hash]["elements"] = elements
        if screenshot_path and not self._g.nodes[page_hash].get("screenshot_path"):
            self._g.nodes[page_hash]["screenshot_path"] = screenshot_path

    def mark_visited(self, page_hash: str, element_id: int) -> None:
        if page_hash not in self._visited_map:
            self._visited_map[page_hash] = set()
        self._visited_map[page_hash].add(element_id)

    def get_visited_ids(self, page_hash: str) -> List[int]:
        return sorted(self._visited_map.get(page_hash, set()))

    def get_elements(self, page_hash: str) -> List[Dict[str, Any]]:
        """返回页面的完整元素列表（含 id/bbox/label/type）。"""
        return list(self._g.nodes[page_hash].get("elements", [])) if self._g.has_node(page_hash) else []

    def get_element_labels(self, page_hash: str) -> List[str]:
        """返回页面元素的 label 列表（向后兼容）。"""
        return [e.get("label", "") for e in self.get_elements(page_hash) if e.get("label")]

    def get_screenshot_path(self, page_hash: str) -> str:
        """返回页面截图路径，未存储时返回空串。"""
        return self._g.nodes[page_hash].get("screenshot_path", "") if self._g.has_node(page_hash) else ""

    def get_unvisited_ids(self, page_hash: str, all_element_ids: List[int]) -> List[int]:
        visited = self._visited_map.get(page_hash, set())
        return [eid for eid in all_element_ids if eid not in visited]

    def is_fully_explored(self, page_hash: str, all_element_ids: List[int]) -> bool:
        return len(self.get_unvisited_ids(page_hash, all_element_ids)) == 0

    def add_transition(
        self,
        from_hash: str,
        to_hash: str,
        action: str,
        element_id: int,
        element_label: str = "",
    ) -> None:
        self._g.add_edge(from_hash, to_hash, action=action, element_id=element_id, label=element_label)
        logger.debug("导航边: %s -[%s:%s]-> %s", from_hash[:8], action, element_label or element_id, to_hash[:8])

    def get_page_visit_count(self, page_hash: str) -> int:
        return self._g.nodes[page_hash].get("visit_count", 0) if self._g.has_node(page_hash) else 0

    def get_navigation_path(self, from_hash: str, to_hash: str) -> Optional[List[str]]:
        try:
            return nx.shortest_path(self._g, from_hash, to_hash)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_breadcrumb(self, hash_sequence: List[str]) -> str:
        parts, seen = [], set()
        for h in hash_sequence:
            if h in seen:
                continue
            seen.add(h)
            elements = self._g.nodes[h].get("elements", []) if self._g.has_node(h) else []
            label = elements[0].get("label", "") if elements else ""
            parts.append(label or h[:8])
        return " > ".join(parts) if parts else "未知路径"

    def find_simple_cycles(self) -> List[List[str]]:
        try:
            return list(nx.simple_cycles(self._g))
        except Exception:
            return []

    def is_in_aba_loop(self, recent_hashes: List[str]) -> bool:
        """检测最近哈希序列是否呈 A-B-A-B 交替循环。"""
        if len(recent_hashes) < 4:
            return False
        tail = recent_hashes[-4:]
        return tail[0] == tail[2] and tail[1] == tail[3] and tail[0] != tail[1]

    def stats(self) -> Dict[str, Any]:
        return {
            "pages":            self._g.number_of_nodes(),
            "transitions":      self._g.number_of_edges(),
            "visited_elements": sum(len(v) for v in self._visited_map.values()),
            "cycles":           len(self.find_simple_cycles()),
        }

    def save_json(self, path: Optional[str] = None) -> None:
        target = path or self._graph_path
        if not target:
            return
        data = {
            "nodes": {n: self._g.nodes[n] for n in self._g.nodes},
            "edges": [{"from": u, "to": v, **self._g.edges[u, v]} for u, v in self._g.edges],
            "visited_map": {k: sorted(v) for k, v in self._visited_map.items()},
        }
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("导航图已保存: %s (%d nodes)", target, self._g.number_of_nodes())

    def load_json(self, path: str) -> None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self._g = nx.DiGraph()
        for node_id, attrs in raw.get("nodes", {}).items():
            self._g.add_node(node_id, **attrs)
        for edge in raw.get("edges", []):
            u, v = edge["from"], edge["to"]
            attrs = {k: val for k, val in edge.items() if k not in ("from", "to")}
            self._g.add_edge(u, v, **attrs)
        self._visited_map = {k: set(v) for k, v in raw.get("visited_map", {}).items()}

    def __repr__(self) -> str:
        s = self.stats()
        return f"NavigationGraph(pages={s['pages']}, transitions={s['transitions']}, visited_elements={s['visited_elements']})"
