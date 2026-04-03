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

    _HASH_MERGE_THRESHOLD = 6  # 汉明距离 ≤ 此值认为是同一逻辑页面（Unity 动态帧内容）

    def normalize_hash(self, page_hash: str) -> str:
        """将 page_hash 归一化到已知最近节点，解决 Unity 动态内容导致的 hash 漂移。

        若与已知节点汉明距离 ≤ _HASH_MERGE_THRESHOLD，复用已知节点 key；
        否则返回原始 hash（作为新节点）。
        """
        try:
            import imagehash
            new_h = imagehash.hex_to_hash(page_hash)
        except Exception:
            return page_hash

        best_key: Optional[str] = None
        best_dist = self._HASH_MERGE_THRESHOLD + 1
        for existing_key in self._g.nodes:
            try:
                dist = new_h - imagehash.hex_to_hash(existing_key)
                if dist < best_dist:
                    best_dist = dist
                    best_key = existing_key
            except Exception:
                continue

        if best_key is not None and best_dist <= self._HASH_MERGE_THRESHOLD:
            if best_key != page_hash:
                logger.debug("hash 归一: %s → %s (dist=%d)", page_hash[:8], best_key[:8], best_dist)
            return best_key
        return page_hash

    def register_page(
        self,
        page_hash: str,
        elements: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """注册或更新页面节点。

        Args:
            page_hash: 页面感知哈希。
            elements:  VisionProvider.detect() 返回的元素列表
                       每项格式: {id, bbox, label, type, interactable}。
        """
        page_hash = self.normalize_hash(page_hash)
        if not self._g.has_node(page_hash):
            self._g.add_node(page_hash, visit_count=0, elements=[])
            self._visited_map[page_hash] = set()
            logger.debug("注册新页面节点: %s", page_hash[:8])
        node = self._g.nodes[page_hash]
        node.setdefault("visit_count", 0)
        node.setdefault("elements", [])
        node["visit_count"] += 1
        if elements:
            self._g.nodes[page_hash]["elements"] = elements

    def reset_visits(self) -> None:
        """清空已访问记录，保留图拓扑（每次 run 开始前调用）。"""
        self._visited_map.clear()

    def mark_visited(self, page_hash: str, element_id: int) -> None:
        page_hash = self.normalize_hash(page_hash)
        if page_hash not in self._visited_map:
            self._visited_map[page_hash] = set()
        self._visited_map[page_hash].add(element_id)

    def get_visited_ids(self, page_hash: str) -> List[int]:
        return sorted(self._visited_map.get(self.normalize_hash(page_hash), set()))

    def get_elements(self, page_hash: str) -> List[Dict[str, Any]]:
        """返回页面的完整元素列表（含 id/bbox/label/type）。"""
        page_hash = self.normalize_hash(page_hash)
        return list(self._g.nodes[page_hash].get("elements", [])) if self._g.has_node(page_hash) else []

    def get_element_labels(self, page_hash: str) -> List[str]:
        """返回页面元素的 label 列表（向后兼容）。"""
        return [e.get("label", "") for e in self.get_elements(page_hash) if e.get("label")]

    def screenshot_path(self, page_hash: str, memory_dir: str) -> str:
        """按约定推导页面截图路径: memory/screenshots/{hash}.png。"""
        return str(Path(memory_dir) / "screenshots" / f"{page_hash}.png")

    def get_unvisited_ids(self, page_hash: str, all_element_ids: List[int]) -> List[int]:
        page_hash = self.normalize_hash(page_hash)
        visited = self._visited_map.get(page_hash, set())
        return [eid for eid in all_element_ids if eid not in visited]

    def is_fully_explored(self, page_hash: str, all_element_ids: List[int]) -> bool:
        return len(self.get_unvisited_ids(page_hash, all_element_ids)) == 0

    def get_outbound_nav_hints(self, page_hash: str) -> List[int]:
        """返回当前页中指向"有未探索元素子页"的元素 ID 列表（用于 DFS 导航）。

        当当前页所有元素均已访问时，SequentialDecider 通过这些 ID 重新点击
        对应按钮导航到还有未探索内容的子页，实现跨页持续探索。
        """
        page_hash = self.normalize_hash(page_hash)
        hints: List[int] = []
        if not self._g.has_node(page_hash):
            return hints
        for _, child_hash, edge_data in self._g.edges(page_hash, data=True):
            child_hash = self.normalize_hash(child_hash)
            if child_hash == page_hash:
                continue  # 自环，跳过
            child_elems = self._g.nodes[child_hash].get("elements", [])
            child_all_ids = [e["id"] for e in child_elems]
            child_unvisited = self.get_unvisited_ids(child_hash, child_all_ids)
            if child_unvisited:
                elem_id = edge_data.get("element_id")
                if elem_id is not None and elem_id not in hints:
                    hints.append(elem_id)
        return hints

    def add_transition(
        self,
        from_hash: str,
        to_hash: str,
        action: str,
        element_id: int,
        element_label: str = "",
    ) -> None:
        from_hash = self.normalize_hash(from_hash)
        to_hash   = self.normalize_hash(to_hash)
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
        }

    def full_stats(self) -> Dict[str, Any]:
        """含环检测的完整统计（较慢，仅在最终报告时调用）。"""
        s = self.stats()
        s["cycles"] = len(self.find_simple_cycles())
        return s

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
        return f"NavigationGraph(pages={self._g.number_of_nodes()}, transitions={self._g.number_of_edges()}, visited_elements={sum(len(v) for v in self._visited_map.values())})"
