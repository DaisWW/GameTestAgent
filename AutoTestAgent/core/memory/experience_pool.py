from __future__ import annotations

"""第三层：长期经验池 (Experience Pool)

SQLite 持久化，跨 session / 跨版本保留测试经验。
三张表：successful_paths / bug_observations / ui_knowledge
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS successful_paths (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task        TEXT    NOT NULL,
    steps_json  TEXT    NOT NULL,
    step_count  INTEGER NOT NULL,
    page_hashes TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS bug_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    page_hash       TEXT    NOT NULL,
    description     TEXT    NOT NULL,
    screenshot_path TEXT    DEFAULT '',
    tags            TEXT    DEFAULT '',
    created_at      TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS ui_knowledge (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT    NOT NULL,
    description TEXT    NOT NULL,
    page_hash   TEXT    DEFAULT '',
    created_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bug_hash ON bug_observations(page_hash);
CREATE INDEX IF NOT EXISTS idx_ui_label ON ui_knowledge(label);
"""


class ExperiencePool:
    """长期经验池（SQLite）。

    Args:
        db_path: SQLite 文件路径；传入 ":memory:" 可用于单元测试。
    """

    def __init__(self, db_path: str = "data/experience.db") -> None:
        self._db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()
        logger.info("ExperiencePool 已连接: %s", db_path)

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    # ── 成功路径 ──────────────────────────────────────────────────────

    def save_successful_path(self, task: str, steps: List[Dict[str, Any]], page_hashes: Optional[List[str]] = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO successful_paths (task, steps_json, step_count, page_hashes, created_at) VALUES (?,?,?,?,?)",
            (task, json.dumps(steps, ensure_ascii=False), len(steps),
             json.dumps(page_hashes or [], ensure_ascii=False), self._now()),
        )
        self._conn.commit()
        logger.info("已保存成功路径 id=%d task=%r steps=%d", cur.lastrowid, task[:40], len(steps))
        return cur.lastrowid

    def query_similar_paths(self, task: str, limit: int = 3) -> List[Dict[str, Any]]:
        keywords = task.split()
        if not keywords:
            return []
        conditions = " OR ".join(["task LIKE ?" for _ in keywords])
        rows = self._conn.execute(
            f"SELECT * FROM successful_paths WHERE {conditions} ORDER BY id DESC LIMIT ?",
            [f"%{kw}%" for kw in keywords] + [limit],
        ).fetchall()
        return [{"id": r["id"], "task": r["task"], "steps": json.loads(r["steps_json"]),
                 "step_count": r["step_count"], "created_at": r["created_at"]} for r in rows]

    # ── Bug 快照 ──────────────────────────────────────────────────────

    def save_bug(self, page_hash: str, description: str, screenshot_path: str = "", tags: Optional[List[str]] = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO bug_observations (page_hash, description, screenshot_path, tags, created_at) VALUES (?,?,?,?,?)",
            (page_hash, description, screenshot_path, json.dumps(tags or [], ensure_ascii=False), self._now()),
        )
        self._conn.commit()
        logger.info("已保存 Bug 快照 id=%d hash=%s", cur.lastrowid, page_hash[:8])
        return cur.lastrowid

    def is_known_bug_page(self, page_hash: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT description FROM bug_observations WHERE page_hash = ? LIMIT 1", (page_hash,)
        ).fetchone()
        return row["description"] if row else None

    def get_all_bugs(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._conn.execute(
            "SELECT page_hash, description, screenshot_path, tags, created_at FROM bug_observations ORDER BY id DESC"
        ).fetchall()]

    # ── UI 知识库 ─────────────────────────────────────────────────────

    def save_ui_knowledge(self, label: str, description: str, page_hash: str = "") -> int:
        cur = self._conn.execute(
            "INSERT INTO ui_knowledge (label, description, page_hash, created_at) VALUES (?,?,?,?)",
            (label, description, page_hash, self._now()),
        )
        self._conn.commit()
        logger.info("已保存 UI 知识: %r → %r", label, description[:40])
        return cur.lastrowid

    def query_ui_knowledge(self, label: str, limit: int = 3) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._conn.execute(
            "SELECT label, description, page_hash FROM ui_knowledge WHERE label LIKE ? ORDER BY id DESC LIMIT ?",
            (f"%{label}%", limit),
        ).fetchall()]

    def get_relevant_experience(self, task: str, page_hash: str, element_labels: Optional[List[str]] = None, limit: int = 5) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []

        for path in self.query_similar_paths(task, limit=min(2, limit)):
            result.append({"type": "path", "data": path})

        bug = self.is_known_bug_page(page_hash)
        if bug and len(result) < limit:
            result.append({"type": "bug", "data": {"page_hash": page_hash, "description": bug}})

        if element_labels:
            for label in element_labels:
                if len(result) >= limit:
                    break
                for k in self.query_ui_knowledge(label, limit=1):
                    result.append({"type": "knowledge", "data": k})

        return result[:limit]

    # ── 统计 ─────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        tables = ["successful_paths", "bug_observations", "ui_knowledge"]
        return {t: self._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}

    def close(self) -> None:
        self._conn.close()

    def __repr__(self) -> str:
        s = self.stats()
        return f"ExperiencePool(db={self._db_path!r}, paths={s['successful_paths']}, bugs={s['bug_observations']}, knowledge={s['ui_knowledge']})"
