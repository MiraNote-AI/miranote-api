"""sqlite-vec wrapper for vector storage + similarity search.

Each item has a string `id`, a JSON `payload` (the full metadata),
and a `dim`-float vector. sqlite-vec is loaded as a SQLite extension;
the data lives in a single .db file (or `:memory:` for tests).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import apsw
import numpy as np
import sqlite_vec


class Store:
    def __init__(self, db_path: Union[str, Path], dim: int = 1024):
        self._dim = dim
        path_str = str(db_path) if isinstance(db_path, Path) else db_path
        self._conn = apsw.Connection(path_str)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._setup()

    def _setup(self) -> None:
        cur = self._conn.cursor()
        # Main metadata table -- one row per item.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        # Vec virtual table -- one row per item, keyed by rowid linking to items.
        cur.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vecs USING vec0(
                embedding float[{self._dim}]
            )
            """
        )

    def count(self) -> int:
        cur = self._conn.cursor()
        result = cur.execute("SELECT COUNT(*) FROM items").fetchall()
        return int(result[0][0]) if result else 0

    def insert(self, id_: str, payload: Dict[str, Any], embedding: np.ndarray) -> None:
        cur = self._conn.cursor()
        # If id_ already exists, delete the old row first (REPLACE on id).
        existing = cur.execute("SELECT rowid FROM items WHERE id = ?", (id_,)).fetchall()
        if existing:
            old_rowid = existing[0][0]
            cur.execute("DELETE FROM items WHERE rowid = ?", (old_rowid,))
            cur.execute("DELETE FROM vecs WHERE rowid = ?", (old_rowid,))

        cur.execute(
            "INSERT INTO items (id, payload) VALUES (?, ?)",
            (id_, json.dumps(payload, ensure_ascii=False)),
        )
        # Get the rowid of the inserted row
        rowid = cur.execute("SELECT last_insert_rowid()").fetchall()[0][0]
        vec_blob = embedding.astype(np.float32).tobytes()
        cur.execute(
            "INSERT INTO vecs (rowid, embedding) VALUES (?, ?)",
            (rowid, vec_blob),
        )

    def search(self, query: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        vec_blob = query.astype(np.float32).tobytes()
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT items.id, items.payload, vecs.distance
            FROM vecs
            JOIN items ON items.rowid = vecs.rowid
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (vec_blob, k),
        ).fetchall()
        return [
            {"id": row[0], "payload": json.loads(row[1]), "distance": float(row[2])}
            for row in rows
        ]
