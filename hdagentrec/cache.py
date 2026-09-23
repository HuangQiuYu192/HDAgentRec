from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Optional, Union


class SQLiteCache:
    def __init__(self, path: Union[str, Path]):
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    @staticmethod
    def key(model: str, prompt_version: str, payload: str) -> str:
        return hashlib.sha256(f"{model}\0{prompt_version}\0{payload}".encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        row = self.connection.execute("SELECT value FROM llm_cache WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        self.connection.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?)", (key, value)); self.connection.commit()
