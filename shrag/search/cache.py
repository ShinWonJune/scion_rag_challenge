from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CacheStats:
    hit_count: int = 0
    miss_count: int = 0
    write_count: int = 0


class RequestCache:
    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        self.stats = CacheStats()
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(
        source: str,
        term: str,
        cur_page: int,
        row_count: int,
        fields: list[str],
    ) -> dict[str, Any]:
        return {
            "source": source.lower().strip(),
            "term": term.strip(),
            "cur_page": int(cur_page),
            "row_count": int(row_count),
            "fields": list(fields),
        }

    def get(self, key: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path_for_key(key)
        if not path.exists():
            self.stats.miss_count += 1
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["hit_count"] = int(payload.get("hit_count", 0)) + 1
            self._write_atomic(path, payload)
            self.stats.hit_count += 1
            return payload.get("value")
        except Exception:
            self.stats.miss_count += 1
            return None

    def put(self, key: dict[str, Any], value: Any) -> None:
        if not self.enabled:
            return
        path = self._path_for_key(key)
        payload = {
            "key": key,
            "value": value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hit_count": 0,
        }
        self._write_atomic(path, payload)
        self.stats.write_count += 1

    def _path_for_key(self, key: dict[str, Any]) -> Path:
        source = str(key.get("source", "unknown")).lower().strip() or "unknown"
        digest = hashlib.sha1(
            json.dumps(key, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return self.root / source / f"{digest}.json"

    def _write_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
