"""Tamper-evident decision log.

Every record stores the SHA-256 of the previous one, so deleting or editing
a past decision breaks the chain and `verify()` points at the first broken
link. A trust layer has to be auditable itself.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64


def _digest(record: dict) -> str:
    body = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.prev = GENESIS
        self.path.write_text("")

    def append(self, **fields) -> dict:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               **fields, "prev_hash": self.prev}
        rec["hash"] = _digest(rec)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        self.prev = rec["hash"]
        return rec


def verify(path: Path) -> tuple[bool, int]:
    """Returns (ok, index of first bad record or -1)."""
    prev = GENESIS
    for k, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        rec = json.loads(line)
        h = rec.pop("hash")
        if rec["prev_hash"] != prev or _digest(rec) != h:
            return False, k
        prev = h
    return True, -1
