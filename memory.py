"""Per-user sqlite facts for Wise Mentor.

Durable first-person claims, keyed by Discord user id. Admin delete is a
CLI on the host (`python -m memory forget <id>`), not an in-Discord command.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(os.environ.get("WISE_MENTOR_MEMORY", "data/memory.sqlite"))

_MENTION = re.compile(r"<@!?\d+>\s*:?\s*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_DURABLE = re.compile(
    r"\b("
    r"my name is|call me|"
    r"i(?:'m| am| like| love| play| hate| collect| wrestle| live|"
    r" go to|'ve been| have been| want to be)"
    r")\b",
    re.IGNORECASE,
)


def extract_facts(text: str, *, limit: int = 3, max_len: int = 200) -> list[str]:
    if not text or not text.strip():
        return []
    cleaned = _MENTION.sub("", text).strip()
    if not cleaned:
        return []
    parts = [p.strip(" \t-—") for p in _SENTENCE.split(cleaned) if p.strip()]
    facts: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if not _DURABLE.search(part):
            continue
        fact = part[:max_len].strip()
        key = fact.casefold()
        if not fact or key in seen:
            continue
        seen.add(key)
        facts.append(fact)
        if len(facts) >= limit:
            break
    return facts


def format_memory_block(facts: list[str]) -> str:
    if not facts:
        return ""
    lines = "\n".join(f"- {fact}" for fact in facts)
    return (
        "Known about this person (use when relevant; do not list unless asked):\n"
        f"{lines}"
    )


class MemoryStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    fact TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS facts_user_key "
                "ON facts(user_id, fact_key)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    message_id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS transcripts_user ON transcripts(user_id)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def remember(self, user_id: int, fact: str) -> bool:
        fact = (fact or "").strip()
        if not fact:
            return False
        key = fact.casefold()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO facts (user_id, fact, fact_key) VALUES (?, ?, ?)",
                (user_id, fact, key),
            )
            return cur.rowcount > 0

    def recall(self, user_id: int, limit: int = 20) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT fact FROM facts WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [row[0] for row in rows]

    def save_transcript(self, message_id: int, user_id: int, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO transcripts (message_id, user_id, text)
                VALUES (?, ?, ?)
                ON CONFLICT(message_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    text = excluded.text
                """,
                (message_id, user_id, text[:20000]),
            )

    def get_transcript(self, message_id: int) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT text FROM transcripts WHERE message_id = ?",
                (message_id,),
            ).fetchone()
        return row[0] if row else None

    def forget_user(self, user_id: int) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM transcripts WHERE user_id = ?", (user_id,))
            return cur.rowcount


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wise Mentor per-user memory")
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB),
        help="sqlite path (or WISE_MENTOR_MEMORY)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    forget = sub.add_parser("forget", help="delete facts and voice transcripts for a Discord user id")
    forget.add_argument("user_id", type=int)
    rec = sub.add_parser("recall", help="print stored facts for a Discord user id")
    rec.add_argument("user_id", type=int)
    args = parser.parse_args(argv)
    store = MemoryStore(Path(args.db))
    if args.cmd == "forget":
        n = store.forget_user(args.user_id)
        print(f"deleted {n} fact(s) for {args.user_id}")
        return 0
    for fact in store.recall(args.user_id):
        print(fact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
