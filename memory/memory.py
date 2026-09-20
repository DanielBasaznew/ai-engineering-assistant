import datetime
import sqlite3
from rich.console import Console
from rich.table import Table

console = Console()


class PersistentMemory:

  def __init__(self, db_path: str = "assistant_memory.db"):
    self.db_path = db_path
    self._init_db()

  def _get_connection(self):
    return sqlite3.connect(self.db_path)

  def _init_db(self):
    """Creates tables for facts (Semantic Memory) and conversation_log (Episodic Memory),

    and ensures metadata columns (confidence, source) exist.
    """
    with self._get_connection() as conn:
      cursor = conn.cursor()

      # 1. Semantic Memory: Extracted facts table
      cursor.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    confidence TEXT DEFAULT 'high',
                    source TEXT DEFAULT 'user_statement',
                    updated_at TEXT NOT NULL
                )
            """)

      # Ensure schema migration for existing databases created before these columns existed
      for col, col_type, default_val in [
          ("confidence", "TEXT", "'high'"),
          ("source", "TEXT", "'user_statement'"),
      ]:
        try:
          cursor.execute(
              f"ALTER TABLE facts ADD COLUMN {col} {col_type} DEFAULT"
              f" {default_val}"
          )
        except sqlite3.OperationalError:
          pass  # Column already exists

      # 2. Episodic Memory: Raw conversation history log
      cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
      conn.commit()

  def store_fact(
      self,
      key: str,
      value: str,
      category: str = "general",
      confidence: str = "high",
      source: str = "user_statement",
  ) -> None:
    """Stores or updates a fact key-value pair using SQLite UPSERT."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
          """
                INSERT INTO facts (key, value, category, confidence, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    category = excluded.category,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    updated_at = excluded.updated_at
            """,
          (key, value, category, confidence, source, now),
      )
      conn.commit()

  def get_fact(self, key: str) -> dict | None:
    """Retrieves a single fact by key."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
          "SELECT key, value, category, confidence, source, updated_at FROM"
          " facts WHERE key = ?",
          (key,),
      )
      row = cursor.fetchone()
      if row:
        return {
            "key": row[0],
            "value": row[1],
            "category": row[2],
            "confidence": row[3],
            "source": row[4],
            "updated_at": row[5],
        }
      return None

  def get_all_facts(self) -> list[dict]:
    """Retrieves all stored facts."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
          "SELECT key, value, category, confidence, source, updated_at FROM"
          " facts ORDER BY updated_at DESC"
      )
      rows = cursor.fetchall()
      return [{
          "key": r[0],
          "value": r[1],
          "category": r[2],
          "confidence": r[3],
          "source": r[4],
          "updated_at": r[5],
      } for r in rows]

  def delete_fact(self, key: str) -> bool:
    """Deletes a stored fact by key."""
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("DELETE FROM facts WHERE key = ?", (key,))
      conn.commit()
      return cursor.rowcount > 0

  def log_conversation(self, session_id: str, role: str, content: str) -> None:
    """Logs a raw turn in the conversation history."""
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute(
          """
                INSERT INTO conversation_log (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
            """,
          (session_id, role, content, now),
      )
      conn.commit()

  def format_for_prompt(self) -> str:
    """Formats all stored facts into a string block for system prompt injection."""
    facts = self.get_all_facts()
    if not facts:
      return "No stored personal facts."

    formatted_lines = ["[User Memory & Personal Context]"]
    for f in facts:
      formatted_lines.append(
          f"- {f['key']}: {f['value']} (category: {f['category']})"
      )

    return "\n".join(formatted_lines)

  def memory_report(self) -> None:
    """Renders a Rich table showing full memory state, provenance, and turn counts."""
    facts = self.get_all_facts()

    with self._get_connection() as conn:
      cursor = conn.cursor()
      cursor.execute("SELECT COUNT(*) FROM conversation_log")
      conv_count = cursor.fetchone()[0]

    table = Table(
        title="🧠 Persistent Memory Store & Data Provenance", show_header=True
    )
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    table.add_column("Category", style="magenta")
    table.add_column("Confidence", style="yellow")
    table.add_column("Source", style="dim")
    table.add_column("Updated", style="dim")

    for f in facts:
      updated_date = f["updated_at"][:10] if f["updated_at"] else "N/A"
      table.add_row(
          f["key"],
          f["value"],
          f["category"],
          f["confidence"],
          f["source"],
          updated_date,
      )

    console.print(table)
    console.print(
        f"[dim]Total conversation turns logged (Episodic Memory):"
        f" {conv_count}[/dim]\n"
    )


if __name__ == "__main__":
  mem = PersistentMemory("test_memory.db")
  mem.store_fact(
      "favorite_language",
      "Python",
      category="preferences",
      confidence="high",
      source="user_statement",
  )
  mem.store_fact(
      "location",
      "Addis Ababa",
      category="background",
      confidence="high",
      source="user_statement",
  )
  mem.log_conversation("sess_1", "user", "Hi assistant!")
  mem.log_conversation("sess_1", "assistant", "Hello!")

  mem.memory_report()
