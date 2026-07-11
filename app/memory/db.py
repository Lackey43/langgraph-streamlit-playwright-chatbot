"""SQLite-backed memory store for per-user conversation history.
Automatically retains ONLY the last MAX_MEMORY_STATES (default 6) turns per user.
This keeps the database lean and the LLM context focused on recent relevant history.
"""
import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)

class MemoryStore:
    """Manages persistent per-user memory with automatic trimming to last N states."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL,           -- 'human' or 'ai'
                    content TEXT NOT NULL,
                    metadata TEXT,                -- JSON: tool_calls, file_refs, etc.
                    UNIQUE(user_id, turn_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_turn 
                ON conversation_turns (user_id, turn_id DESC)
            """)
            conn.commit()
        logger.info(f"Memory DB initialized at {self.db_path}")
    
    def get_next_turn_id(self, user_id: str) -> int:
        """Get the next turn number for a user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COALESCE(MAX(turn_id), 0) + 1 FROM conversation_turns WHERE user_id = ?",
                (user_id,)
            )
            return cursor.fetchone()[0]
    
    def add_turn(
        self, 
        user_id: str, 
        role: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a new conversation turn and automatically trim old ones.
        Returns the turn_id of the newly added turn.
        """
        turn_id = self.get_next_turn_id(user_id)
        timestamp = datetime.utcnow().isoformat()
        meta_json = json.dumps(metadata or {})
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO conversation_turns 
                   (user_id, turn_id, timestamp, role, content, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, turn_id, timestamp, role, content, meta_json)
            )
            conn.commit()
        
        # Trim to keep only last MAX_MEMORY_STATES
        self._trim_old_turns(user_id)
        
        logger.debug(f"Added turn {turn_id} for user {user_id}")
        return turn_id
    
    def _trim_old_turns(self, user_id: str):
        """Delete turns older than the last MAX_MEMORY_STATES for this user."""
        max_states = settings.max_memory_states
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                DELETE FROM conversation_turns 
                WHERE user_id = ? 
                  AND turn_id < (
                      SELECT MIN(turn_id) FROM (
                          SELECT turn_id FROM conversation_turns 
                          WHERE user_id = ? 
                          ORDER BY turn_id DESC 
                          LIMIT ?
                      )
                  )
                """,
                (user_id, user_id, max_states)
            )
            conn.commit()
    
    def get_last_n_turns(self, user_id: str, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve the last N turns (default = MAX_MEMORY_STATES) for context."""
        n = n or settings.max_memory_states
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT turn_id, timestamp, role, content, metadata 
                FROM conversation_turns 
                WHERE user_id = ? 
                ORDER BY turn_id DESC 
                LIMIT ?
                """,
                (user_id, n)
            )
            rows = cursor.fetchall()
        
        turns = []
        for row in reversed(rows):  # Return in chronological order
            turn_id, ts, role, content, meta_json = row
            turns.append({
                "turn_id": turn_id,
                "timestamp": ts,
                "role": role,
                "content": content,
                "metadata": json.loads(meta_json) if meta_json else {}
            })
        return turns
    
    def get_memory_context(self, user_id: str) -> str:
        """Format last turns into a readable context string for prompts or messages."""
        turns = self.get_last_n_turns(user_id)
        if not turns:
            return "No previous conversation history."
        
        context_lines = ["### Previous Conversation Context (last 6 turns max):"]
        for t in turns:
            prefix = "Human" if t["role"] == "human" else "Assistant"
            context_lines.append(f"**{prefix} (turn {t['turn_id']}): ** {t['content'][:500]}...")
            if t.get("metadata"):
                context_lines.append(f"  _Metadata: {json.dumps(t['metadata'], indent=2)[:200]}_")
        return "\n".join(context_lines)
    
    def clear_user_memory(self, user_id: str):
        """Clear all history for a user (useful for demo reset)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversation_turns WHERE user_id = ?", (user_id,))
            conn.commit()
        logger.info(f"Cleared memory for user {user_id}")
    
    def get_all_users(self) -> List[str]:
        """List all users who have conversation history."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT user_id FROM conversation_turns")
            return [row[0] for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Return simple stats about the memory DB."""
        with sqlite3.connect(self.db_path) as conn:
            total_turns = conn.execute("SELECT COUNT(*) FROM conversation_turns").fetchone()[0]
            users = len(self.get_all_users())
        return {
            "total_turns": total_turns,
            "unique_users": users,
            "max_states_per_user": settings.max_memory_states,
            "db_path": self.db_path
        }
