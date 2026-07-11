"""Basic tests for memory module."""
import pytest
import tempfile
import os
from app.memory.db import MemoryStore

def test_memory_trim():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        store = MemoryStore(db_path=db_path)
        
        user = "test_user"
        # Add 10 turns
        for i in range(10):
            store.add_turn(user, "human", f"Message {i}")
        
        turns = store.get_last_n_turns(user, n=6)
        assert len(turns) == 6, "Should retain only last 6 turns"
        assert turns[0]["content"] == "Message 4"  # After trim, oldest kept is 4 (0-9 -> keep 4 to 9)
        
        # Verify trim actually deleted older
        all_turns = store.get_last_n_turns(user, n=20)
        assert len(all_turns) == 6
