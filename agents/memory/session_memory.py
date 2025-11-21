# memory/session_memory.py

class SessionMemory:
    """A lightweight in-memory store for the current session (short-term memory)."""

    def __init__(self):
        self._store = {}

    def set(self, key, value):
        """Store a value in memory."""
        self._store[key] = value

    def get(self, key, default=None):
        """Retrieve a value from memory."""
        return self._store.get(key, default)

    def clear(self):
        """Clear all stored memory."""
        self._store.clear()
