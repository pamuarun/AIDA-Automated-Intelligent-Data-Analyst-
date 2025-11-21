# memory/long_term_memory.py
import json
from pathlib import Path

class LongTermMemory:
    """File-backed memory for storing summaries or insights across sessions."""

    def __init__(self, path: str = 'memory_store.json'):
        self.path = Path(path)
        if not self.path.exists():
            self._write({})

    def _read(self):
        with open(self.path, 'r', encoding='utf8') as f:
            return json.load(f)

    def _write(self, data):
        with open(self.path, 'w', encoding='utf8') as f:
            json.dump(data, f, indent=2)

    def store_summary(self, name: str, summary: dict):
        """Store a summary by name."""
        data = self._read()
        data[name] = summary
        self._write(data)

    def get_summary(self, name: str):
        """Retrieve a stored summary by name."""
        data = self._read()
        return data.get(name)
