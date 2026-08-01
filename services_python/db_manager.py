"""Compatibility shim: prefer ``services_python.db.DBManager`` for new code.

Re-exports the SQLite orchestrator facade so older imports
(``from services_python.db_manager import DBManager``) keep working.
"""

from services_python.db import DBManager

__all__ = ["DBManager"]
