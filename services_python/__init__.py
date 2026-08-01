"""Backend service package for DistribAI.

Keep this package initializer lightweight: importing a leaf module such as
``services_python.db_manager`` must not start importing the orchestrator stack.
"""

__all__ = ["serve", "DBManager"]


def __getattr__(name: str):
    if name == "DBManager":
        from .db_manager import DBManager

        return DBManager
    if name == "serve":
        from .orchestrator_grpc import serve

        return serve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Version information for security compliance
__version__ = "1.0.0"
__version_info__ = (1, 0, 0)
