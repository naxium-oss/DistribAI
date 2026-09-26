"""
Database migration system for DistribAI.

Manages schema versioning and migrations.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Database migration definition."""

    version: int
    name: str
    up_sql: str
    down_sql: str | None = None


class MigrationManager:
    """Manage database migrations."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.migrations: list[Migration] = []

    def register_migration(self, migration: Migration):
        """Register a migration."""
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)

    def _ensure_migrations_table(self, conn: sqlite3.Connection):
        """Ensure migrations tracking table exists."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

    def get_current_version(self) -> int:
        """Get current database schema version."""
        if not self.db_path.exists():
            return 0

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                self._ensure_migrations_table(conn)
                cursor = conn.execute("SELECT MAX(version) FROM schema_migrations")
                result = cursor.fetchone()
                return result[0] or 0
        except sqlite3.Error as e:
            logger.error("Failed to get current version: %s", e)
            return 0

    def migrate_up(self, target_version: int | None = None) -> bool:
        """Apply pending migrations."""
        current = self.get_current_version()

        if target_version is None:
            target_version = max(m.version for m in self.migrations) if self.migrations else 0

        if current >= target_version:
            logger.info("Database already at version %s", current)
            return True

        pending = [m for m in self.migrations if current < m.version <= target_version]

        if not pending:
            logger.info("No pending migrations")
            return True

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                self._ensure_migrations_table(conn)

                for migration in pending:
                    logger.info("Applying migration %s: %s", migration.version, migration.name)

                    # Execute migration SQL
                    conn.executescript(migration.up_sql)

                    # Record migration
                    conn.execute(
                        "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                        (migration.version, migration.name),
                    )
                    conn.commit()

                    logger.info("Migration %s applied successfully", migration.version)

            logger.info("Database migrated to version %s", target_version)
            return True

        except sqlite3.Error as e:
            logger.error("Migration failed: %s", e)
            return False

    def migrate_down(self, target_version: int) -> bool:
        """Rollback to a previous version."""
        current = self.get_current_version()

        if current <= target_version:
            logger.info("Database already at or below version %s", target_version)
            return True

        to_rollback = [m for m in self.migrations if target_version < m.version <= current]
        to_rollback.sort(key=lambda m: m.version, reverse=True)

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                for migration in to_rollback:
                    if not migration.down_sql:
                        logger.warning("Migration %s has no rollback", migration.version)
                        continue

                    logger.info("Rolling back migration %s", migration.version)
                    conn.executescript(migration.down_sql)

                    conn.execute(
                        "DELETE FROM schema_migrations WHERE version = ?", (migration.version,)
                    )
                    conn.commit()

            logger.info("Database rolled back to version %s", target_version)
            return True

        except sqlite3.Error as e:
            logger.error("Rollback failed: %s", e)
            return False

    def get_status(self) -> dict:
        """Get migration status."""
        current = self.get_current_version()
        latest = max(m.version for m in self.migrations) if self.migrations else 0

        pending = [m for m in self.migrations if m.version > current]

        return {
            "current_version": current,
            "latest_version": latest,
            "pending_count": len(pending),
            "pending_migrations": [m.name for m in pending],
            "is_latest": current >= latest,
        }


# Define migrations
MIGRATIONS = [
    Migration(
        version=1,
        name="initial_schema",
        up_sql="""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                status TEXT DEFAULT 'queued',
                model_name TEXT,
                steps INTEGER,
                batch_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'offline',
                last_seen TIMESTAMP,
                credits_earned REAL DEFAULT 0,
                jobs_completed INTEGER DEFAULT 0,
                benchmark_score REAL
            );

            CREATE TABLE IF NOT EXISTS credits (
                node_id TEXT PRIMARY KEY,
                balance REAL DEFAULT 0,
                lifetime_earned REAL DEFAULT 0,
                lifetime_spent REAL DEFAULT 0,
                FOREIGN KEY (node_id) REFERENCES nodes(node_id)
            );
        """,
        down_sql="""
            DROP TABLE IF EXISTS jobs;
            DROP TABLE IF EXISTS nodes;
            DROP TABLE IF EXISTS credits;
        """,
    ),
    Migration(
        version=2,
        name="add_job_progress",
        up_sql="""
            ALTER TABLE jobs ADD COLUMN current_step INTEGER DEFAULT 0;
            ALTER TABLE jobs ADD COLUMN progress_percent REAL DEFAULT 0;
            ALTER TABLE jobs ADD COLUMN assigned_to TEXT;
        """,
        down_sql="""
            -- SQLite doesn't support dropping columns directly
            -- Would need to recreate table
        """,
    ),
    Migration(
        version=3,
        name="add_node_metadata",
        up_sql="""
            ALTER TABLE nodes ADD COLUMN region TEXT;
            ALTER TABLE nodes ADD COLUMN cpu_cores INTEGER;
            ALTER TABLE nodes ADD COLUMN memory_gb REAL;
            ALTER TABLE nodes ADD COLUMN gpu_model TEXT;
            ALTER TABLE nodes ADD COLUMN gpu_vram_gb REAL;
        """,
        down_sql=None,
    ),
    Migration(
        version=4,
        name="add_resource_limits",
        up_sql="""
            CREATE TABLE IF NOT EXISTS node_settings (
                node_id TEXT PRIMARY KEY,
                cpu_percent INTEGER DEFAULT 50,
                gpu_percent INTEGER DEFAULT 50,
                ram_percent INTEGER DEFAULT 50,
                FOREIGN KEY (node_id) REFERENCES nodes(node_id)
            );
        """,
        down_sql="""
            DROP TABLE IF EXISTS node_settings;
        """,
    ),
]


def create_migration_manager(db_path: Path) -> MigrationManager:
    """Create a migration manager with all registered migrations."""
    manager = MigrationManager(db_path)

    for migration in MIGRATIONS:
        manager.register_migration(migration)

    return manager


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python migrations.py <db_path> [command] [args]")
        print("Commands:")
        print("  status              Show migration status")
        print("  migrate [version]   Run migrations (default: latest)")
        print("  rollback <version>  Rollback to version")
        sys.exit(1)

    db_path = Path(sys.argv[1])
    command = sys.argv[2] if len(sys.argv) > 2 else "status"

    manager = create_migration_manager(db_path)

    if command == "status":
        status = manager.get_status()
        print(json.dumps(status, indent=2))

    elif command == "migrate":
        target = int(sys.argv[3]) if len(sys.argv) > 3 else None
        success = manager.migrate_up(target)
        sys.exit(0 if success else 1)

    elif command == "rollback":
        if len(sys.argv) < 4:
            print("Error: rollback requires a version number")
            sys.exit(1)
        target = int(sys.argv[3])
        success = manager.migrate_down(target)
        sys.exit(0 if success else 1)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
