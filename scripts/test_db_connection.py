"""Test database connection using settings from .env"""

import sys
from pathlib import Path

# Allow imports from project root when run as: python scripts/test_db_connection.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import verify_database_connection, DATABASE_URL, USE_LOCAL_DB


def main() -> None:
    if USE_LOCAL_DB:
        print("Mode: SQLite (USE_LOCAL_DB=true)")
    else:
        host = DATABASE_URL.split("@")[-1].split("/")[0] if "@" in DATABASE_URL else "unknown"
        print(f"Mode: Render PostgreSQL ({host})")

    ok, message = verify_database_connection()
    if ok:
        print(f"SUCCESS: {message}")
    else:
        print("FAILED:", message[:300])
        print()
        print("Fix on Render dashboard:")
        print("  1. Open your PostgreSQL instance")
        print("  2. Confirm status is 'Available' (not Suspended)")
        print("  3. Copy 'External Database URL'")
        print("  4. Paste it as DATABASE_URL in .env")
        print("  5. Keep USE_LOCAL_DB=false")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
