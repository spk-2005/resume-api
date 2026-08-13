import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

logger = logging.getLogger(__name__)

USE_LOCAL_DB = os.getenv("USE_LOCAL_DB", "").lower() in ("1", "true", "yes")
DATABASE_URL = os.getenv("DATABASE_URL")

if USE_LOCAL_DB or not DATABASE_URL:
    DATABASE_URL = "sqlite:///./resume_api.db"

# SQLAlchemy requires postgresql:// (Render sometimes returns postgres://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_is_sqlite = DATABASE_URL.startswith("sqlite")
_is_postgres = DATABASE_URL.startswith("postgresql")

connect_args: dict = {"check_same_thread": False} if _is_sqlite else {}

if _is_postgres:
    connect_args["connect_timeout"] = 15
    if "sslmode=" not in DATABASE_URL:
        separator = "&" if "?" in DATABASE_URL else "?"
        DATABASE_URL = f"{DATABASE_URL}{separator}sslmode=require"

engine_kwargs = {
    "connect_args": connect_args,
    "pool_pre_ping": True,
}

if _is_postgres:
    engine_kwargs.update(
        pool_size=5,
        max_overflow=10,
        pool_recycle=1800,
    )

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_database_connection() -> tuple[bool, str]:
    """Returns (ok, message) for startup health checks."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        if _is_sqlite:
            return True, "sqlite (local file: resume_api.db)"
        host = DATABASE_URL.split("@")[-1].split("/")[0] if "@" in DATABASE_URL else "postgresql"
        return True, f"postgresql ({host})"
    except Exception as exc:
        return False, str(exc)


_db_ok, _db_message = verify_database_connection()
if _db_ok:
    logger.info("Database connected: %s", _db_message)
else:
    logger.error(
        "Database connection failed: %s. "
        "For Render: copy a fresh External Database URL into DATABASE_URL in .env "
        "and confirm the database is not suspended.",
        _db_message,
    )
