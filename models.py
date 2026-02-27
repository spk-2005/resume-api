
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String, unique=True, index=True, nullable=False)
    plan          = Column(String, default="basic")           # basic / pro / ultra / mega
    monthly_limit = Column(Integer, default=50)               # max requests per month
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=datetime.utcnow)

    api_keys   = relationship("APIKey",   back_populates="user", cascade="all, delete")
    usage_logs = relationship("UsageLog", back_populates="user", cascade="all, delete")

    def __repr__(self):
        return f"<User id={self.id} email={self.email} plan={self.plan} limit={self.monthly_limit}>"


class APIKey(Base):
    __tablename__ = "api_keys"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    api_key    = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="api_keys")

    def __repr__(self):
        return f"<APIKey user_id={self.user_id}>"


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False)

    # ── THIS IS THE CRITICAL COLUMN ──────────────────────────────
    # request_time is used to count how many requests this month.
    # It MUST default to datetime.utcnow so every log row is timestamped.
    request_time = Column(DateTime, default=datetime.utcnow, nullable=False)

    endpoint     = Column(String, default="unknown")   # which endpoint was called
    status       = Column(String, default="success")   # success / error

    user = relationship("User", back_populates="usage_logs")

    def __repr__(self):
        return f"<UsageLog user_id={self.user_id} time={self.request_time}>"
