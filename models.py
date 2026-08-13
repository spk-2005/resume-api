
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Index, Text, JSON
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True) # Nullable for users created via old system
    plan          = Column(String, default="basic")           # basic / pro / ultra / mega
    monthly_limit = Column(Integer, default=50)               # max requests per month
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=datetime.utcnow)

    usage_logs = relationship("UsageLog", back_populates="user", cascade="all, delete")

    def __repr__(self):
        return f"<User id={self.id} email={self.email} plan={self.plan} limit={self.monthly_limit}>"


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id           = Column(Integer, primary_key=True, index=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False)

    # ── THIS IS THE CRITICAL COLUMN ──────────────────────────────
    # request_time is used to count how many requests this month.
    # It MUST default to datetime.utcnow so every log row is timestamped.
    request_time = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    endpoint     = Column(String, default="unknown")   # which endpoint was called
    status       = Column(String, default="success")   # success / error

    user = relationship("User", back_populates="usage_logs")

    __table_args__ = (
        Index('idx_user_request_time', 'user_id', 'request_time'),
    )

    def __repr__(self):
        return f"<UsageLog user_id={self.user_id} time={self.request_time}>"


class Job(Base):
    __tablename__ = "jobs"

    id                      = Column(Integer, primary_key=True, index=True)
    user_id                 = Column(Integer, ForeignKey("users.id"), nullable=False)
    raw_text                = Column(Text, nullable=False)
    structured_requirements = Column(JSON, nullable=True) # Stores the output of the JD Analyzer
    created_at              = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")

    def __repr__(self):
        return f"<Job id={self.id} user_id={self.user_id}>"


class Candidate(Base):
    __tablename__ = "candidates"

    id                = Column(Integer, primary_key=True, index=True)
    user_id           = Column(Integer, ForeignKey("users.id"), nullable=False)
    raw_text          = Column(Text, nullable=False)
    structured_evidence = Column(JSON, nullable=True) # Stores the output of the Evidence Analyzer
    created_at        = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")

    def __repr__(self):
        return f"<Candidate id={self.id} user_id={self.user_id}>"


class AnalysisReport(Base):
    __tablename__ = "analysis_reports"

    id           = Column(Integer, primary_key=True, index=True)
    job_id       = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    report_data  = Column(JSON, nullable=False) # The final evidence-backed report
    created_at   = Column(DateTime, default=datetime.utcnow)
