"""
schemas.py ─ Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, ConfigDict, EmailStr
from typing import Optional


# ─── User & Auth Schemas ──────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    plan: str
    monthly_limit: int


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None
