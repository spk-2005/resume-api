"""
main.py  ─  Resume ATS Intelligence API  v3.0.0
─────────────────────────────────────────────────
HOW MONTHLY LIMITS WORK:
─────────────────────────
  Every request goes through get_rapidapi_user() FIRST.
  It counts rows in usage_logs WHERE user_id = ? AND month = current_month.
  If count >= monthly_limit  →  HTTP 429 is raised immediately.
  If count < monthly_limit   →  request proceeds, then log_usage() adds 1 row.

  Plan limits (auto-synced from X-RapidAPI-Subscription header):
    BASIC   →   50 requests / month
    PRO     →  500 requests / month
    ULTRA   → 2000 requests / month
    MEGA    → 10000 requests / month

  Set RAPIDAPI_PROXY_SECRET env variable from:
  RapidAPI Dashboard → My APIs → Your API → Security → Proxy Secret
"""

import io
import os
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import (
    FastAPI, Depends, HTTPException,
    UploadFile, File, Form, Request
)
from sqlalchemy import func, extract
from sqlalchemy.orm import Session

from database import SessionLocal, engine
from models import Base, User, APIKey, UsageLog
from services.resume_analyzer import analyze_resume


# ══════════════════════════════════════════════════════════════════
#  OPTIONAL IMPORTS
# ══════════════════════════════════════════════════════════════════

try:
    from pypdf import PdfReader
    PYPDF_OK = True
except ImportError:
    PYPDF_OK = False

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_OK = True
except ImportError:
    OCR_OK = False

try:
    import docx
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

try:
    import textract
    TEXTRACT_OK = True
except ImportError:
    TEXTRACT_OK = False

try:
    from PIL import Image
    import pytesseract
    PIL_OK = True
except ImportError:
    PIL_OK = False


# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════

RAPIDAPI_PROXY_SECRET = os.environ.get("RAPIDAPI_PROXY_SECRET", "")

PLAN_LIMITS = {
    "BASIC":   50,
    "PRO":     500,
    "ULTRA":   2000,
    "MEGA":    10000,
    "CUSTOM":  99999,
}

PDF_EXTS   = {".pdf"}
DOCX_EXTS  = {".docx"}
DOC_EXTS   = {".doc"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
TEXT_EXTS  = {".txt"}
ALL_SUPPORTED = PDF_EXTS | DOCX_EXTS | DOC_EXTS | IMAGE_EXTS | TEXT_EXTS


# ══════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Resume ATS Intelligence API",
    version="3.0.0",
    description="""
## Resume ATS Intelligence API  v3.0.0

Analyze resumes against job descriptions with a full ATS score report.

### How Monthly Limits Work
Each API plan comes with a monthly request limit:

| Plan  | Requests / Month |
|-------|-----------------|
| Basic  | 50  |
| Pro    | 500 |
| Ultra  | 2,000 |
| Mega   | 10,000 |

Once your limit is reached you will receive a **429 Too Many Requests** error
until the next calendar month resets your count.

### Supported File Formats
`PDF` · `DOCX` · `DOC` · `PNG` · `JPG` · `WEBP` · `TXT`
    """,
)
Base.metadata.create_all(bind=engine)


# ══════════════════════════════════════════════════════════════════
#  DB
# ══════════════════════════════════════════════════════════════════

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════════
#  USAGE COUNTING  (the heart of the rate limiting)
# ══════════════════════════════════════════════════════════════════

def _count_usage_this_month(user_id: int, db: Session) -> int:
    """
    Count how many requests this user has made in the current calendar month.
    Uses the request_time column in usage_logs table.
    """
    now = datetime.utcnow()
    return db.query(UsageLog).filter(
        UsageLog.user_id == user_id,
        extract("year",  UsageLog.request_time) == now.year,
        extract("month", UsageLog.request_time) == now.month,
    ).count()


def log_usage(user_id: int, endpoint: str, db: Session):
    """
    Write one row to usage_logs after a successful request.
    This is what gets counted next time _count_usage_this_month() runs.
    """
    db.add(UsageLog(
        user_id=user_id,
        request_time=datetime.utcnow(),
        endpoint=endpoint,
        status="success",
    ))
    db.commit()


# ══════════════════════════════════════════════════════════════════
#  AUTH + LIMIT ENFORCEMENT
# ══════════════════════════════════════════════════════════════════

def get_rapidapi_user(request: Request, db: Session = Depends(get_db)) -> User:
    """
    Runs before EVERY protected endpoint.

    Steps:
      1. Verify X-RapidAPI-Proxy-Secret  (proves request came from RapidAPI)
      2. Read X-RapidAPI-User            (the subscriber's username)
      3. Read X-RapidAPI-Subscription    (their plan → monthly limit)
      4. Auto-create user in DB if first time seen
      5. Count requests this month
      6. Block with HTTP 429 if limit reached
    """

    # ── Step 1: Proxy secret check ────────────────────────────────
    incoming_secret = request.headers.get("x-rapidapi-proxy-secret", "")
    if RAPIDAPI_PROXY_SECRET and incoming_secret != RAPIDAPI_PROXY_SECRET:
        raise HTTPException(
            status_code=403,
            detail="Forbidden. This API must be called through RapidAPI.",
        )

    # ── Step 2: RapidAPI username ─────────────────────────────────
    rapidapi_user = request.headers.get("x-rapidapi-user", "").strip()
    if not rapidapi_user:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized. Subscribe to this API on RapidAPI to get access.",
        )

    # ── Step 3: Plan & limit ──────────────────────────────────────
    subscription  = request.headers.get("x-rapidapi-subscription", "BASIC").strip().upper()
    monthly_limit = PLAN_LIMITS.get(subscription, 50)

    # ── Step 4: Auto-create / sync user ──────────────────────────
    user = db.query(User).filter(User.email == rapidapi_user).first()
    if not user:
        # First time this RapidAPI user hits your API — create their record
        user = User(
            email=rapidapi_user,
            plan=subscription.lower(),
            monthly_limit=monthly_limit,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # Sync plan changes (e.g. user upgraded from BASIC to PRO)
        if user.plan != subscription.lower() or user.monthly_limit != monthly_limit:
            user.plan          = subscription.lower()
            user.monthly_limit = monthly_limit
            db.commit()

    # ── Step 5 & 6: Count + enforce ──────────────────────────────
    used = _count_usage_this_month(user.id, db)

    if used >= user.monthly_limit:
        raise HTTPException(
            status_code=429,
            detail={
                "error":         "Monthly request limit reached.",
                "plan":          user.plan.upper(),
                "limit":         user.monthly_limit,
                "used":          used,
                "remaining":     0,
                "resets":        f"1st of next month (UTC)",
                "upgrade":       "Visit RapidAPI to upgrade your plan for more requests.",
            },
        )

    return user


# ══════════════════════════════════════════════════════════════════
#  TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════

def _extract_pdf(file_bytes: bytes, filename: str) -> str:
    if not PYPDF_OK:
        raise HTTPException(status_code=501, detail="pypdf not installed.")
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text   = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read PDF '{filename}': {e}")

    if text:
        return text

    if not OCR_OK:
        raise HTTPException(status_code=422, detail=f"'{filename}' is a scanned PDF. OCR not available.")
    try:
        images = convert_from_bytes(file_bytes, dpi=200)
        text   = "\n".join(pytesseract.image_to_string(img, lang="eng") for img in images).strip()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"OCR failed on '{filename}': {e}")

    if not text:
        raise HTTPException(status_code=422, detail=f"No text extracted from '{filename}' even after OCR.")
    return text


def _extract_docx(file_bytes: bytes, filename: str) -> str:
    if not DOCX_OK:
        raise HTTPException(status_code=501, detail="python-docx not installed.")
    try:
        doc   = docx.Document(io.BytesIO(file_bytes))
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        for section in doc.sections:
            for p in (section.header.paragraphs if section.header else []):
                parts.append(p.text)
            for p in (section.footer.paragraphs if section.footer else []):
                parts.append(p.text)
        text = "\n".join(p for p in parts if p.strip()).strip()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read DOCX '{filename}': {e}")
    if not text:
        raise HTTPException(status_code=422, detail=f"No text extracted from '{filename}'.")
    return text


def _extract_doc(file_bytes: bytes, filename: str) -> str:
    if not TEXTRACT_OK:
        raise HTTPException(status_code=501, detail="textract not installed.")
    try:
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        text = textract.process(tmp_path).decode("utf-8", errors="replace").strip()
        os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read DOC '{filename}': {e}")
    if not text:
        raise HTTPException(status_code=422, detail=f"No text extracted from '{filename}'.")
    return text


def _extract_image(file_bytes: bytes, filename: str) -> str:
    if not PIL_OK:
        raise HTTPException(status_code=501, detail="Pillow/pytesseract not installed.")
    try:
        text = pytesseract.image_to_string(Image.open(io.BytesIO(file_bytes)), lang="eng").strip()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"OCR failed on '{filename}': {e}")
    if not text:
        raise HTTPException(status_code=422, detail=f"No text found in image '{filename}'.")
    return text


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in TEXT_EXTS:   return file_bytes.decode("utf-8", errors="replace").strip()
    if ext in PDF_EXTS:    return _extract_pdf(file_bytes, filename)
    if ext in DOCX_EXTS:   return _extract_docx(file_bytes, filename)
    if ext in DOC_EXTS:    return _extract_doc(file_bytes, filename)
    if ext in IMAGE_EXTS:  return _extract_image(file_bytes, filename)
    raise HTTPException(
        status_code=415,
        detail=f"Unsupported file type '{ext}'. Accepted: {', '.join(sorted(ALL_SUPPORTED))}",
    )


def filename_to_name(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def _collect_files(form) -> List[UploadFile]:
    """Collect uploaded files regardless of field name used by the client."""
    found, seen = [], set()
    for field in ("resumes", "resume_file", "resume", "file", "files"):
        for item in form.getlist(field):
            if hasattr(item, "read") and id(item) not in seen:
                seen.add(id(item))
                found.append(item)
    for key in form.keys():
        for item in form.getlist(key):
            if hasattr(item, "read") and id(item) not in seen:
                seen.add(id(item))
                found.append(item)
    return found


# ══════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post(
    "/analyze-resume/",
    tags=["Resume Analysis"],
    summary="Upload one resume → full ATS report",
)
async def analyze_resume_endpoint(
    request:         Request,
    job_description: str           = Form(...),
    resume_file:     UploadFile    = File(...),
    candidate_name:  Optional[str] = Form(None),
    user: User    = Depends(get_rapidapi_user),
    db:   Session = Depends(get_db),
):
    """
    Upload **one** resume + paste the job description.
    Returns a full structured ATS report.

    **Counts as 1 request** toward your monthly limit.
    """
    # Read & analyze
    raw_bytes = await resume_file.read()
    text      = extract_text(raw_bytes, resume_file.filename)
    name      = candidate_name or filename_to_name(resume_file.filename)
    result    = analyze_resume(resume_text=text, job_description=job_description, candidate_name=name)

    # Log AFTER success
    log_usage(user.id, endpoint="/analyze-resume/", db=db)

    # Append usage info to response
    used = _count_usage_this_month(user.id, db)
    result["usage"] = {
        "requests_used":      used,
        "requests_limit":     user.monthly_limit,
        "requests_remaining": max(0, user.monthly_limit - used),
        "plan":               user.plan.upper(),
    }

    return result


@app.post(
    "/bulk-analyze/",
    tags=["Resume Analysis"],
    summary="Upload multiple resumes → ranked ATS list",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["job_description", "resumes"],
                        "properties": {
                            "job_description": {"type": "string"},
                            "resumes": {
                                "type":  "array",
                                "items": {"type": "string", "format": "binary"},
                            },
                        },
                    }
                }
            },
        }
    },
)
async def bulk_analyze(
    request: Request,
    user: User    = Depends(get_rapidapi_user),
    db:   Session = Depends(get_db),
):
    """
    Upload **multiple resumes** against one job description.
    Returns all candidates ranked by ATS score (highest first).

    **Each file counts as 1 request** toward your monthly limit.
    Uploading 5 files = 5 requests consumed.
    """
    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse form data: {e}")

    job_description = form.get("job_description", "").strip()
    if not job_description:
        raise HTTPException(status_code=422, detail="Field 'job_description' is required.")

    resume_files = _collect_files(form)
    if not resume_files:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "No resume files found.",
                "tip":   "Send files as multipart fields named 'resumes'.",
                "received_fields": list(form.keys()),
            },
        )

    # ── Pre-flight: do they have enough requests left? ────────────
    used      = _count_usage_this_month(user.id, db)
    remaining = user.monthly_limit - used

    if len(resume_files) > remaining:
        raise HTTPException(
            status_code=429,
            detail={
                "error":         "Not enough requests remaining for this bulk upload.",
                "files_sent":    len(resume_files),
                "requests_remaining": remaining,
                "plan":          user.plan.upper(),
                "limit":         user.monthly_limit,
                "tip":           "Upload fewer files or upgrade your plan on RapidAPI.",
            },
        )

    # ── Process each file ─────────────────────────────────────────
    results: List[dict] = []
    errors:  List[dict] = []

    for resume_file in resume_files:
        fname = getattr(resume_file, "filename", None) or "unknown"
        try:
            raw_bytes = await resume_file.read()
            if not raw_bytes:
                errors.append({"file": fname, "error": "File is empty."})
                continue
            text   = extract_text(raw_bytes, fname)
            report = analyze_resume(
                resume_text=text,
                job_description=job_description,
                candidate_name=filename_to_name(fname),
            )
            results.append(report)
            log_usage(user.id, endpoint="/bulk-analyze/", db=db)  # 1 log per file
        except HTTPException as e:
            errors.append({"file": fname, "error": e.detail})
        except Exception as e:
            errors.append({"file": fname, "error": str(e)})

    if not results and errors:
        raise HTTPException(
            status_code=422,
            detail={"message": "All files failed to process.", "errors": errors},
        )

    # ── Rank ──────────────────────────────────────────────────────
    ranked      = sorted(results, key=lambda r: r["ats_score"]["score"], reverse=True)
    shortlisted = [r for r in ranked if r["ats_score"]["score"] >= 70]

    # Final usage count
    final_used = _count_usage_this_month(user.id, db)

    return {
        "summary": {
            "total_uploaded":        len(resume_files),
            "successfully_analyzed": len(ranked),
            "failed":                len(errors),
            "shortlisted_above_70":  len(shortlisted),
            "top_candidate":         ranked[0]["candidate"]["name"] if ranked else None,
            "top_ats_score":         ranked[0]["ats_score"]["score"] if ranked else None,
        },
        "usage": {
            "requests_used":      final_used,
            "requests_limit":     user.monthly_limit,
            "requests_remaining": max(0, user.monthly_limit - final_used),
            "plan":               user.plan.upper(),
        },
        "ranked_candidates": [
            {
                "rank":                  i + 1,
                "candidate":             r["candidate"],
                "ats_score":             r["ats_score"],
                "skill_analysis":        r["skill_analysis"],
                "keyword_density":       r["keyword_density"],
                "section_analysis":      r["section_analysis"],
                "experience":            r["experience"],
                "education":             r["education"],
                "job_title_alignment":   r["job_title_alignment"],
                "writing_quality":       r["writing_quality"],
                "format_and_contact":    r["format_and_contact"],
            }
            for i, r in enumerate(ranked)
        ],
        "failed_files": errors,
    }