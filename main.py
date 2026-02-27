"""
main.py  ─  Resume ATS Intelligence API  v3.0.0
"""

import io
import os
import logging
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import (
    FastAPI, Depends, HTTPException,
    UploadFile, File, Form, Request
)
from fastapi.responses import JSONResponse
from sqlalchemy import extract
from sqlalchemy.orm import Session

from database import SessionLocal, engine
from models import Base, User, APIKey, UsageLog
from services.resume_analyzer import analyze_resume


# ══════════════════════════════════════════════════════════════════
#  LOGGING  — errors will now appear in your server logs
# ══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  OPTIONAL IMPORTS
# ══════════════════════════════════════════════════════════════════

try:
    from pypdf import PdfReader
    PYPDF_OK = True
except ImportError:
    PYPDF_OK = False
    logger.warning("pypdf not installed — PDF support disabled.")

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_OK = True
except ImportError:
    OCR_OK = False
    logger.warning("pdf2image/pytesseract not installed — scanned PDF OCR disabled.")

try:
    import docx
    DOCX_OK = True
except ImportError:
    DOCX_OK = False
    logger.warning("python-docx not installed — DOCX support disabled.")

try:
    import textract
    TEXTRACT_OK = True
except ImportError:
    TEXTRACT_OK = False
    logger.warning("textract not installed — DOC support disabled.")

try:
    from PIL import Image
    import pytesseract
    PIL_OK = True
except ImportError:
    PIL_OK = False
    logger.warning("Pillow/pytesseract not installed — image OCR disabled.")


# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════

RAPIDAPI_PROXY_SECRET = os.environ.get("RAPIDAPI_PROXY_SECRET", "")

PLAN_LIMITS = {
    "BASIC":  50,
    "PRO":    500,
    "ULTRA":  2000,
    "MEGA":   10000,
    "CUSTOM": 99999,
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
)

# Create all DB tables on startup
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified OK.")
except Exception as e:
    logger.error(f"FATAL: Could not create DB tables: {e}")


# ══════════════════════════════════════════════════════════════════
#  HEALTH CHECK  — required for RapidAPI health check to pass
#  Set Health Check URL to: https://resume-api-c908.onrender.com/health
# ══════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Health"], include_in_schema=False)
@app.get("/",       tags=["Health"], include_in_schema=False)
async def health_check():
    """
    Health check endpoint.
    RapidAPI pings this daily — must return HTTP 200.
    Also handles root URL so there is no 404.
    """
    db_status = "ok"
    try:
        db = SessionLocal()
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db.close()
    except Exception as e:
        db_status = f"error: {e}"

    return {
        "status":  "ok",
        "service": "Resume ATS Intelligence API",
        "version": "3.0.0",
        "database": db_status,
    }


# ══════════════════════════════════════════════════════════════════
#  GLOBAL ERROR HANDLER  — turns any unhandled crash into clean JSON
# ══════════════════════════════════════════════════════════════════

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(f"Unhandled error on {request.url.path}:\n{tb}")
    return JSONResponse(
        status_code=500,
        content={
            "error":   "Internal Server Error",
            "detail":  str(exc),
            "type":    type(exc).__name__,
            # Remove 'trace' in production for security
            "trace":   tb.splitlines()[-3:],
        },
    )


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
#  USAGE TRACKING
# ══════════════════════════════════════════════════════════════════

def _count_usage_this_month(user_id: int, db: Session) -> int:
    now = datetime.utcnow()
    try:
        return db.query(UsageLog).filter(
            UsageLog.user_id == user_id,
            extract("year",  UsageLog.request_time) == now.year,
            extract("month", UsageLog.request_time) == now.month,
        ).count()
    except Exception as e:
        logger.error(f"Error counting usage for user {user_id}: {e}")
        return 0


def log_usage(user_id: int, endpoint: str, db: Session):
    try:
        db.add(UsageLog(
            user_id=user_id,
            request_time=datetime.utcnow(),
            endpoint=endpoint,
            status="success",
        ))
        db.commit()
        logger.info(f"Usage logged: user_id={user_id} endpoint={endpoint}")
    except Exception as e:
        logger.error(f"Failed to log usage for user {user_id}: {e}")
        db.rollback()


# ══════════════════════════════════════════════════════════════════
#  AUTH + LIMIT ENFORCEMENT
# ══════════════════════════════════════════════════════════════════

def get_rapidapi_user(request: Request, db: Session = Depends(get_db)) -> User:
    try:
        # Step 1: Proxy secret
        incoming_secret = request.headers.get("x-rapidapi-proxy-secret", "")
        if RAPIDAPI_PROXY_SECRET and incoming_secret != RAPIDAPI_PROXY_SECRET:
            logger.warning(f"Invalid proxy secret from {request.client.host}")
            raise HTTPException(
                status_code=403,
                detail="Forbidden. This API must be called through RapidAPI.",
            )

        # Step 2: RapidAPI username
        rapidapi_user = request.headers.get("x-rapidapi-user", "").strip()
        if not rapidapi_user:
            logger.warning("Request missing x-rapidapi-user header")
            raise HTTPException(
                status_code=401,
                detail="Unauthorized. Subscribe to this API on RapidAPI to get access.",
            )

        # Step 3: Plan & limit
        subscription  = request.headers.get("x-rapidapi-subscription", "BASIC").strip().upper()
        monthly_limit = PLAN_LIMITS.get(subscription, 50)
        logger.info(f"Request from user={rapidapi_user} plan={subscription} limit={monthly_limit}")

        # Step 4: Auto-create / sync user
        user = db.query(User).filter(User.email == rapidapi_user).first()
        if not user:
            logger.info(f"New user detected, creating record: {rapidapi_user}")
            user = User(
                email=rapidapi_user,
                plan=subscription.lower(),
                monthly_limit=monthly_limit,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            if user.plan != subscription.lower() or user.monthly_limit != monthly_limit:
                logger.info(f"Updating plan for {rapidapi_user}: {user.plan} → {subscription.lower()}")
                user.plan          = subscription.lower()
                user.monthly_limit = monthly_limit
                db.commit()

        # Step 5: Count usage
        used = _count_usage_this_month(user.id, db)
        logger.info(f"User {rapidapi_user} used {used}/{monthly_limit} requests this month")

        # Step 6: Enforce limit
        if used >= user.monthly_limit:
            raise HTTPException(
                status_code=429,
                detail={
                    "error":              "Monthly request limit reached.",
                    "plan":               user.plan.upper(),
                    "limit":              user.monthly_limit,
                    "used":               used,
                    "remaining":          0,
                    "resets":             "1st of next month (UTC)",
                    "upgrade":            "Visit RapidAPI to upgrade your plan.",
                },
            )

        return user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_rapidapi_user: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Auth error: {str(e)}")


# ══════════════════════════════════════════════════════════════════
#  DEBUG ENDPOINT  (remove after debugging is done)
# ══════════════════════════════════════════════════════════════════

@app.get("/debug/", include_in_schema=False)
async def debug(request: Request):
    """
    Call this through RapidAPI to see exactly what headers are arriving.
    Delete this endpoint once everything is working.
    """
    return {
        "headers_received": dict(request.headers),
        "rapidapi_user":    request.headers.get("x-rapidapi-user", "NOT FOUND"),
        "rapidapi_plan":    request.headers.get("x-rapidapi-subscription", "NOT FOUND"),
        "proxy_secret_set": bool(RAPIDAPI_PROXY_SECRET),
        "proxy_secret_ok":  request.headers.get("x-rapidapi-proxy-secret", "") == RAPIDAPI_PROXY_SECRET
                            if RAPIDAPI_PROXY_SECRET else "PROXY_SECRET_NOT_SET_IN_ENV",
        "db_ok":            _check_db(),
        "imports": {
            "pypdf":        PYPDF_OK,
            "pdf2image":    OCR_OK,
            "python_docx":  DOCX_OK,
            "textract":     TEXTRACT_OK,
            "pillow":       PIL_OK,
        },
    }


def _check_db() -> str:
    try:
        db = SessionLocal()
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db.close()
        return "OK"
    except Exception as e:
        return f"ERROR: {e}"


# ══════════════════════════════════════════════════════════════════
#  TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════

def _extract_pdf(file_bytes: bytes, filename: str) -> str:
    if not PYPDF_OK:
        raise HTTPException(status_code=501, detail="pypdf not installed. Run: pip install pypdf")
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text   = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not read PDF '{filename}': {e}")

    if text:
        return text

    if not OCR_OK:
        raise HTTPException(status_code=422, detail=f"'{filename}' is a scanned PDF and OCR is not available.")
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
        document = docx.Document(io.BytesIO(file_bytes))
        parts    = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        for section in document.sections:
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
    try:
        logger.info(f"analyze-resume called by user_id={user.id} file={resume_file.filename}")

        raw_bytes = await resume_file.read()
        if not raw_bytes:
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")

        text   = extract_text(raw_bytes, resume_file.filename)
        name   = candidate_name or filename_to_name(resume_file.filename)
        result = analyze_resume(
            resume_text=text,
            job_description=job_description,
            candidate_name=name,
        )

        log_usage(user.id, endpoint="/analyze-resume/", db=db)

        used = _count_usage_this_month(user.id, db)
        result["usage"] = {
            "requests_used":      used,
            "requests_limit":     user.monthly_limit,
            "requests_remaining": max(0, user.monthly_limit - used),
            "plan":               user.plan.upper(),
        }

        logger.info(f"analyze-resume success for user_id={user.id} score={result.get('ats_score', {}).get('score')}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"analyze-resume crashed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


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
    try:
        logger.info(f"bulk-analyze called by user_id={user.id}")

        try:
            form = await request.form()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse form data: {e}")

        job_description = form.get("job_description", "").strip()
        if not job_description:
            raise HTTPException(status_code=422, detail="Field 'job_description' is required.")

        resume_files = _collect_files(form)
        logger.info(f"bulk-analyze: {len(resume_files)} files received, fields={list(form.keys())}")

        if not resume_files:
            raise HTTPException(
                status_code=422,
                detail={
                    "error":           "No resume files found in the request.",
                    "tip":             "Send files as multipart fields named 'resumes'.",
                    "received_fields": list(form.keys()),
                },
            )

        # Pre-flight usage check
        used      = _count_usage_this_month(user.id, db)
        remaining = user.monthly_limit - used
        if len(resume_files) > remaining:
            raise HTTPException(
                status_code=429,
                detail={
                    "error":               "Not enough requests remaining.",
                    "files_sent":          len(resume_files),
                    "requests_remaining":  remaining,
                    "plan":                user.plan.upper(),
                    "tip":                 "Upload fewer files or upgrade your plan on RapidAPI.",
                },
            )

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
                log_usage(user.id, endpoint="/bulk-analyze/", db=db)
                logger.info(f"bulk-analyze: processed {fname} score={report.get('ats_score', {}).get('score')}")

            except HTTPException as e:
                logger.warning(f"bulk-analyze: skipping {fname} — {e.detail}")
                errors.append({"file": fname, "error": e.detail})
            except Exception as e:
                logger.error(f"bulk-analyze: error on {fname}: {traceback.format_exc()}")
                errors.append({"file": fname, "error": str(e)})

        if not results and errors:
            raise HTTPException(
                status_code=422,
                detail={"message": "All files failed to process.", "errors": errors},
            )

        ranked      = sorted(results, key=lambda r: r["ats_score"]["score"], reverse=True)
        shortlisted = [r for r in ranked if r["ats_score"]["score"] >= 70]
        final_used  = _count_usage_this_month(user.id, db)

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
                    "rank":                i + 1,
                    "candidate":           r["candidate"],
                    "ats_score":           r["ats_score"],
                    "skill_analysis":      r["skill_analysis"],
                    "keyword_density":     r["keyword_density"],
                    "section_analysis":    r["section_analysis"],
                    "experience":          r["experience"],
                    "education":           r["education"],
                    "job_title_alignment": r["job_title_alignment"],
                    "writing_quality":     r["writing_quality"],
                    "format_and_contact":  r["format_and_contact"],
                }
                for i, r in enumerate(ranked)
            ],
            "failed_files": errors,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"bulk-analyze crashed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Bulk analysis failed: {str(e)}")