"""
main.py  ─  Resume ATS Intelligence API  v3.0.0
─────────────────────────────────────────────────
HOW RAPIDAPI AUTHENTICATION WORKS (when selling on marketplace):
─────────────────────────────────────────────────────────────────
  RapidAPI does NOT forward the user's X-RapidAPI-Key to your server.
  Instead, RapidAPI acts as a proxy and sends these headers to YOUR server:

    X-RapidAPI-Proxy-Secret  → a secret YOU set in RapidAPI dashboard
                               (proves the request came from RapidAPI)
    X-RapidAPI-User          → the subscribing user's RapidAPI username
    X-RapidAPI-Subscription  → their subscription plan name

  So your server should:
    1. Verify  X-RapidAPI-Proxy-Secret  matches your env variable
    2. Use     X-RapidAPI-User          as the user identifier
    3. Track usage per user in your DB

  Set this in your environment:
    RAPIDAPI_PROXY_SECRET=<copy from RapidAPI dashboard → My APIs → Security>

Accepts resumes as any of:
  • PDF   (text-based or scanned image PDF)
  • DOCX / DOC
  • PNG / JPG / JPEG / WEBP / GIF
  • TXT
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
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import SessionLocal, engine
from models import Base, User, APIKey, UsageLog
from services.resume_analyzer import analyze_resume


# ══════════════════════════════════════════════════════════════════
#  RAPIDAPI PROXY SECRET
#  Copy this value from:
#  RapidAPI Dashboard → My APIs → Your API → Security → Proxy Secret
# ══════════════════════════════════════════════════════════════════

RAPIDAPI_PROXY_SECRET = os.environ.get("RAPIDAPI_PROXY_SECRET", "")


# ══════════════════════════════════════════════════════════════════
#  OPTIONAL DEPENDENCY IMPORTS
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
#  SUPPORTED EXTENSIONS
# ══════════════════════════════════════════════════════════════════

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
    description="""
## Resume ATS Intelligence API  v3.0.0

Analyze resumes against job descriptions with a full ATS score.

### Authentication
This API is sold via **RapidAPI**. Subscribe on RapidAPI and send
your `X-RapidAPI-Key` header — RapidAPI handles authentication for you.

### Supported File Formats
| Format | Notes |
|--------|-------|
| **PDF** | Text-based PDFs parsed directly; scanned PDFs auto-OCR'd |
| **DOCX** | Word 2007+ documents |
| **DOC** | Legacy Word format |
| **PNG / JPG / WEBP / GIF** | Resume screenshots via OCR |
| **TXT** | Plain text |

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze-resume/` | Single file → ATS report |
| POST | `/bulk-analyze/` | Multiple files → ranked list |
| GET  | `/usage/` | Check your monthly usage |
    """,
    version="3.0.0",
)
Base.metadata.create_all(bind=engine)


# ══════════════════════════════════════════════════════════════════
#  DB HELPER
# ══════════════════════════════════════════════════════════════════

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════════
#  RAPIDAPI AUTH
#
#  When a user calls your API through RapidAPI, RapidAPI's proxy
#  forwards the request to your server with these headers:
#
#    X-RapidAPI-Proxy-Secret  →  your secret (verify this!)
#    X-RapidAPI-User          →  the caller's RapidAPI username
#    X-RapidAPI-Subscription  →  their plan (BASIC, PRO, etc.)
#
#  Your job:
#    1. Check the proxy secret matches what's in your dashboard
#    2. Auto-create a local user record for new RapidAPI users
#    3. Track & enforce usage limits per user
# ══════════════════════════════════════════════════════════════════

def _usage_this_month(user_id: int, db: Session) -> int:
    return db.query(UsageLog).filter(
        UsageLog.user_id == user_id,
        func.extract("month", UsageLog.request_time) == datetime.now().month,
    ).count()


def get_rapidapi_user(request: Request, db: Session = Depends(get_db)) -> User:
    """
    Auth dependency for all endpoints.

    Verifies the RapidAPI proxy secret, then looks up (or auto-creates)
    a local user based on the RapidAPI username header.
    """
    # ── Step 1: Verify this request actually came from RapidAPI ───
    proxy_secret = request.headers.get("x-rapidapi-proxy-secret", "")

    if not RAPIDAPI_PROXY_SECRET:
        # If env var not set, warn in logs but don't block (dev mode)
        import warnings
        warnings.warn(
            "RAPIDAPI_PROXY_SECRET env variable is not set! "
            "Set it to the value from your RapidAPI dashboard to secure your API.",
            stacklevel=2,
        )
    elif proxy_secret != RAPIDAPI_PROXY_SECRET:
        raise HTTPException(
            status_code=403,
            detail="Forbidden. This API must be called through RapidAPI.",
        )

    # ── Step 2: Get the caller's RapidAPI username ─────────────────
    rapidapi_user = request.headers.get("x-rapidapi-user", "").strip()
    if not rapidapi_user:
        raise HTTPException(
            status_code=401,
            detail=(
                "Could not identify caller. "
                "Please call this API through RapidAPI with a valid subscription."
            ),
        )

    # ── Step 3: Get subscription/plan ─────────────────────────────
    subscription = request.headers.get("x-rapidapi-subscription", "BASIC").strip().upper()

    # Map RapidAPI plan names to monthly limits
    plan_limits = {
        "BASIC":      50,
        "PRO":        500,
        "ULTRA":      2000,
        "MEGA":       10000,
        "CUSTOM":     99999,
    }
    monthly_limit = plan_limits.get(subscription, 50)

    # ── Step 4: Auto-create user if first time seen ────────────────
    # Use rapidapi_user as the email/identifier (stored as email in DB)
    user = db.query(User).filter(User.email == rapidapi_user).first()
    if not user:
        user = User(
            email=rapidapi_user,
            plan=subscription.lower(),
            monthly_limit=monthly_limit,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # Update plan/limit in case they upgraded
        if user.plan != subscription.lower() or user.monthly_limit != monthly_limit:
            user.plan          = subscription.lower()
            user.monthly_limit = monthly_limit
            db.commit()

    # ── Step 5: Enforce monthly limit ─────────────────────────────
    used = _usage_this_month(user.id, db)
    if used >= user.monthly_limit:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Monthly limit of {user.monthly_limit} requests reached. "
                "Please upgrade your plan on RapidAPI."
            ),
        )

    return user


def log_usage(user_id: int, db: Session):
    db.add(UsageLog(user_id=user_id))
    db.commit()


# ══════════════════════════════════════════════════════════════════
#  MULTI-FORMAT TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════

def _extract_pdf(file_bytes: bytes, filename: str) -> str:
    if not PYPDF_OK:
        raise HTTPException(status_code=501, detail="pypdf not installed. Run: pip install pypdf")
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages  = [page.extract_text() or "" for page in reader.pages]
        text   = "\n".join(pages).strip()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read '{filename}': {exc}")

    if text:
        return text

    if not OCR_OK:
        raise HTTPException(
            status_code=422,
            detail=f"'{filename}' is a scanned PDF. Install pdf2image + pytesseract for OCR.",
        )
    try:
        images    = convert_from_bytes(file_bytes, dpi=200)
        ocr_pages = [pytesseract.image_to_string(img, lang="eng") for img in images]
        text      = "\n".join(ocr_pages).strip()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"OCR failed on '{filename}': {exc}")

    if not text:
        raise HTTPException(status_code=422, detail=f"No text could be extracted from '{filename}' even after OCR.")
    return text


def _extract_docx(file_bytes: bytes, filename: str) -> str:
    if not DOCX_OK:
        raise HTTPException(status_code=501, detail="python-docx not installed. Run: pip install python-docx")
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        parts: List[str] = []
        for para in document.paragraphs:
            parts.append(para.text)
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
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read DOCX '{filename}': {exc}")
    if not text:
        raise HTTPException(status_code=422, detail=f"No text extracted from '{filename}'.")
    return text


def _extract_doc(file_bytes: bytes, filename: str) -> str:
    if not TEXTRACT_OK:
        raise HTTPException(status_code=501, detail="textract not installed. Run: pip install textract")
    try:
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        text = textract.process(tmp_path).decode("utf-8", errors="replace").strip()
        os.unlink(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not extract text from .doc '{filename}': {exc}")
    if not text:
        raise HTTPException(status_code=422, detail=f"No text extracted from '{filename}'.")
    return text


def _extract_image(file_bytes: bytes, filename: str) -> str:
    if not PIL_OK:
        raise HTTPException(status_code=501, detail="Pillow/pytesseract not installed.")
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text  = pytesseract.image_to_string(image, lang="eng").strip()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"OCR failed on '{filename}': {exc}")
    if not text:
        raise HTTPException(status_code=422, detail=f"No text found in '{filename}'.")
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
    stem = Path(filename).stem
    return stem.replace("_", " ").replace("-", " ").title()


# ══════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/usage/", tags=["Account"], summary="Check your monthly usage")
def get_usage(
    user: User    = Depends(get_rapidapi_user),
    db:   Session = Depends(get_db),
):
    """Returns how many requests you have used and how many remain this month."""
    used = _usage_this_month(user.id, db)
    return {
        "rapidapi_user":   user.email,
        "plan":            user.plan,
        "monthly_limit":   user.monthly_limit,
        "used_this_month": used,
        "remaining":       max(0, user.monthly_limit - used),
    }


@app.post(
    "/analyze-resume/",
    tags=["Resume Analysis"],
    summary="Upload one resume → full ATS report",
)
async def analyze_resume_endpoint(
    request:         Request,
    job_description: str           = Form(..., description="Paste the full job description text."),
    resume_file:     UploadFile    = File(..., description="Resume: PDF, DOCX, DOC, PNG, JPG, WEBP, or TXT."),
    candidate_name:  Optional[str] = Form(None, description="Optional candidate name. Defaults to filename."),
    user: User    = Depends(get_rapidapi_user),
    db:   Session = Depends(get_db),
):
    """
    Upload **one** resume + paste the job description → get a full ATS score report.

    ### Supported formats
    `PDF` · `DOCX` · `DOC` · `PNG` · `JPG` · `JPEG` · `WEBP` · `TXT`

    ### Scanned PDFs
    Automatically OCR'd if no text layer is found.
    """
    log_usage(user.id, db)
    raw_bytes = await resume_file.read()
    text      = extract_text(raw_bytes, resume_file.filename)
    name      = candidate_name or filename_to_name(resume_file.filename)
    return analyze_resume(resume_text=text, job_description=job_description, candidate_name=name)


@app.post(
    "/bulk-analyze/",
    tags=["Resume Analysis"],
    summary="Upload multiple resumes → ranked ATS candidate list",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["job_description", "resumes"],
                        "properties": {
                            "job_description": {
                                "type":        "string",
                                "description": "Paste the full job description text.",
                            },
                            "resumes": {
                                "type":        "array",
                                "items":       {"type": "string", "format": "binary"},
                                "description": "Multiple resume files — any mix of PDF/DOCX/DOC/PNG/JPG/TXT.",
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
    Returns candidates ranked by ATS score.

    ### Python example
    ```python
    import requests
    files = [
        ("resumes", ("alice.pdf",  open("alice.pdf",  "rb"), "application/pdf")),
        ("resumes", ("bob.docx",   open("bob.docx",   "rb"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
    ]
    r = requests.post(
        "https://YOUR-API.p.rapidapi.com/bulk-analyze/",
        files=files,
        data={"job_description": "We need a Python developer..."},
        headers={
            "X-RapidAPI-Key":  "YOUR_RAPIDAPI_KEY",
            "X-RapidAPI-Host": "YOUR-API.p.rapidapi.com",
        },
    )
    print(r.json())
    ```
    """
    try:
        form = await request.form()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse multipart form data.")

    job_description: str = form.get("job_description", "").strip()
    if not job_description:
        raise HTTPException(status_code=422, detail="Field 'job_description' is required.")

    resume_files: List[UploadFile] = form.getlist("resumes")
    resume_files = [f for f in resume_files if hasattr(f, "read")]
    if not resume_files:
        raise HTTPException(
            status_code=422,
            detail="No resume files received. Send files as multipart fields named 'resumes'.",
        )

    # Usage pre-flight
    used      = _usage_this_month(user.id, db)
    remaining = user.monthly_limit - used
    if len(resume_files) > remaining:
        raise HTTPException(
            status_code=429,
            detail=f"You have {remaining} requests left this month but uploaded {len(resume_files)} files.",
        )

    results: List[dict] = []
    errors:  List[dict] = []

    for resume_file in resume_files:
        fname = resume_file.filename or "unknown"
        try:
            raw_bytes = await resume_file.read()
            text      = extract_text(raw_bytes, fname)
            report    = analyze_resume(
                resume_text=text,
                job_description=job_description,
                candidate_name=filename_to_name(fname),
            )
            results.append(report)
            log_usage(user.id, db)
        except HTTPException as exc:
            errors.append({"file": fname, "error": exc.detail})

    if not results and errors:
        raise HTTPException(
            status_code=422,
            detail={"message": "All uploaded files failed to process.", "errors": errors},
        )

    ranked      = sorted(results, key=lambda r: r["ats_score"], reverse=True)
    shortlisted = [r for r in ranked if r["ats_score"] >= 70]

    return {
        "summary": {
            "total_uploaded":        len(resume_files),
            "successfully_analyzed": len(ranked),
            "failed":                len(errors),
            "shortlisted_ats_70":    len(shortlisted),
            "top_candidate":         ranked[0]["candidate_name"] if ranked else None,
            "top_ats_score":         ranked[0]["ats_score"]      if ranked else None,
        },
        "ranked_candidates": [
            {
                "rank":                   i + 1,
                "candidate_name":         r["candidate_name"],
                "ats_score":              r["ats_score"],
                "ats_rating":             r["ats_rating"],
                "skill_match_percentage": r["skill_match_percentage"],
                "matched_skills":         r["matched_skills"],
                "missing_skills":         r["missing_skills"],
                "keyword_density":        r["keyword_density"],
                "score_breakdown":        r["score_breakdown"],
                "sections_detected":      r["sections_detected"],
                "missing_sections":       r["missing_sections"],
                "power_verbs_found":      r["power_verbs_found"],
                "has_quantified_metrics": r["has_quantified_metrics"],
                "format_checks":          r["format_checks"],
                "recommendations":        r["recommendations"],
            }
            for i, r in enumerate(ranked)
        ],
        "failed_files": errors,
    }