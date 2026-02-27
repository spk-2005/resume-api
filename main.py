"""
main.py  ─  Resume ATS Intelligence API  v3.0.0
─────────────────────────────────────────────────
HOW RAPIDAPI AUTHENTICATION WORKS (when selling on marketplace):
─────────────────────────────────────────────────────────────────
  RapidAPI does NOT forward the user's X-RapidAPI-Key to your server.
  Instead, RapidAPI acts as a proxy and sends these headers to YOUR server:

    X-RapidAPI-Proxy-Secret  → a secret YOU set in RapidAPI dashboard
    X-RapidAPI-User          → the subscribing user's RapidAPI username
    X-RapidAPI-Subscription  → their subscription plan name

  Set this in your environment:
    RAPIDAPI_PROXY_SECRET=<copy from RapidAPI dashboard → My APIs → Security>
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
#  AUTH
# ══════════════════════════════════════════════════════════════════

def _usage_this_month(user_id: int, db: Session) -> int:
    return db.query(UsageLog).filter(
        UsageLog.user_id == user_id,
        func.extract("month", UsageLog.request_time) == datetime.now().month,
    ).count()


def get_rapidapi_user(request: Request, db: Session = Depends(get_db)) -> User:
    # Verify proxy secret
    proxy_secret = request.headers.get("x-rapidapi-proxy-secret", "")
    if RAPIDAPI_PROXY_SECRET and proxy_secret != RAPIDAPI_PROXY_SECRET:
        raise HTTPException(
            status_code=403,
            detail="Forbidden. This API must be called through RapidAPI.",
        )

    # Get RapidAPI username
    rapidapi_user = request.headers.get("x-rapidapi-user", "").strip()
    if not rapidapi_user:
        raise HTTPException(
            status_code=401,
            detail="Could not identify caller. Please call this API through RapidAPI.",
        )

    # Map plan to limits
    subscription  = request.headers.get("x-rapidapi-subscription", "BASIC").strip().upper()
    plan_limits   = {"BASIC": 50, "PRO": 500, "ULTRA": 2000, "MEGA": 10000, "CUSTOM": 99999}
    monthly_limit = plan_limits.get(subscription, 50)

    # Auto-create or update user
    user = db.query(User).filter(User.email == rapidapi_user).first()
    if not user:
        user = User(email=rapidapi_user, plan=subscription.lower(), monthly_limit=monthly_limit)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        if user.plan != subscription.lower() or user.monthly_limit != monthly_limit:
            user.plan          = subscription.lower()
            user.monthly_limit = monthly_limit
            db.commit()

    # Enforce limit
    if _usage_this_month(user.id, db) >= user.monthly_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly limit of {user.monthly_limit} requests reached. Upgrade on RapidAPI.",
        )

    return user


def log_usage(user_id: int, db: Session):
    db.add(UsageLog(user_id=user_id))
    db.commit()


# ══════════════════════════════════════════════════════════════════
#  TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════

def _extract_pdf(file_bytes: bytes, filename: str) -> str:
    if not PYPDF_OK:
        raise HTTPException(status_code=501, detail="pypdf not installed.")
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text   = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read '{filename}': {exc}")

    if text:
        return text

    if not OCR_OK:
        raise HTTPException(status_code=422, detail=f"'{filename}' is a scanned PDF. OCR libraries not installed.")
    try:
        images = convert_from_bytes(file_bytes, dpi=200)
        text   = "\n".join(pytesseract.image_to_string(img, lang="eng") for img in images).strip()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"OCR failed on '{filename}': {exc}")

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
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read DOCX '{filename}': {exc}")
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
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not extract text from '{filename}': {exc}")
    if not text:
        raise HTTPException(status_code=422, detail=f"No text extracted from '{filename}'.")
    return text


def _extract_image(file_bytes: bytes, filename: str) -> str:
    if not PIL_OK:
        raise HTTPException(status_code=501, detail="Pillow/pytesseract not installed.")
    try:
        text = pytesseract.image_to_string(Image.open(io.BytesIO(file_bytes)), lang="eng").strip()
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
    return Path(filename).stem.replace("_", " ").replace("-", " ").title()


def _collect_upload_files(form) -> List[UploadFile]:
    """
    Collect all uploaded files from a multipart form.

    RapidAPI's test console and different HTTP clients send files under
    different field names. We check ALL of these to be safe:
      • 'resumes'      → standard bulk field (list)
      • 'resume_file'  → some clients use singular name
      • 'file'         → generic fallback
      • anything else  → scan every field for UploadFile objects
    """
    found: List[UploadFile] = []
    seen_ids = set()

    def _add(f):
        if hasattr(f, "read") and id(f) not in seen_ids:
            seen_ids.add(id(f))
            found.append(f)

    # Check known field names (getlist handles both single and multiple)
    for field_name in ("resumes", "resume_file", "resume", "file", "files"):
        items = form.getlist(field_name)
        for item in items:
            _add(item)

    # Fallback: scan every field in the form for any UploadFile
    for key in form.keys():
        items = form.getlist(key)
        for item in items:
            _add(item)

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
    job_description: str           = Form(..., description="Paste the full job description text."),
    resume_file:     UploadFile    = File(..., description="Resume: PDF, DOCX, DOC, PNG, JPG, WEBP, or TXT."),
    candidate_name:  Optional[str] = Form(None, description="Optional candidate name. Defaults to filename."),
    user: User    = Depends(get_rapidapi_user),
    db:   Session = Depends(get_db),
):
    """
    Upload **one** resume + paste the job description → full ATS score report.

    **Supported formats:** PDF · DOCX · DOC · PNG · JPG · JPEG · WEBP · TXT

    Scanned PDFs are automatically OCR'd if no text layer is found.
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
)
async def bulk_analyze(
    request: Request,
    user: User    = Depends(get_rapidapi_user),
    db:   Session = Depends(get_db),
):
    """
    Upload **multiple resumes** against one job description.
    Returns all candidates ranked by ATS score (highest first).

    ### How to send files

    **cURL:**
    ```bash
    curl -X POST https://YOUR-API.p.rapidapi.com/bulk-analyze/ \\
      -H "X-RapidAPI-Key: YOUR_KEY" \\
      -H "X-RapidAPI-Host: YOUR-API.p.rapidapi.com" \\
      -F "job_description=We need a Python developer..." \\
      -F "resumes=@alice.pdf" \\
      -F "resumes=@bob.docx"
    ```

    **Python:**
    ```python
    import requests
    files = [
        ("resumes", ("alice.pdf", open("alice.pdf", "rb"), "application/pdf")),
        ("resumes", ("bob.docx",  open("bob.docx",  "rb"), "application/octet-stream")),
    ]
    r = requests.post(
        "https://YOUR-API.p.rapidapi.com/bulk-analyze/",
        headers={"X-RapidAPI-Key": "YOUR_KEY", "X-RapidAPI-Host": "YOUR-API.p.rapidapi.com"},
        data={"job_description": "We need a Python developer..."},
        files=files,
    )
    ```
    """
    # ── Parse form ────────────────────────────────────────────────
    try:
        form = await request.form()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse form data: {exc}")

    # ── Job description ───────────────────────────────────────────
    job_description: str = form.get("job_description", "").strip()
    if not job_description:
        raise HTTPException(status_code=422, detail="Field 'job_description' is required.")

    # ── Collect files (handles any field name RapidAPI might use) ─
    resume_files = _collect_upload_files(form)

    if not resume_files:
        # Return a helpful error showing what fields were actually received
        received_fields = {k: str(type(form.get(k))) for k in form.keys()}
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "No resume files found in the request.",
                "tip":     "Send files as multipart form fields named 'resumes'. See endpoint docs for examples.",
                "received_fields": received_fields,
            },
        )

    # ── Usage pre-flight ──────────────────────────────────────────
    used      = _usage_this_month(user.id, db)
    remaining = user.monthly_limit - used
    if len(resume_files) > remaining:
        raise HTTPException(
            status_code=429,
            detail=f"You have {remaining} requests left this month but sent {len(resume_files)} files.",
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
            log_usage(user.id, db)
        except HTTPException as exc:
            errors.append({"file": fname, "error": exc.detail})
        except Exception as exc:
            errors.append({"file": fname, "error": str(exc)})

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