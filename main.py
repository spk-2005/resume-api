"""
main.py  ─  Resume ATS Intelligence API  v3.0.0
─────────────────────────────────────────────────
Accepts resumes as any of:
  • PDF   (text-based or scanned image PDF)
  • DOCX / DOC  (Word documents, any number of pages)
  • PNG / JPG / JPEG / WEBP / GIF  (resume screenshots)
  • TXT   (plain text)

Multi-format extraction pipeline:
  ┌──────────────┬────────────────────────────────────────────┐
  │ Format       │ Extraction Method                          │
  ├──────────────┼────────────────────────────────────────────┤
  │ .txt         │ Read bytes directly                        │
  │ .pdf         │ pypdf  →  fallback: pdf2image + pytesseract│
  │ .docx        │ python-docx  (all paragraphs + tables)     │
  │ .doc         │ textract  (LibreOffice-based conversion)   │
  │ .png/.jpg etc│ pytesseract OCR                            │
  └──────────────┴────────────────────────────────────────────┘
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
    Header, UploadFile, File, Form, Request
)
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import SessionLocal, engine
from models import Base, User, APIKey, UsageLog
from services.resume_analyzer import analyze_resume


# ══════════════════════════════════════════════════════════════════
#  OPTIONAL DEPENDENCY IMPORTS  (graceful degradation)
# ══════════════════════════════════════════════════════════════════

# ── PDF: text-based ───────────────────────────────────────────────
try:
    from pypdf import PdfReader
    PYPDF_OK = True
except ImportError:
    PYPDF_OK = False

# ── PDF: scanned / image-only  →  OCR fallback ───────────────────
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_OK = True
except ImportError:
    OCR_OK = False

# ── DOCX ──────────────────────────────────────────────────────────
try:
    import docx  # python-docx
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

# ── DOC (old binary format) ───────────────────────────────────────
try:
    import textract
    TEXTRACT_OK = True
except ImportError:
    TEXTRACT_OK = False

# ── Images (PNG / JPG / WEBP …) ───────────────────────────────────
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
| **PDF** | Text-based PDFs parsed directly; scanned/image PDFs auto-OCR'd via Tesseract |
| **DOCX** | Word 2007+ documents, any number of pages |
| **DOC** | Legacy Word format via textract/LibreOffice |
| **PNG / JPG / JPEG / WEBP / GIF** | Resume screenshots via OCR |
| **TXT** | Plain text |

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/create-user/` | Register and get an API key |
| POST | `/analyze-resume/` | Single file → ATS report |
| POST | `/bulk-analyze/` | Multiple files → ranked list |
| GET  | `/usage/` | Check monthly usage |

### Authentication
Pass your API key as the **`X-Api-Key`** request header.
    """,
    version="3.0.0",
)
Base.metadata.create_all(bind=engine)


# ══════════════════════════════════════════════════════════════════
#  DB + AUTH HELPERS
# ══════════════════════════════════════════════════════════════════

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _usage_this_month(user_id: int, db: Session) -> int:
    return db.query(UsageLog).filter(
        UsageLog.user_id == user_id,
        func.extract("month", UsageLog.request_time) == datetime.now().month,
    ).count()


def get_authenticated_user(x_api_key: str, db: Session) -> User:
    key = db.query(APIKey).filter(APIKey.api_key == x_api_key).first()
    if not key:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    user = db.query(User).filter(User.id == key.user_id).first()
    if _usage_this_month(user.id, db) >= user.monthly_limit:
        raise HTTPException(
            status_code=403,
            detail=f"Monthly limit of {user.monthly_limit} requests reached. Upgrade your plan.",
        )
    return user


def log_usage(user_id: int, db: Session):
    db.add(UsageLog(user_id=user_id))
    db.commit()


# ══════════════════════════════════════════════════════════════════
#  MULTI-FORMAT TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════

def _extract_pdf(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from a PDF.
    Step 1: Try pypdf (fast, works on text-based PDFs).
    Step 2: If no text is found → OCR with pdf2image + pytesseract.
    """
    if not PYPDF_OK:
        raise HTTPException(
            status_code=501,
            detail="pypdf not installed. Run: pip install pypdf",
        )

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages  = [page.extract_text() or "" for page in reader.pages]
        text   = "\n".join(pages).strip()
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read '{filename}': {exc}",
        )

    # Text found — return it
    if text:
        return text

    # No text → scanned PDF, try OCR
    if not OCR_OK:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'{filename}' appears to be a scanned/image PDF with no text layer. "
                "Install pdf2image and pytesseract to enable OCR: "
                "`pip install pdf2image pytesseract` and install Tesseract."
            ),
        )

    try:
        images = convert_from_bytes(file_bytes, dpi=200)
        ocr_pages = [pytesseract.image_to_string(img, lang="eng") for img in images]
        text = "\n".join(ocr_pages).strip()
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"OCR failed on '{filename}': {exc}",
        )

    if not text:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No text could be extracted from '{filename}' even after OCR. "
                "Ensure the file is a readable PDF."
            ),
        )
    return text


def _extract_docx(file_bytes: bytes, filename: str) -> str:
    """Extract text from a .docx file (python-docx)."""
    if not DOCX_OK:
        raise HTTPException(
            status_code=501,
            detail="python-docx not installed. Run: pip install python-docx",
        )
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        parts: List[str] = []

        # Body paragraphs
        for para in document.paragraphs:
            parts.append(para.text)

        # Tables (skills grids, contact tables, etc.)
        for table in document.tables:
            for row in table.rows:
                for cell in row.cells:
                    parts.append(cell.text)

        # Headers & footers (sometimes contain contact info)
        for section in document.sections:
            for hdr_para in (section.header.paragraphs if section.header else []):
                parts.append(hdr_para.text)
            for ftr_para in (section.footer.paragraphs if section.footer else []):
                parts.append(ftr_para.text)

        text = "\n".join(p for p in parts if p.strip()).strip()
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read DOCX '{filename}': {exc}",
        )

    if not text:
        raise HTTPException(
            status_code=422,
            detail=f"No text extracted from '{filename}'. The document may be empty.",
        )
    return text


def _extract_doc(file_bytes: bytes, filename: str) -> str:
    """Extract text from a legacy .doc file via textract."""
    if not TEXTRACT_OK:
        raise HTTPException(
            status_code=501,
            detail=(
                "textract not installed. Run: pip install textract "
                "(also requires LibreOffice on the server for .doc support)."
            ),
        )
    try:
        # textract needs a file on disk
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        text = textract.process(tmp_path).decode("utf-8", errors="replace").strip()
        os.unlink(tmp_path)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract text from .doc '{filename}': {exc}",
        )

    if not text:
        raise HTTPException(
            status_code=422,
            detail=f"No text extracted from '{filename}'.",
        )
    return text


def _extract_image(file_bytes: bytes, filename: str) -> str:
    """Extract text from an image (PNG/JPG/WEBP…) via Tesseract OCR."""
    if not PIL_OK:
        raise HTTPException(
            status_code=501,
            detail=(
                "Pillow / pytesseract not installed. "
                "Run: pip install pillow pytesseract  (also install Tesseract binary)."
            ),
        )
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text  = pytesseract.image_to_string(image, lang="eng").strip()
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"OCR failed on image '{filename}': {exc}",
        )

    if not text:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No text found in image '{filename}'. "
                "Ensure the image is clear and the text is legible."
            ),
        )
    return text


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Master dispatcher: routes to the right extractor based on file extension.
    Raises HTTPException(422) if format is unsupported or extraction fails.
    """
    ext = Path(filename).suffix.lower()

    if ext in TEXT_EXTS:
        return file_bytes.decode("utf-8", errors="replace").strip()

    if ext in PDF_EXTS:
        return _extract_pdf(file_bytes, filename)

    if ext in DOCX_EXTS:
        return _extract_docx(file_bytes, filename)

    if ext in DOC_EXTS:
        return _extract_doc(file_bytes, filename)

    if ext in IMAGE_EXTS:
        return _extract_image(file_bytes, filename)

    raise HTTPException(
        status_code=415,
        detail=(
            f"Unsupported file type '{ext}'. "
            f"Accepted: {', '.join(sorted(ALL_SUPPORTED))}"
        ),
    )


def filename_to_name(filename: str) -> str:
    """'rahul_sharma_resume.pdf'  →  'Rahul Sharma Resume'"""
    stem = Path(filename).stem
    return stem.replace("_", " ").replace("-", " ").title()


# ══════════════════════════════════════════════════════════════════
#  AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/create-user/", tags=["Auth"], summary="Register → get API key")
def create_user(email: str, db: Session = Depends(get_db)):
    """
    Register with your email.  
    Returns an API key (basic plan = 50 requests / month).
    """
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Account already exists for this email.")

    user = User(email=email, plan="basic", monthly_limit=50)
    db.add(user)
    db.commit()
    db.refresh(user)

    api_key = str(uuid.uuid4())
    db.add(APIKey(user_id=user.id, api_key=api_key))
    db.commit()

    return {
        "message":       "Account created.",
        "email":         email,
        "plan":          "basic",
        "monthly_limit": 50,
        "api_key":       api_key,
        "tip":           "Send your api_key in the X-Api-Key header on every analysis request.",
    }


@app.get("/usage/", tags=["Auth"], summary="Check monthly usage")
def get_usage(x_api_key: str = Header(..., alias="X-RapidAPI-Key"), db: Session = Depends(get_db)):
    key = db.query(APIKey).filter(APIKey.api_key == x_api_key).first()
    if not key:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    user = db.query(User).filter(User.id == key.user_id).first()
    used = _usage_this_month(user.id, db)
    return {
        "email":           user.email,
        "plan":            user.plan,
        "monthly_limit":   user.monthly_limit,
        "used_this_month": used,
        "remaining":       max(0, user.monthly_limit - used),
    }


# ══════════════════════════════════════════════════════════════════
#  SINGLE FILE ANALYSIS  (replaces /analyze-resume-pdf/)
# ══════════════════════════════════════════════════════════════════

@app.post(
    "/analyze-resume/",
    tags=["Job Seeker"],
    summary="Upload one resume (PDF/DOCX/DOC/Image/TXT) → full ATS report",
)
async def analyze_resume_endpoint(
    job_description: str = Form(..., description="Paste the full job description text."),
    resume_file:     UploadFile = File(..., description="Resume file: PDF, DOCX, DOC, PNG, JPG, WEBP, or TXT."),
    candidate_name:  Optional[str] = Form(None, description="Optional. Defaults to filename."),
    x_api_key: str = Header(..., alias="X-RapidAPI-Key", description="API key from /create-user/"),
    db:              Session = Depends(get_db),
):
    """
    Upload **one** resume file + paste the job description.

    ### Supported formats
    `PDF` · `DOCX` · `DOC` · `PNG` · `JPG` · `JPEG` · `WEBP` · `TXT`

    ### Scanned / image PDFs
    Automatically OCR'd with Tesseract if no text layer is found.

    ### Tips
    - Paste the **full** JD text for the most accurate skill matching.
    - Multi-page documents are fully supported.
    """
    user       = get_authenticated_user(x_api_key, db)
    log_usage(user.id, db)

    raw_bytes  = await resume_file.read()
    text       = extract_text(raw_bytes, resume_file.filename)
    name       = candidate_name or filename_to_name(resume_file.filename)

    return analyze_resume(resume_text=text, job_description=job_description, candidate_name=name)


# ── Keep old endpoint alive for backwards compatibility ───────────
@app.post(
    "/analyze-resume-pdf/",
    tags=["Job Seeker"],
    summary="[Deprecated] Use /analyze-resume/ instead",
    include_in_schema=False,   # hides from Swagger docs
)
async def analyze_resume_pdf_compat(
    job_description: str = Form(...),
    resume_pdf:      UploadFile = File(...),
    candidate_name:  Optional[str] = Form(None),
    x_api_key: str = Header(..., alias="X-RapidAPI-Key"),
    db:              Session = Depends(get_db),
):
    user      = get_authenticated_user(x_api_key, db)
    log_usage(user.id, db)
    raw_bytes = await resume_pdf.read()
    text      = extract_text(raw_bytes, resume_pdf.filename)
    name      = candidate_name or filename_to_name(resume_pdf.filename)
    return analyze_resume(resume_text=text, job_description=job_description, candidate_name=name)


# ══════════════════════════════════════════════════════════════════
#  BULK ANALYSIS  (PDF / DOCX / Images / TXT — any mix)
# ══════════════════════════════════════════════════════════════════

@app.post(
    "/bulk-analyze/",
    tags=["Hiring Team"],
    summary="Upload multiple resumes (any format) → ranked ATS candidate list",
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
                                "description": "Upload multiple resume files (PDF/DOCX/DOC/PNG/JPG/TXT). Any mix of formats.",
                            },
                        },
                    }
                }
            },
        }
    },
)
async def bulk_analyze(
    request:   Request,
    x_api_key: str = Header(..., alias="X-RapidAPI-Key",description="API key from /create-user/"),
    db:        Session = Depends(get_db),
):
    """
    ## Bulk Resume Ranking — Hiring Team Mode

    Upload **multiple resumes** (any mix of PDF, DOCX, DOC, PNG, JPG, TXT)
    against one job description. Each file is analyzed individually,
    then candidates are ranked by ATS score.

    ### Sending files — 3 options

    **Option A — Swagger UI**  
    Click *Try it out*, fill `job_description`, then click **Add item** under
    `resumes` for each file.

    **Option B — cURL**
    ```bash
    curl -X POST http://localhost:8000/bulk-analyze/ \\
      -H "X-Api-Key: YOUR_KEY" \\
      -F "job_description=We need a Python developer..." \\
      -F "resumes=@rahul.pdf" \\
      -F "resumes=@anita.docx" \\
      -F "resumes=@vikram_resume.png"
    ```

    **Option C — Python requests**
    ```python
    import requests

    files = [
        ("resumes", ("rahul.pdf",   open("rahul.pdf",   "rb"), "application/pdf")),
        ("resumes", ("anita.docx",  open("anita.docx",  "rb"), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
        ("resumes", ("vikram.png",  open("vikram.png",  "rb"), "image/png")),
    ]
    r = requests.post(
        "http://localhost:8000/bulk-analyze/",
        files=files,
        data={"job_description": "We need a Python developer..."},
        headers={"X-Api-Key": "YOUR_KEY"},
    )
    print(r.json())
    ```

    ### Returns
    - `summary` — totals, shortlisted (≥70), top candidate
    - `ranked_candidates` — full ATS report per candidate, sorted highest → lowest
    - `failed_files` — any files that could not be parsed, with reasons
    """
    user = get_authenticated_user(x_api_key, db)

    # ── Parse raw multipart ────────────────────────────────────────
    try:
        form = await request.form()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not parse form data. Ensure Content-Type is multipart/form-data.",
        )

    job_description: str = form.get("job_description", "").strip()
    if not job_description:
        raise HTTPException(status_code=422, detail="Field 'job_description' is required.")

    resume_files: List[UploadFile] = form.getlist("resumes")
    resume_files = [f for f in resume_files if hasattr(f, "read")]
    if not resume_files:
        raise HTTPException(
            status_code=422,
            detail=(
                "No resume files received. "
                "Send files as multipart fields all named 'resumes'. "
                "Supported: PDF, DOCX, DOC, PNG, JPG, WEBP, TXT."
            ),
        )

    # ── Usage pre-flight ───────────────────────────────────────────
    used      = _usage_this_month(user.id, db)
    remaining = user.monthly_limit - used
    if len(resume_files) > remaining:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Upload has {len(resume_files)} files but you only have "
                f"{remaining} requests remaining this month."
            ),
        )

    # ── Process each file ─────────────────────────────────────────
    results: List[dict] = []
    errors:  List[dict] = []

    for resume_file in resume_files:
        fname = resume_file.filename or "unknown"
        candidate_name = filename_to_name(fname)
        try:
            raw_bytes = await resume_file.read()
            text      = extract_text(raw_bytes, fname)
            report    = analyze_resume(
                resume_text=text,
                job_description=job_description,
                candidate_name=candidate_name,
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

    # ── Rank ──────────────────────────────────────────────────────
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