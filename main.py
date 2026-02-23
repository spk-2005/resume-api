import io
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import (
    FastAPI, Depends, HTTPException,
    Header, UploadFile, File, Form, Request
)
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import SessionLocal
from models import User, APIKey, UsageLog
from services.resume_analyzer import analyze_resume


# ── PDF parser ────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


# ══════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Resume ATS Intelligence API",
    description="""
## Resume ATS Intelligence API

Analyze resumes (PDF) against job descriptions with a full ATS score.

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/create-user/` | Register and get an API key |
| POST | `/analyze-resume-pdf/` | Single PDF → ATS report |
| POST | `/bulk-analyze/` | Multiple PDFs → ranked list |
| GET  | `/usage/` | Check monthly usage |

### Authentication
Pass your API key as the **`X-Api-Key`** request header on every analysis call.

### ATS Report Fields
| Field | Description |
|---|---|
| `ats_score` | 0–100 overall score |
| `ats_rating` | Excellent / Good / Average / Below Average / Poor |
| `score_breakdown` | Points earned per component |
| `skill_match_percentage` | % of JD skills found in resume |
| `matched_skills` | Skills in both resume and JD |
| `missing_skills` | Skills JD requires but resume lacks |
| `keyword_density` | How many times each required skill appears |
| `sections_detected` | Which standard sections the resume has |
| `power_verbs_found` | Action verbs detected |
| `has_quantified_metrics` | Whether resume has numbers/% |
| `format_checks` | Email, phone, LinkedIn, GitHub, word count |
| `recommendations` | Prioritised fix list (HIGH → MEDIUM → LOW) |
    """,
    version="2.0.0",
)


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
#  PDF HELPERS
# ══════════════════════════════════════════════════════════════════

def extract_pdf_text(file_bytes: bytes, filename: str = "resume.pdf") -> str:
    """
    Extract plain text from PDF bytes.
    - 501  →  pypdf not installed
    - 422  →  corrupt PDF or scanned image (no text layer)
    """
    if not PDF_SUPPORT:
        raise HTTPException(
            status_code=501,
            detail="PDF support not installed. Run: pip install pypdf",
        )
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read '{filename}'. Ensure it is a valid PDF. Error: {exc}",
        )
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                f"No text extracted from '{filename}'. "
                "This is likely a scanned/image PDF. "
                "Export your resume from Word or Google Docs as a PDF instead."
            ),
        )
    return text


def filename_to_name(filename: str) -> str:
    """'rahul_sharma_resume.pdf'  →  'Rahul Sharma Resume'"""
    return (
        filename
        .replace(".pdf", "")
        .replace("_", " ")
        .replace("-", " ")
        .title()
    )


# ══════════════════════════════════════════════════════════════════
#  AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/create-user/", tags=["Auth"], summary="Register → get API key")
def create_user(email: str, db: Session = Depends(get_db)):
    """
    Register with your email. Returns an API key (basic plan = 50 req/month).
    Use this key in the `X-Api-Key` header for all analysis endpoints.
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
def get_usage(x_api_key: str = Header(...), db: Session = Depends(get_db)):
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
#  SINGLE PDF ANALYSIS
# ══════════════════════════════════════════════════════════════════

@app.post(
    "/analyze-resume-pdf/",
    tags=["Job Seeker"],
    summary="Upload one PDF resume → full ATS report",
)
async def analyze_resume_pdf(
    job_description: str = Form(..., description="Paste the full job description text."),
    resume_pdf: UploadFile = File(..., description="Your resume as a text-based PDF."),
    candidate_name: Optional[str] = Form(None, description="Optional name. Defaults to filename."),
    x_api_key: str = Header(..., description="API key from /create-user/"),
    db: Session = Depends(get_db),
):
    """
    Upload **one** PDF resume + paste the job description.
    Returns the complete ATS analysis report.

    **Tips:**
    - Export your resume from Word / Google Docs as PDF (not a scan).
    - Paste the **full** JD text for the most accurate skill matching.
    """
    user = get_authenticated_user(x_api_key, db)
    log_usage(user.id, db)

    pdf_bytes = await resume_pdf.read()
    text      = extract_pdf_text(pdf_bytes, resume_pdf.filename)
    name      = candidate_name or filename_to_name(resume_pdf.filename)

    return analyze_resume(resume_text=text, job_description=job_description, candidate_name=name)


# ══════════════════════════════════════════════════════════════════
#  BULK PDF ANALYSIS  ← THE FIXED ENDPOINT
# ══════════════════════════════════════════════════════════════════
#
#  ROOT CAUSE OF THE ORIGINAL BUG
#  ───────────────────────────────
#  FastAPI's  List[UploadFile] = File(...)  sends the correct Python
#  type, but it emits an OpenAPI schema where the field type is
#  "string/binary" (single file) — Swagger UI therefore renders a
#  single-file picker and clients only send one file.
#
#  THE FIX
#  ───────
#  We override the auto-generated OpenAPI schema for this endpoint
#  via  openapi_extra  so the field is described as:
#       type: array, items: { type: string, format: binary }
#  This tells Swagger UI to render the "Add item" multi-file picker.
#  The FastAPI handler still receives  List[UploadFile]  correctly
#  because the actual multipart parsing is unchanged — only the
#  schema description changes.
#
# ══════════════════════════════════════════════════════════════════

@app.post(
    "/bulk-analyze/",
    tags=["Hiring Team"],
    summary="Upload multiple PDF resumes → ranked ATS candidate list",
    # ↓ This overrides the OpenAPI schema for this specific endpoint.
    # It fixes the Swagger UI to show a proper multi-file upload button.
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
                                # KEY FIX: array of files, not a single file
                                "type":  "array",
                                "items": {"type": "string", "format": "binary"},
                                "description": "Upload multiple PDF resume files.",
                            },
                        },
                    }
                }
            },
        }
    },
)
async def bulk_analyze(
    request: Request,                                          # raw request for manual parsing
    x_api_key: str = Header(..., description="API key from /create-user/"),
    db: Session = Depends(get_db),
):
    """
    ## Bulk Resume Ranking — Hiring Team Mode

    Upload **multiple PDF resumes** at once against one job description.
    Every file is analyzed individually, then candidates are ranked by ATS score.

    ### How to send files (3 options)

    **Option A — Swagger UI**
    Click "Try it out", fill in `job_description`, then click **"Add item"**
    under `resumes` for each PDF you want to upload.

    **Option B — cURL**
    ```bash
    curl -X POST http://localhost:8000/bulk-analyze/ \\
      -H "X-Api-Key: YOUR_KEY" \\
      -F "job_description=We need a Python developer..." \\
      -F "resumes=@rahul.pdf" \\
      -F "resumes=@anita.pdf" \\
      -F "resumes=@vikram.pdf"
    ```

    **Option C — Python requests**
    ```python
    import requests
    files = [
        ("resumes", ("rahul.pdf",  open("rahul.pdf",  "rb"), "application/pdf")),
        ("resumes", ("anita.pdf",  open("anita.pdf",  "rb"), "application/pdf")),
        ("resumes", ("vikram.pdf", open("vikram.pdf", "rb"), "application/pdf")),
    ]
    data   = {"job_description": "We need a Python developer..."}
    headers= {"X-Api-Key": "YOUR_KEY"}
    r = requests.post("http://localhost:8000/bulk-analyze/", files=files, data=data, headers=headers)
    print(r.json())
    ```

    ### Returns
    - `summary` — quick stats: total, analyzed, failed, shortlisted (≥70)
    - `ranked_candidates` — full ATS report per candidate, sorted highest → lowest
    - `failed_files` — any PDFs that could not be parsed
    """
    user = get_authenticated_user(x_api_key, db)

    # ── Parse raw multipart form manually ─────────────────────────
    # We do this because FastAPI's automatic List[UploadFile] injection
    # works at runtime but breaks Swagger's schema (as explained above).
    # Using request.form() gives us full control over both.
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

    # Collect all uploaded files — getlist handles multiple values with same key
    resume_files: List[UploadFile] = form.getlist("resumes")
    if not resume_files:
        raise HTTPException(
            status_code=422,
            detail=(
                "No resume files received. "
                "Send files as multipart form fields all named 'resumes'. "
                "See the endpoint description for cURL / Python examples."
            ),
        )

    # Filter out any non-UploadFile entries (e.g. stray string values)
    resume_files = [f for f in resume_files if hasattr(f, "read")]
    if not resume_files:
        raise HTTPException(
            status_code=422,
            detail="Files were not received as binary uploads. Check your request format.",
        )

    # ── Pre-flight usage limit check ──────────────────────────────
    used      = _usage_this_month(user.id, db)
    remaining = user.monthly_limit - used
    if len(resume_files) > remaining:
        raise HTTPException(
            status_code=403,
            detail=(
                f"This upload has {len(resume_files)} files but you only have "
                f"{remaining} requests remaining this month."
            ),
        )

    # ── Analyze each file ─────────────────────────────────────────
    results: List[dict] = []
    errors:  List[dict] = []

    for resume_file in resume_files:
        candidate_name = filename_to_name(resume_file.filename or "unknown.pdf")
        try:
            pdf_bytes = await resume_file.read()
            text      = extract_pdf_text(pdf_bytes, resume_file.filename)
            report    = analyze_resume(
                resume_text=text,
                job_description=job_description,
                candidate_name=candidate_name,
            )
            results.append(report)
            log_usage(user.id, db)
        except HTTPException as exc:
            # File failed — record error, don't charge quota, continue with others
            errors.append({"file": resume_file.filename, "error": exc.detail})

    if not results and errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "All uploaded files failed to process.",
                "errors":  errors,
            },
        )

    # ── Rank and return ───────────────────────────────────────────
    ranked      = sorted(results, key=lambda r: r["ats_score"], reverse=True)
    shortlisted = [r for r in ranked if r["ats_score"] >= 70]

    return {
        # Quick summary for HR dashboard display
        "summary": {
            "total_uploaded":        len(resume_files),
            "successfully_analyzed": len(ranked),
            "failed":                len(errors),
            "shortlisted_ats_70":    len(shortlisted),
            "top_candidate":         ranked[0]["candidate_name"] if ranked else None,
            "top_ats_score":         ranked[0]["ats_score"]      if ranked else None,
        },

        # Full ATS report per candidate, ranked 1st to last
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

        # Files that failed parsing (scanned PDFs, corrupt files, etc.)
        "failed_files": errors,
    }
