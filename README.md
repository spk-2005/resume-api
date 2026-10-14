# Resume ATS Intelligence API (v3.0.0)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=Python&logoColor=white)](https://www.python.org/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0+-D71F00.svg?style=flat&logo=SQLAlchemy&logoColor=white)](https://www.sqlalchemy.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supported-336791.svg?style=flat&logo=PostgreSQL&logoColor=white)](https://www.postgresql.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Local_Zero_Config-003B57.svg?style=flat&logo=SQLite&logoColor=white)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-grade, high-performance Applicant Tracking System (ATS) Intelligence API.**  
> Evaluates candidate resumes against job descriptions with enterprise-grade accuracy using custom algorithmic NLP (BM25 Okapi, TF-IDF, N-grams, Lemmatization, Positional Weighting, and Negation-aware Skill Extraction) — **zero heavy machine-learning model overhead, ultra-fast cold starts, and minimal memory usage.**

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & System Flow](#architecture--system-flow)
- [ATS Scoring Methodology](#ats-scoring-methodology)
- [NLP & Extraction Engine](#nlp--extraction-engine)
- [Project Directory Structure](#project-directory-structure)
- [API Reference](#api-reference)
  - [Health & System Checks](#health--system-checks)
  - [Authentication & Users](#authentication--users)
  - [Resume Analysis (`/analyze-resume/`)](#single-resume-analysis-post-analyze-resume)
  - [Bulk Resume Screening (`/bulk-analyze/`)](#bulk-resume-screening-post-bulk-analyze)
- [Database Schema & Multi-DB Architecture](#database-schema--multi-db-architecture)
- [Environment Configuration](#environment-configuration)
- [Installation & Local Setup](#installation--local-setup)
- [Running the Application](#running-the-application)
- [Testing & Verification](#testing--verification)
- [Deployment (Render / Cloud)](#deployment-render--cloud)
- [License](#license)

---

## Overview

Modern Applicant Tracking Systems filter out up to 75% of qualified resumes due to formatting issues, missing keywords, unoptimized section structures, and misaligned job titles. 

The **Resume ATS Intelligence API** provides a complete backend solution for job boards, recruitment platforms, career coaches, and candidate-facing applications. It extracts deep semantic signals from resumes and matches them against target Job Descriptions (JDs), returning an evidence-backed score (0–100), section-by-section audit, keyword gap analysis, writing quality assessment, and actionable AI insights.

---

## Key Features

- **Blazing Fast Algorithmic NLP**: Built entirely from algorithmic principles (BM25 Okapi, TF-IDF, Cosine Similarity, Lemmatization, Negation Detection) without needing gigabyte-sized transformer weights or spaCy pipelines. Cold-starts in seconds with sub-second analysis latency.
- **Multi-Format Document Parsing**:
  - **PDF**: Native text extraction via `pypdf`, with automatic fallback to OCR via `pdf2image` + `pytesseract` for scanned documents.
  - **Word**: Microsoft Word `.docx` (including tables, headers, and footers via `python-docx`) and legacy `.doc` (via `textract`).
  - **Images**: Scanned resumes in `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff` via Tesseract OCR.
  - **Plain Text / Markup**: `.txt`, `.md`, `.rtf`, `.csv`, `.html`, `.htm`.
- **Flexible Job Description Ingestion**:
  - Direct raw text payload.
  - Job description file upload (`.pdf`, `.docx`, etc.).
  - **Live URL Web Scraping**: Automatically fetches and extracts clean text from public job posting URLs (stripping scripts, headers, navigation, and boilerplate).
- **Master Skill Database (1,000+ Taxonomy Tokens)**:
  - Deep coverage across modern tech stacks: Generative AI, LLMs, LangChain, RAG, Vector DBs, PyTorch, Cloud/DevOps (AWS, GCP, Kubernetes), Full-Stack, Mobile, and Data Engineering.
  - Context-aware boosting (skills near *"required"*, *"must have"*, *"mandatory"* receive amplified weights).
  - Negation detection (filters out false positives such as *"no experience with X"* or *"not required"*).
- **Detailed 100-Point ATS Scoring Formula**: Evaluates 6 distinct dimensions plus a job-title alignment multiplier.
- **Bulk Resume Screening & Ranking**: Submit dozens of candidate resumes in a single request against a target JD. Automatically ranks candidates by ATS score, identifies top performers, and filters those meeting the shortlisting benchmark (score $\ge$ 70).
- **User Authentication & Quota Management**:
  - JWT Bearer token authentication (`bcrypt` password hashing).
  - Subscription tier tracking (`basic`, `pro`, `ultra`, `mega`).
  - Monthly per-user usage limits and atomic request logging.
- **Dual Database Flexibility**:
  - **SQLite**: Zero-configuration local database for development and rapid testing.
  - **PostgreSQL**: Production-ready connection pooling, automated `sslmode=require`, and pre-ping liveness checks (optimized for Render, AWS RDS, Supabase, Neon).

---

## Architecture & System Flow

```mermaid
flowchart TD
    A[Client Request: Resume + JD] --> B{Input Handler}
    B -->|File/Image/PDF/Word| C[Multi-Format Extractor\npypdf / python-docx / OCR]
    B -->|JD URL Scraping| D[Web Scraper & HTML Cleaner]
    B -->|Plain Text| E[Raw Text Normalizer]

    C --> F[NLP & Parsing Engine]
    D --> F
    E --> F

    subgraph "Core NLP Processing (services/)"
        F --> G[Section Boundary Classifier\nresume_parser.py]
        F --> H[1000+ Skill Extractor & Negation Filter\nskill_extractor.py]
        F --> I[BM25 Okapi & TF-IDF Vectorizer\nnlp_core.py]
        F --> J[Experience & Timeline Extractor]
        F --> K[Writing Quality & Action Verb Analyzer]
    end

    G --> L[Multi-Factor ATS Scorer\nresume_analyzer.py]
    H --> L
    I --> L
    J --> L
    K --> L

    L --> M[AI Insights Generator\nStrengths, Gaps, Quick Wins]
    M --> N[Usage Quota Logging & DB Persistence]
    N --> O[JSON API Response]
```

---

## ATS Scoring Methodology

The overall ATS Score (0–100) is calculated using an objective, multi-factor rubric:

| Scoring Dimension | Max Weight | Description & Methodology |
| :--- | :---: | :--- |
| **Skill Match** | **38 pts** | Weighted match of required and preferred skills. Accounts for frequency in JD, proximity to requirement keywords (*"must have"*, *"essential"*), and presence in dedicated skills sections (2× weight boost). |
| **JD Semantic Similarity** | **17 pts** | BM25 Okapi term relevance combined with TF-IDF cosine similarity between resume text and job description. |
| **Section Completeness** | **15 pts** | Structural integrity check across 8 key sections: Contact, Summary, Work Experience, Education, Skills, Projects, Certifications, and Achievements. |
| **Experience Match** | **12 pts** | Parses required years of experience from the JD vs. cumulative years detected in the candidate's work timeline. |
| **Format & Contact Info** | **10 pts** | Validates email, telephone, LinkedIn, GitHub, portfolio URLs, word count, and length appropriateness. |
| **Writing Quality** | **5 pts** | Density of strong action/power verbs (*"Architected"*, *"Spearheaded"*, *"Optimized"*) and quantified metrics/achievements (%, $, user scale) minus penalties for weak/passive verbs. |
| **Title Alignment Multiplier** | **×0.85 – 1.00** | Cross-references target job title in the JD against the candidate's current or past job titles. |

### ATS Score Classification

- **88 – 100**: 🏆 Outstanding (Top-tier shortlist recommendation)
- **78 – 87**: ✅ Excellent (Strong candidate alignment)
- **65 – 77**: 🟢 Good (Meets core requirements; minor gaps)
- **50 – 64**: 🟡 Average (Noticeable skill or experience gaps)
- **35 – 49**: 🟠 Below Average (Significant missing criteria)
- **0 – 34**: 🔴 Poor (Low alignment / format deficiencies)

---

## NLP & Extraction Engine

The NLP subsystem in `services/` contains zero heavyweight external dependencies (no PyTorch, TensorFlow, or spaCy runtime overhead):

1. **`nlp_core.py`**:
   - **BM25 Okapi Ranking**: Optimized for short-document matching ($k_1 = 1.5, b = 0.75$).
   - **TF-IDF & Cosine Similarity**: Custom vector space modeling.
   - **Rule-Based English Lemmatization**: Suffix reduction without dictionary loading overhead.
   - **N-Gram Generator**: Unigram, bigram, and trigram extraction for compound technical terms (*"natural language processing"*, *"retrieval augmented generation"*).
   - **Synonym Normalization**: Standardizes industry acronyms (e.g., `k8s` $\rightarrow$ `kubernetes`, `reactjs` $\rightarrow$ `react`, `golang` $\rightarrow$ `go`, `ts` $\rightarrow$ `typescript`).

2. **`skill_extractor.py`**:
   - **Master Skill Database**: 1,000+ technical tokens mapped across categories: AI/ML, Backend, Frontend, Cloud/DevOps, Mobile, Databases, Testing, Security, and Methodologies.
   - **Negation Detection**: Uses lookbehind windows to discount phrases like *"no hands-on experience in AWS"* or *"not familiar with Docker"*.
   - **Contextual Requirement Boosting**: Increases weight for skills found adjacent to terms like *"required"*, *"mandatory"*, *"must have"*, *"minimum 3 years"*.

3. **`resume_parser.py`**:
   - **Section Splitting**: Pattern-matching regex boundaries recognizing 100+ heading variations.
   - **Contact Extraction**: Email regex, phone normalization (E.164 and international formats), social profile detection (LinkedIn, GitHub, LeetCode, Behance, Portfolios).
   - **Date & Timeline Analysis**: Analyzes employment durations, handling *"Present"*, date ranges (*"Jun 2021 - Present"*), and calculates career years.
   - **GPA & Degree Recognition**: Detects degree levels (Bachelor's, Master's, Ph.D., Associate) and GPA/CGPA formats (`3.8/4.0`, `9.5/10`).

---

## Project Directory Structure

```text
resume-api/
├── main.py                     # FastAPI application setup, middleware, routing & error handlers
├── database.py                 # SQLAlchemy engine, session factory, PostgreSQL/SQLite config
├── models.py                   # ORM database models (User, UsageLog, Job, Candidate, Report)
├── schemas.py                  # Pydantic schemas for request/response serialization
├── requirements.txt            # Python production dependencies
├── package.json                # Project metadata & npm script definitions
├── .env.example                # Template for environment variables
├── resume_api.db               # SQLite database file (for local use)
│
├── services/                   # Core business logic & NLP microservices
│   ├── __init__.py
│   ├── auth.py                 # JWT authentication, token hashing, and route guards
│   ├── nlp_core.py             # Algorithmic NLP: BM25, TF-IDF, Cosine Similarity, Lemmatizer
│   ├── resume_analyzer.py      # Master analyzer orchestrator & 100-point ATS scorer
│   ├── resume_parser.py        # Section splitting, contact parsing, timeline & writing quality
│   └── skill_extractor.py      # 1,000+ skill taxonomy, negation detection, weight calculator
│
└── scripts/                    # Verification and utility scripts
    ├── test_db_connection.py   # Verifies active DB connection (PostgreSQL/SQLite)
    └── verify_accuracy.py      # Unit test verifying contact, GPA, date, and skill extraction
```

---

## API Reference

Interactive API documentation is automatically generated and accessible when the server is running:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Health & System Checks

#### `GET /health` or `GET /`
Returns the operational health of the API and database connection. (Required for Render / RapidAPI health pingers).

**Response (200 OK):**
```json
{
  "status": "ok",
  "service": "Resume ATS Intelligence API",
  "version": "3.0.0",
  "database": "postgresql (dpg-xxxx.render.com)"
}
```

---

### Authentication & Users

#### `POST /auth/register`
Creates a new user account.

**Request Body:**
```json
{
  "email": "developer@example.com",
  "password": "StrongPassword123!"
}
```

**Response (200 OK):**
```json
{
  "id": 1,
  "email": "developer@example.com",
  "plan": "basic",
  "monthly_limit": 50
}
```

#### `POST /auth/login`
Authenticates user credentials and returns a JWT bearer access token (`OAuth2PasswordRequestForm`).

**Form Data:**
- `username`: developer@example.com
- `password`: StrongPassword123!

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

#### `GET /users/me`
Retrieves current authenticated user details, plan tier, and remaining monthly quota.

**Headers:**
`Authorization: Bearer <access_token>`

---

### Single Resume Analysis (`POST /analyze-resume/`)

Analyzes a single resume against a target job description and returns the full ATS audit.

- **URL**: `/analyze-resume/`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Headers**: `Authorization: Bearer <access_token>`

#### Form Parameters:

| Field | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `resume_file` | File | **Yes** | The candidate's resume (`.pdf`, `.docx`, `.doc`, `.txt`, `.png`, etc.) |
| `job_description` | Text | Optional* | The raw text of the job description or a URL pointing to the job post |
| `jd_file` | File | Optional* | Uploaded file containing the job description |
| `jd_url` | Text | Optional* | Public URL to scrape the job description from |
| `candidate_name` | Text | Optional | Candidate name (defaults to parsed name or filename) |

*\* At least one job description source (`job_description`, `jd_file`, or `jd_url`) must be provided.*

#### Example `curl` Request:

```bash
curl -X POST "http://localhost:8000/analyze-resume/" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "resume_file=@/path/to/resume.pdf" \
  -F "job_description=We are seeking a Senior Python Engineer experienced in FastAPI, PostgreSQL, Docker, and LangChain."
```

#### Sample Response:

```json
{
  "candidate": {
    "name": "Alex Mercer",
    "email": "alex.mercer@example.com",
    "phone": "+1 (555) 019-2834",
    "links": {
      "linkedin": "https://linkedin.com/in/alexmercer",
      "github": "https://github.com/alexmercer",
      "portfolio": "https://alexmercer.dev"
    }
  },
  "scores": {
    "ats_score": {
      "value": 84.5,
      "grade": "Excellent",
      "badge": "✅ Excellent"
    },
    "breakdown": {
      "skill_match": "33.2/38",
      "semantic_similarity": "14.8/17",
      "section_completeness": "15.0/15",
      "experience_match": "12.0/12",
      "format_and_contact": "10.0/10",
      "writing_quality": "4.2/5",
      "title_alignment_multiplier": 0.95
    }
  },
  "skill_analysis": {
    "overall_match_percentage": 87.5,
    "matched_skills": ["python", "fastapi", "postgresql", "docker", "git", "rest api"],
    "missing_skills": ["langchain"],
    "categorized_skills": {
      "Languages": ["python", "sql"],
      "Frameworks & Backend": ["fastapi", "sqlalchemy"],
      "DevOps & Cloud": ["docker"]
    }
  },
  "experience": {
    "required_by_jd": "5 years",
    "detected_in_resume": "~6.0 years",
    "match_score": "100%",
    "match_status": "Exceeds Requirement ✅"
  },
  "writing_quality": {
    "power_verbs": {
      "count": 14,
      "examples": ["architected", "streamlined", "optimized", "spearheaded"]
    },
    "quantified_achievements": {
      "count": 6,
      "examples": ["reduced latency by 42%", "scaled to 2M+ active users"]
    },
    "weak_verbs": {
      "count": 1,
      "found": ["assisted"]
    }
  },
  "ai_insights": {
    "strengths": [
      "Strong skill match — 87.5% of required skills present",
      "Experience meets or exceeds JD requirements",
      "Strong action verbs — 14 power verbs found",
      "Good quantified impact — 6 metrics found"
    ],
    "gaps": [
      "Missing 1 JD skill: langchain"
    ],
    "quick_wins": [
      "💡 Priority: Add 'langchain' to your resume if you have experience with it.",
      "⚡ Replace passive language like 'assisted' with authoritative power verbs."
    ],
    "summary": "ATS Score: 84.5/100 (Excellent). JD Match: 87.5/100. Skill match: 87.5% (6 matched, 1 missing). 8/8 sections present."
  },
  "usage": {
    "requests_used": 12,
    "requests_limit": 50,
    "requests_remaining": 38,
    "plan": "BASIC"
  }
}
```

---

### Bulk Resume Screening (`POST /bulk-analyze/`)

Process and rank multiple resumes against a job description in a single operation.

- **URL**: `/bulk-analyze/`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Headers**: `Authorization: Bearer <access_token>`

#### Form Parameters:
- `job_description` (or `jd_file` / `jd_url`): Job description text, file, or URL.
- `resumes`: Multiple file uploads (`resumes` multi-part array).

#### Sample Response:

```json
{
  "summary": {
    "total_uploaded": 15,
    "successfully_analyzed": 15,
    "failed": 0,
    "shortlisted_above_70": 4,
    "top_candidate": "Jane Doe",
    "top_ats_score": 92.4
  },
  "usage": {
    "requests_used": 27,
    "requests_limit": 500,
    "requests_remaining": 473,
    "plan": "PRO"
  },
  "ranked_candidates": [
    {
      "rank": 1,
      "candidate": { "name": "Jane Doe", "email": "jane@example.com" },
      "scores": { "ats_score": { "value": 92.4, "badge": "🏆 Outstanding" } },
      "skill_analysis": { "overall_match_percentage": 94.0 }
    },
    {
      "rank": 2,
      "candidate": { "name": "John Smith", "email": "john@example.com" },
      "scores": { "ats_score": { "value": 78.1, "badge": "✅ Excellent" } },
      "skill_analysis": { "overall_match_percentage": 81.5 }
    }
  ],
  "failed_files": []
}
```

---

## Database Schema & Multi-DB Architecture

The API uses **SQLAlchemy 2.0** with complete support for both SQLite and PostgreSQL.

```mermaid
erDiagram
    USERS ||--o{ USAGE_LOGS : "logs requests"
    USERS ||--o{ JOBS : "creates"
    USERS ||--o{ CANDIDATES : "creates"
    JOBS ||--o{ ANALYSIS_REPORTS : "evaluates"
    CANDIDATES ||--o{ ANALYSIS_REPORTS : "evaluated in"

    USERS {
        int id PK
        string email UK
        string hashed_password
        string plan "basic | pro | ultra | mega"
        int monthly_limit
        boolean is_active
        datetime created_at
    }

    USAGE_LOGS {
        int id PK
        int user_id FK
        datetime request_time
        string endpoint
        string status
    }

    JOBS {
        int id PK
        int user_id FK
        text raw_text
        json structured_requirements
        datetime created_at
    }

    CANDIDATES {
        int id PK
        int user_id FK
        text raw_text
        json structured_evidence
        datetime created_at
    }

    ANALYSIS_REPORTS {
        int id PK
        int job_id FK
        int candidate_id FK
        json report_data
        datetime created_at
    }
```

### Database Features:
- **Automatic SSL Enforcement**: When connecting to PostgreSQL, `sslmode=require` is injected automatically if absent.
- **Connection Pooling**: Configured with `pool_size=5`, `max_overflow=10`, `pool_recycle=1800`, and `pool_pre_ping=True` to eliminate stale connections across serverless and cloud hosting restarts.
- **Dynamic Dialect Translation**: Automatically rewrites legacy `postgres://` prefixes to `postgresql://` as mandated by SQLAlchemy 2.0.

---

## Environment Configuration

Create a `.env` file in the root directory by copying `.env.example`:

```bash
cp .env.example .env
```

### Available Configuration Keys:

| Variable | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `USE_LOCAL_DB` | Boolean | `false` | Set to `true` to force SQLite (`sqlite:///./resume_api.db`). |
| `DATABASE_URL` | String | *None* | PostgreSQL connection string (`postgresql://USER:PASSWORD@HOST/DB`). |
| `SECRET_KEY` | String | *Required* | Secret key for signing JWT tokens. Generate with: `openssl rand -hex 32` |
| `CORS_ORIGINS` | String | `http://localhost:3000,http://localhost:5173` | Comma-separated list of allowed CORS origin URLs. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Integer | `10080` (7 days) | Expiration time for JWT access tokens in minutes. |

---

## Installation & Local Setup

### 1. Prerequisites
- **Python 3.10+**
- **pip** (Python package installer)
- *(Optional for image OCR & scanned PDFs)*:
  - **Tesseract OCR**: [Install Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html)
  - **Poppler**: Required by `pdf2image` for scanned PDFs.

### 2. Clone the Repository
```bash
git clone https://github.com/spk-2005/resume-api.git
cd resume-api
```

### 3. Create and Activate a Virtual Environment
```bash
# On macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Application

### Development Server (with hot reloading)

Using Uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or using the npm helper script defined in `package.json`:
```bash
npm start
```

Once started, open:
- API Root: `http://localhost:8000/`
- Interactive API Docs: `http://localhost:8000/docs`
- ReDoc Docs: `http://localhost:8000/redoc`

---

## Testing & Verification

The project includes built-in verification scripts located in `scripts/`:

### 1. Verify Database Connection
Tests the database connectivity based on `.env` configuration (SQLite or PostgreSQL):
```bash
python scripts/test_db_connection.py
```

### 2. Verify NLP & Parser Accuracy
Runs an automated regression test covering contact extraction, GPA formats, timeline dates, and modern skill detection (LangChain, RAG, Vector DBs, SolidJS, Astro):
```bash
python scripts/verify_accuracy.py
```

Expected output:
```text
Starting Accuracy Verification...

Testing Contact Info:
  - Name: John Doe (Expected: John Doe) Pass
  - Phone: +1 (555) 123-4567 (Expected: +1 (555) 123-4567) Pass

Testing GPA Extraction:
  - Input: 'CGPA: 9.5/10' -> Extracted: 9.5/10
  - Input: 'GPA: 3.8 / 4.0' -> Extracted: 3.8/4.0
  - Input: 'Grade: A' -> Extracted: A/10

Testing Date Parsing:
  - Input: 'June 2021 - Present'
  - End Year detected correctly for 'Present'
  - Approx Years: 5.0

Testing New Skill Extraction:
  - Skill 'langchain': Found
  - Skill 'rag': Found
  - Skill 'openai': Found
  - Skill 'vector database': Found
  - Skill 'solidjs': Found
  - Skill 'astro': Found
  - Skill 'vercel': Found
  - Skill 'supabase': Found

Verification Complete!
```

---

## Deployment (Render / Cloud)

### Deploying to Render as a Web Service

1. **Create Web Service** on Render connected to this repository.
2. Configure settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
3. Add **Environment Variables** in Render Dashboard:
   - `DATABASE_URL`: Your Render External PostgreSQL URL (`postgresql://...`)
   - `USE_LOCAL_DB`: `false`
   - `SECRET_KEY`: A cryptographically secure random secret
   - `CORS_ORIGINS`: Your frontend URLs (e.g., `https://your-app.vercel.app`)

