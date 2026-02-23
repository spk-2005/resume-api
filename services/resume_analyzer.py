"""
resume_analyzer.py
──────────────────
Pure Python ATS engine. Accepts raw resume text (extracted from PDF)
and a job description string. Returns a full structured ATS report.

ATS Score Weightings (total = 100):
  ┌─────────────────────────┬──────────┐
  │ Component               │ Max Pts  │
  ├─────────────────────────┼──────────┤
  │ Skill Match             │   40     │
  │ Section Structure       │   25     │
  │ Format & Contact Info   │   15     │
  │ Power Verbs             │   10     │
  │ Quantified Achievements │   10     │
  └─────────────────────────┴──────────┘
"""

import re
from typing import Optional, Set, Dict, List, Tuple


# ══════════════════════════════════════════════════════════════════
#  KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════

KNOWN_SKILLS: Set[str] = {
    # ── Languages ──────────────────────────────────────────────────
    "python", "javascript", "typescript", "java", "c++", "c#", "go",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
    "bash", "shell", "perl", "dart", "elixir",

    # ── Frontend ───────────────────────────────────────────────────
    "html", "css", "react", "angular", "vue", "nextjs", "svelte",
    "jquery", "bootstrap", "tailwind", "sass", "webpack", "vite", "redux",

    # ── Backend ────────────────────────────────────────────────────
    "nodejs", "fastapi", "django", "flask", "express", "spring",
    "laravel", "rails", "graphql", "rest", "grpc",

    # ── Data / ML / AI ─────────────────────────────────────────────
    "sql", "nosql", "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "keras", "spark", "hadoop", "airflow", "dbt",
    "tableau", "powerbi", "machine learning", "deep learning",
    "nlp", "computer vision", "data analysis", "data engineering",
    "data science", "mlops", "xgboost", "lightgbm",

    # ── Cloud / DevOps ─────────────────────────────────────────────
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "ansible", "jenkins", "github actions", "gitlab ci", "ci/cd",
    "linux", "nginx", "prometheus", "grafana", "datadog", "helm",

    # ── Databases ──────────────────────────────────────────────────
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "cassandra", "sqlite", "dynamodb", "firebase", "neo4j", "oracle",

    # ── Mobile ─────────────────────────────────────────────────────
    "react native", "flutter", "android", "ios",

    # ── Security ───────────────────────────────────────────────────
    "oauth", "jwt", "ssl", "cybersecurity",

    # ── General Tech ───────────────────────────────────────────────
    "git", "agile", "scrum", "jira", "microservices", "api",
    "system design", "oop", "tdd", "unit testing", "design patterns",
    "solid principles", "code review",
}

# Aliases → canonical skill name
SKILL_ALIASES: Dict[str, str] = {
    "reactjs": "react",         "react.js": "react",
    "vuejs":   "vue",           "vue.js":   "vue",
    "node.js": "nodejs",        "node":     "nodejs",
    "postgres":"postgresql",    "pg":       "postgresql",
    "k8s":     "kubernetes",
    "sklearn": "scikit-learn",
    "js":      "javascript",    "ts":       "typescript",
    "ml":      "machine learning",
    "dl":      "deep learning",
    "ec2":     "aws",           "s3":       "aws",    "lambda": "aws",
    "gha":     "github actions",
    "ci":      "ci/cd",         "cd":       "ci/cd",
}

# ATS section keywords — checks resume completeness
ATS_SECTIONS: Dict[str, List[str]] = {
    "contact":        ["email", "phone", "linkedin", "github", "mobile", "contact"],
    "summary":        ["summary", "objective", "profile", "about", "overview"],
    "experience":     ["experience", "work experience", "employment", "work history", "career"],
    "education":      ["education", "degree", "university", "college", "bachelor",
                       "master", "phd", "b.tech", "m.tech", "b.e", "m.e"],
    "skills":         ["skills", "technical skills", "core competencies", "expertise", "technologies"],
    "projects":       ["projects", "project", "portfolio"],
    "certifications": ["certification", "certificate", "certified", "credential"],
    "achievements":   ["achievement", "award", "honor", "recognition", "accomplishment"],
}

# Section weights for structure scoring (total = 100)
SECTION_WEIGHTS: Dict[str, int] = {
    "contact":        20,
    "experience":     20,
    "skills":         15,
    "education":      15,
    "projects":       10,
    "summary":        10,
    "certifications":  5,
    "achievements":    5,
}

# Strong action verbs ATS and recruiters love
POWER_VERBS: Set[str] = {
    "developed", "designed", "built", "implemented", "deployed", "led",
    "managed", "created", "architected", "optimized", "improved",
    "reduced", "increased", "delivered", "launched", "automated",
    "integrated", "migrated", "scaled", "mentored", "collaborated",
    "analyzed", "engineered", "maintained", "resolved", "streamlined",
}


# ══════════════════════════════════════════════════════════════════
#  TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════

def tokenize(text: str) -> Set[str]:
    """Split text into a set of lowercase word tokens, keeping special chars for c++/c#."""
    return set(re.findall(r'\b[\w#+\.]+\b', text.lower()))


def extract_skills(text: str) -> Set[str]:
    """
    Pull known skills from text using three passes:
      1. Alias resolution  (reactjs → react)
      2. Multi-word phrases (machine learning, github actions)
      3. Single-word tokens (python, docker)
    """
    text_lower = text.lower()
    found: Set[str] = set()

    # Pass 1 — aliases
    for alias, canonical in SKILL_ALIASES.items():
        if re.search(r'\b' + re.escape(alias) + r'\b', text_lower):
            found.add(canonical)

    # Pass 2 — multi-word skills
    for skill in KNOWN_SKILLS:
        if " " in skill and skill in text_lower:
            found.add(skill)

    # Pass 3 — single-word skills
    tokens = tokenize(text_lower)
    for skill in KNOWN_SKILLS:
        if " " not in skill and skill in tokens:
            found.add(skill)

    return found


# ══════════════════════════════════════════════════════════════════
#  ANALYSIS MODULES
# ══════════════════════════════════════════════════════════════════

def detect_sections(resume_text: str) -> Dict[str, bool]:
    """Return which standard sections exist in the resume."""
    text_lower = resume_text.lower()
    return {
        sec: any(kw in text_lower for kw in keywords)
        for sec, keywords in ATS_SECTIONS.items()
    }


def score_sections(sections: Dict[str, bool]) -> Tuple[float, List[str]]:
    """
    Score resume structure 0–100.
    Returns (score, list_of_missing_section_names).
    """
    earned  = sum(SECTION_WEIGHTS[s] for s, present in sections.items() if present)
    missing = [s for s, present in sections.items() if not present]
    return float(earned), missing


def keyword_density(resume_text: str, job_skills: Set[str]) -> Dict[str, int]:
    """
    Count occurrences of each required job skill in the resume.
    count = 0 → skill is completely absent.
    count = 1 → mentioned once (weak signal).
    count ≥ 2 → good density.
    """
    text_lower = resume_text.lower()
    return {
        skill: len(re.findall(r'\b' + re.escape(skill) + r'\b', text_lower))
        for skill in sorted(job_skills)
    }


def detect_power_verbs(resume_text: str) -> Tuple[List[str], int]:
    """Return (list_of_found_verbs, count)."""
    found = sorted(tokenize(resume_text) & POWER_VERBS)
    return found, len(found)


def check_metrics(resume_text: str) -> Tuple[bool, int]:
    """
    Detect quantified achievements — numbers, percentages, scale indicators.
    e.g. '40%', '1M users', 'team of 8', 'reduced by 3x'
    """
    pattern = r'\b\d+[\+\%x]?\b|\b\d+\s*(percent|million|billion|thousand|k)\b'
    matches = re.findall(pattern, resume_text.lower())
    return bool(matches), len(matches)


def check_format(resume_text: str) -> Dict[str, object]:
    """
    Infer ATS-friendliness signals from the extracted plain text.
    Returns a dict of checks with bool values + word count.
    """
    wc = len(resume_text.split())
    return {
        "has_email":         bool(re.search(r'[\w\.-]+@[\w\.-]+\.\w+', resume_text)),
        "has_phone":         bool(re.search(r'[\+\d][\d\s\-\(\)]{7,}', resume_text)),
        "has_linkedin":      "linkedin"   in resume_text.lower(),
        "has_github":        "github"     in resume_text.lower(),
        "reasonable_length": 200 <= wc <= 1200,
        "word_count":        wc,
    }


# ══════════════════════════════════════════════════════════════════
#  ATS SCORE CALCULATOR
# ══════════════════════════════════════════════════════════════════

def calculate_ats_score(
    skill_pct:    float,
    section_pct:  float,
    verb_count:   int,
    has_metrics:  bool,
    fmt:          Dict,
) -> Tuple[float, Dict[str, float]]:
    """
    Weighted scoring model — all components add to max 100.

    Component            Max   Formula
    ──────────────────── ───── ──────────────────────────────────
    Skill Match           40   skill_pct × 0.40  (capped at 40)
    Section Structure     25   section_pct × 0.25 (capped at 25)
    Format & Contact      15   5+4+2+2+2 per check (capped at 15)
    Power Verbs           10   verb_count, capped at 10
    Quantified Impact     10   10 if metrics found, else 0
    """
    skill_pts   = min(skill_pct   * 0.40, 40.0)
    section_pts = min(section_pct * 0.25, 25.0)

    fmt_pts = (
        (5 if fmt.get("has_email")         else 0) +
        (4 if fmt.get("has_phone")         else 0) +
        (2 if fmt.get("has_linkedin")      else 0) +
        (2 if fmt.get("has_github")        else 0) +
        (2 if fmt.get("reasonable_length") else 0)
    )
    fmt_pts     = min(fmt_pts, 15.0)
    verb_pts    = float(min(verb_count, 10))
    metrics_pts = 10.0 if has_metrics else 0.0

    total = skill_pts + section_pts + fmt_pts + verb_pts + metrics_pts

    breakdown = {
        "skill_match_score    (max 40)": round(skill_pts,   1),
        "section_structure    (max 25)": round(section_pts, 1),
        "format_contact       (max 15)": round(fmt_pts,     1),
        "power_verbs          (max 10)": round(verb_pts,    1),
        "quantified_impact    (max 10)": round(metrics_pts, 1),
    }
    return round(total, 1), breakdown


def ats_rating(score: float) -> str:
    if score >= 80: return "Excellent ✅"
    if score >= 65: return "Good 🟢"
    if score >= 50: return "Average 🟡"
    if score >= 35: return "Below Average 🟠"
    return "Poor 🔴"


# ══════════════════════════════════════════════════════════════════
#  RECOMMENDATIONS ENGINE
# ══════════════════════════════════════════════════════════════════

def build_recommendations(
    ats_score:       float,
    missing_skills:  List[str],
    missing_sections:List[str],
    has_metrics:     bool,
    verb_count:      int,
    fmt:             Dict,
    density:         Dict[str, int],
) -> List[Dict[str, str]]:
    """
    Returns a list of recommendation dicts:
      { "priority": "HIGH|MEDIUM|LOW", "category": "...", "action": "..." }
    Sorted by priority so the consumer can render them cleanly.
    """
    recs: List[Dict[str, str]] = []

    def add(priority: str, category: str, action: str):
        recs.append({"priority": priority, "category": category, "action": action})

    # ── Overall verdict ─────────────────────────────────────────────
    if ats_score >= 80:
        add("LOW",  "Overall", "Strong ATS profile. Fine-tune the gaps below to push past 90.")
    elif ats_score >= 60:
        add("MEDIUM","Overall","Moderate score — address the HIGH priority items to improve shortlisting chances.")
    else:
        add("HIGH", "Overall", "Low ATS score. Resume may be auto-rejected. Fix HIGH priority items urgently.")

    # ── Missing skills ──────────────────────────────────────────────
    if missing_skills:
        add("HIGH", "Skills",
            f"Add these missing required skills: {', '.join(missing_skills[:6])}. "
            "Include them in your Skills section AND in experience bullets — ATS scans the full document.")

    # ── Low keyword density ─────────────────────────────────────────
    single_mention = [s for s, c in density.items() if c == 1]
    if single_mention:
        add("MEDIUM", "Keyword Density",
            f"These skills appear only once — try to mention them in 2+ places: "
            f"{', '.join(single_mention[:4])}.")

    # ── Missing sections ────────────────────────────────────────────
    section_advice = {
        "contact":        "Add a Contact section with your email, phone, and LinkedIn URL — ATS needs this to parse your identity.",
        "summary":        "Add a 2–3 line Professional Summary at the top, tailored to this specific job description.",
        "experience":     "Add a Work Experience section — the most critical section for ATS parsing.",
        "education":      "Add an Education section with your degree, institution, and year of graduation.",
        "skills":         "Add a dedicated Skills section — most ATS systems specifically scan for this heading.",
        "projects":       "Add a Projects section with your tech stack and measurable impact per project.",
        "certifications": "Add a Certifications section to boost credibility for technical roles.",
        "achievements":   "Add an Achievements section highlighting measurable wins.",
    }
    priority_map = {
        "contact": "HIGH", "experience": "HIGH", "skills": "HIGH", "education": "HIGH",
        "summary": "MEDIUM", "projects": "MEDIUM",
        "certifications": "LOW", "achievements": "LOW",
    }
    for sec in missing_sections:
        add(priority_map.get(sec, "LOW"), f"Missing Section: {sec.title()}", section_advice[sec])

    # ── Quantified metrics ──────────────────────────────────────────
    if not has_metrics:
        add("HIGH", "Impact Metrics",
            "No measurable achievements found. Add numbers to your bullets: "
            "'Reduced load time by 40%', 'Led team of 8', '50,000 daily active users', '₹2L revenue increase'.")

    # ── Power verbs ─────────────────────────────────────────────────
    if verb_count < 5:
        add("MEDIUM", "Action Verbs",
            "Use stronger action verbs at the start of each bullet point: "
            "Developed · Built · Architected · Optimized · Deployed · Automated · Scaled · Delivered · Reduced.")

    # ── Contact / format ────────────────────────────────────────────
    if not fmt.get("has_email"):
        add("HIGH", "Contact Info", "No email detected. Ensure it is in plain text — not embedded in a header image.")
    if not fmt.get("has_phone"):
        add("HIGH", "Contact Info", "No phone number detected. Add it in a standard format (+91-XXXXXXXXXX).")
    if not fmt.get("has_linkedin"):
        add("MEDIUM", "Contact Info", "Add your LinkedIn profile URL (linkedin.com/in/yourname).")
    if not fmt.get("has_github"):
        add("MEDIUM", "Contact Info", "Add your GitHub profile URL — essential for engineering roles.")

    wc = fmt.get("word_count", 0)
    if not fmt.get("reasonable_length"):
        if wc < 200:
            add("MEDIUM", "Resume Length",
                f"Resume is too short ({wc} words). Expand experience bullets, add more project detail.")
        else:
            add("MEDIUM", "Resume Length",
                f"Resume is too long ({wc} words). Trim to 1–2 pages — recruiters and ATS both prefer concise resumes.")

    # ── Universal ATS format tips ───────────────────────────────────
    add("LOW", "ATS Formatting",
        "Use a single-column layout. Avoid tables, text boxes, headers/footers, and images — "
        "ATS parsers frequently skip content inside these elements.")
    add("LOW", "File Format",
        "Save your resume as a text-based PDF or .docx file — NOT a scanned image PDF. "
        "Scanned PDFs produce zero text when parsed by ATS.")

    # Sort: HIGH → MEDIUM → LOW
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recs.sort(key=lambda r: order[r["priority"]])
    return recs


# ══════════════════════════════════════════════════════════════════
#  MASTER FUNCTION
# ══════════════════════════════════════════════════════════════════

def analyze_resume(
    resume_text:     str,
    job_description: str,
    candidate_name:  Optional[str] = None,
) -> Dict:
    """
    Full ATS analysis of extracted resume text vs a job description.
    Called by the FastAPI endpoints after PDF text extraction.

    Returns a structured dict with:
      ats_score, ats_rating, score_breakdown,
      skill_match_percentage, matched_skills, missing_skills,
      keyword_density, sections_detected, missing_sections,
      power_verbs_found, has_quantified_metrics,
      format_checks, recommendations
    """

    # 1 ── Skill extraction & matching
    resume_skills  = extract_skills(resume_text)
    job_skills     = extract_skills(job_description)

    matched  = sorted(resume_skills & job_skills)
    missing  = sorted(job_skills  - resume_skills)
    extra    = sorted(resume_skills - job_skills)          # bonus: you have these but JD doesn't require them
    skill_pct = round((len(matched) / len(job_skills)) * 100, 1) if job_skills else 0.0

    # 2 ── Section detection
    sections             = detect_sections(resume_text)
    sec_score, miss_secs = score_sections(sections)

    # 3 ── Keyword density
    density = keyword_density(resume_text, job_skills)

    # 4 ── Power verbs
    verbs, verb_count = detect_power_verbs(resume_text)

    # 5 ── Quantified achievements
    has_metrics, metric_count = check_metrics(resume_text)

    # 6 ── Format signals
    fmt = check_format(resume_text)

    # 7 ── ATS score
    score, breakdown = calculate_ats_score(
        skill_pct=skill_pct, section_pct=sec_score,
        verb_count=verb_count, has_metrics=has_metrics, fmt=fmt,
    )

    # 8 ── Recommendations
    recs = build_recommendations(
        ats_score=score, missing_skills=missing, missing_sections=miss_secs,
        has_metrics=has_metrics, verb_count=verb_count, fmt=fmt, density=density,
    )

    return {
        # ── Identity ───────────────────────────────────────────────
        "candidate_name":           candidate_name,

        # ── ATS Score ──────────────────────────────────────────────
        "ats_score":                score,            # e.g. 73.5  (out of 100)
        "ats_rating":               ats_rating(score),# e.g. "Good 🟢"
        "score_breakdown":          breakdown,        # per-component points

        # ── Skill Analysis ─────────────────────────────────────────
        "skill_match_percentage":   skill_pct,        # e.g. 62.5
        "total_required_skills":    len(job_skills),
        "matched_skills":           matched,          # skills in BOTH resume & JD
        "missing_skills":           missing,          # in JD but NOT in resume
        "extra_skills":             extra,            # in resume but not required by JD

        # ── Keyword Density ────────────────────────────────────────
        "keyword_density":          density,          # {skill: count_in_resume}

        # ── Section Analysis ───────────────────────────────────────
        "sections_detected":        sections,         # {section: true/false}
        "missing_sections":         miss_secs,

        # ── Quality Signals ────────────────────────────────────────
        "power_verbs_found":        verbs,
        "has_quantified_metrics":   has_metrics,
        "metric_count":             metric_count,
        "format_checks":            fmt,

        # ── Recommendations ────────────────────────────────────────
        "recommendations":          recs,             # sorted HIGH → MEDIUM → LOW
    }