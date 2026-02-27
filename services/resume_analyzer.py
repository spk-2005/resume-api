"""
resume_analyzer.py
──────────────────
High-accuracy ATS engine. Returns a clean, professional structured
report — no recommendations, easy to read.

ATS Score Breakdown (Total = 100 pts):
  ┌──────────────────────────┬──────────┐
  │ Component                │ Max Pts  │
  ├──────────────────────────┼──────────┤
  │ Skill Match              │   45     │
  │ Section Structure        │   20     │
  │ Experience Match         │   15     │
  │ Format & Contact Info    │   10     │
  │ Power Verbs + Metrics    │   10     │
  └──────────────────────────┴──────────┘
"""

import re
from typing import Optional, Set, Dict, List, Tuple


# ══════════════════════════════════════════════════════════════════
#  SKILL PATTERNS  (canonical → surface forms)
# ══════════════════════════════════════════════════════════════════

SKILL_PATTERNS: List[Tuple[str, List[str]]] = [
    # Multi-word (must come before single-word variants)
    ("machine learning",            ["machine learning", "ml"]),
    ("deep learning",               ["deep learning", "dl"]),
    ("natural language processing", ["natural language processing", "nlp"]),
    ("computer vision",             ["computer vision"]),
    ("data science",                ["data science"]),
    ("data engineering",            ["data engineering"]),
    ("data analysis",               ["data analysis", "data analytics"]),
    ("data visualization",          ["data visualization", "data viz"]),
    ("big data",                    ["big data"]),
    ("react native",                ["react native"]),
    ("github actions",              ["github actions", "gha"]),
    ("gitlab ci",                   ["gitlab ci", "gitlab ci/cd"]),
    ("google cloud",                ["google cloud", "gcp", "google cloud platform"]),
    ("ci/cd",                       ["ci/cd", "ci cd", "continuous integration", "continuous deployment", "continuous delivery"]),
    ("system design",               ["system design"]),
    ("design patterns",             ["design patterns"]),
    ("solid principles",            ["solid principles"]),
    ("unit testing",                ["unit testing", "unit tests"]),
    ("code review",                 ["code review"]),
    ("object oriented",             ["object oriented", "oop", "object-oriented"]),
    ("test driven development",     ["test driven", "tdd", "test-driven"]),
    ("scikit-learn",                ["scikit-learn", "sklearn", "scikit learn"]),
    ("power bi",                    ["power bi", "powerbi"]),
    ("next.js",                     ["next.js", "nextjs", "next js"]),
    ("vue.js",                      ["vue.js", "vuejs", "vue js", "vue"]),
    ("node.js",                     ["node.js", "nodejs", "node js", "node"]),
    ("react.js",                    ["react.js", "reactjs", "react js", "react"]),
    ("express.js",                  ["express.js", "expressjs", "express js", "express"]),
    ("spring boot",                 ["spring boot", "spring"]),
    ("ruby on rails",               ["ruby on rails", "rails"]),
    ("amazon web services",         ["amazon web services", "aws"]),
    ("microsoft azure",             ["microsoft azure", "azure"]),
    ("apache spark",                ["apache spark", "spark", "pyspark"]),
    ("apache airflow",              ["airflow", "apache airflow"]),
    ("rest api",                    ["rest api", "restful api", "restful", "rest"]),
    ("problem solving",             ["problem solving", "problem-solving"]),
    ("time management",             ["time management"]),
    ("critical thinking",           ["critical thinking"]),
    ("ruby on rails",               ["ruby on rails", "rails"]),

    # Single-word skills
    ("python",         ["python"]),
    ("javascript",     ["javascript", "js"]),
    ("typescript",     ["typescript", "ts"]),
    ("java",           ["java"]),
    ("c++",            ["c++", "cpp"]),
    ("c#",             ["c#", "csharp"]),
    ("golang",         ["golang", "go"]),
    ("rust",           ["rust"]),
    ("ruby",           ["ruby"]),
    ("php",            ["php"]),
    ("swift",          ["swift"]),
    ("kotlin",         ["kotlin"]),
    ("scala",          ["scala"]),
    ("r",              ["\\br\\b"]),
    ("matlab",         ["matlab"]),
    ("bash",           ["bash", "shell scripting"]),
    ("dart",           ["dart"]),
    ("elixir",         ["elixir"]),
    ("html",           ["html", "html5"]),
    ("css",            ["css", "css3"]),
    ("angular",        ["angular", "angularjs"]),
    ("svelte",         ["svelte"]),
    ("jquery",         ["jquery"]),
    ("bootstrap",      ["bootstrap"]),
    ("tailwind",       ["tailwind", "tailwindcss"]),
    ("sass",           ["sass", "scss"]),
    ("webpack",        ["webpack"]),
    ("redux",          ["redux"]),
    ("fastapi",        ["fastapi"]),
    ("django",         ["django"]),
    ("flask",          ["flask"]),
    ("graphql",        ["graphql"]),
    ("grpc",           ["grpc"]),
    ("laravel",        ["laravel"]),
    ("sql",            ["sql"]),
    ("nosql",          ["nosql"]),
    ("pandas",         ["pandas"]),
    ("numpy",          ["numpy"]),
    ("tensorflow",     ["tensorflow"]),
    ("pytorch",        ["pytorch"]),
    ("keras",          ["keras"]),
    ("hadoop",         ["hadoop"]),
    ("dbt",            ["dbt"]),
    ("tableau",        ["tableau"]),
    ("xgboost",        ["xgboost"]),
    ("lightgbm",       ["lightgbm"]),
    ("mlops",          ["mlops"]),
    ("docker",         ["docker"]),
    ("kubernetes",     ["kubernetes", "k8s"]),
    ("terraform",      ["terraform"]),
    ("ansible",        ["ansible"]),
    ("jenkins",        ["jenkins"]),
    ("linux",          ["linux", "ubuntu", "centos"]),
    ("nginx",          ["nginx"]),
    ("prometheus",     ["prometheus"]),
    ("grafana",        ["grafana"]),
    ("datadog",        ["datadog"]),
    ("helm",           ["helm"]),
    ("postgresql",     ["postgresql", "postgres"]),
    ("mysql",          ["mysql"]),
    ("mongodb",        ["mongodb", "mongo"]),
    ("redis",          ["redis"]),
    ("elasticsearch",  ["elasticsearch"]),
    ("cassandra",      ["cassandra"]),
    ("sqlite",         ["sqlite"]),
    ("dynamodb",       ["dynamodb"]),
    ("firebase",       ["firebase"]),
    ("neo4j",          ["neo4j"]),
    ("flutter",        ["flutter"]),
    ("android",        ["android"]),
    ("ios",            ["ios"]),
    ("oauth",          ["oauth", "oauth2"]),
    ("jwt",            ["jwt"]),
    ("cybersecurity",  ["cybersecurity", "cyber security", "infosec"]),
    ("git",            ["git"]),
    ("agile",          ["agile"]),
    ("scrum",          ["scrum"]),
    ("jira",           ["jira"]),
    ("microservices",  ["microservices", "microservice"]),
    ("excel",          ["excel", "ms excel"]),
    ("communication",  ["communication"]),
    ("leadership",     ["leadership", "team lead", "tech lead"]),
    ("teamwork",       ["teamwork", "team player"]),
]

# Pre-compile all regex patterns
_SKILL_REGEXES: Dict[str, List[re.Pattern]] = {}
for _canonical, _forms in SKILL_PATTERNS:
    _pats = []
    for _form in _forms:
        if _form.startswith("\\b"):
            _pats.append(re.compile(_form, re.IGNORECASE))
        else:
            _pats.append(re.compile(
                r'(?<![a-zA-Z0-9])' + re.escape(_form) + r'(?![a-zA-Z0-9])',
                re.IGNORECASE
            ))
    _SKILL_REGEXES[_canonical] = _pats


# ══════════════════════════════════════════════════════════════════
#  SECTIONS
# ══════════════════════════════════════════════════════════════════

ATS_SECTIONS: Dict[str, List[str]] = {
    "Contact Info":     ["email", "phone", "linkedin", "github", "mobile", "contact", "@"],
    "Summary":          ["summary", "objective", "profile", "about me", "overview", "career objective"],
    "Work Experience":  ["experience", "work experience", "employment", "work history", "professional experience", "internship"],
    "Education":        ["education", "academic", "degree", "university", "college",
                         "bachelor", "master", "phd", "b.tech", "m.tech", "b.e", "m.e",
                         "b.sc", "m.sc", "mba", "bca", "mca"],
    "Skills":           ["skills", "technical skills", "core competencies", "expertise", "technologies", "tech stack"],
    "Projects":         ["projects", "project work", "portfolio", "open source"],
    "Certifications":   ["certification", "certificate", "certified", "credential", "course"],
    "Achievements":     ["achievement", "award", "honor", "recognition", "accomplishment", "scholarship"],
}

SECTION_WEIGHTS: Dict[str, int] = {
    "Contact Info":    20,
    "Work Experience": 20,
    "Skills":          15,
    "Education":       15,
    "Projects":        10,
    "Summary":         10,
    "Certifications":   5,
    "Achievements":     5,
}

POWER_VERBS: Set[str] = {
    "developed", "designed", "built", "implemented", "deployed", "led",
    "managed", "created", "architected", "optimized", "improved",
    "reduced", "increased", "delivered", "launched", "automated",
    "integrated", "migrated", "scaled", "mentored", "collaborated",
    "analyzed", "engineered", "maintained", "resolved", "streamlined",
    "established", "accelerated", "spearheaded", "orchestrated",
    "pioneered", "revamped", "negotiated", "coordinated", "generated",
}

EDU_LEVELS: Dict[str, int] = {
    "phd": 5, "doctorate": 5,
    "m.tech": 4, "mtech": 4, "master": 4, "mba": 4, "msc": 4, "mca": 4,
    "b.tech": 3, "btech": 3, "b.e": 3, "bachelor": 3, "bsc": 3, "bca": 3,
    "diploma": 2,
    "12th": 1, "hsc": 1,
    "10th": 0, "ssc": 0,
}

EDU_LABELS: Dict[int, str] = {
    0: "Not Specified", 1: "10th / SSC", 2: "Diploma / 12th",
    3: "Bachelor's Degree", 4: "Master's Degree", 5: "PhD / Doctorate",
}

JOB_TITLES: Dict[str, List[str]] = {
    "Software Engineer":    ["software engineer", "software developer", "sde", "swe"],
    "Frontend Developer":   ["frontend", "front-end", "front end", "ui developer"],
    "Backend Developer":    ["backend", "back-end", "back end", "server side"],
    "Full Stack Developer": ["full stack", "fullstack", "full-stack"],
    "Data Scientist":       ["data scientist", "data science"],
    "Data Engineer":        ["data engineer", "data pipeline"],
    "ML Engineer":          ["machine learning engineer", "ml engineer", "mlops engineer"],
    "DevOps Engineer":      ["devops", "dev ops", "site reliability", "sre", "platform engineer"],
    "Cloud Engineer":       ["cloud engineer", "cloud architect", "solutions architect"],
    "Android Developer":    ["android developer", "android engineer"],
    "iOS Developer":        ["ios developer", "ios engineer"],
    "Mobile Developer":     ["mobile developer", "mobile engineer"],
    "Data Analyst":         ["data analyst", "business analyst", "bi analyst"],
    "QA Engineer":          ["qa engineer", "quality assurance", "test engineer", "sdet"],
    "Security Engineer":    ["security engineer", "cybersecurity engineer", "infosec engineer"],
    "Product Manager":      ["product manager", "product owner"],
}


# ══════════════════════════════════════════════════════════════════
#  EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════

def extract_skills(text: str) -> Dict[str, int]:
    """Extract skills with occurrence count. Returns {skill: count}."""
    text_lower = text.lower()
    found: Dict[str, int] = {}
    consumed: List[Tuple[int, int]] = []

    for canonical, _ in SKILL_PATTERNS:
        patterns = _SKILL_REGEXES[canonical]
        count = 0
        spans = []
        for pat in patterns:
            for m in pat.finditer(text_lower):
                span = (m.start(), m.end())
                if not any(s <= span[0] and span[1] <= e for s, e in consumed):
                    count += 1
                    spans.append(span)
        if count > 0:
            found[canonical] = found.get(canonical, 0) + count
            consumed.extend(spans)

    return found


def detect_sections(text: str) -> Dict[str, bool]:
    text_lower = text.lower()
    return {
        sec: any(kw in text_lower for kw in kws)
        for sec, kws in ATS_SECTIONS.items()
    }


def detect_education_level(text: str) -> int:
    text_lower = text.lower()
    level = 0
    for kw, lvl in EDU_LEVELS.items():
        if re.search(r'(?<![a-zA-Z])' + re.escape(kw) + r'(?![a-zA-Z])', text_lower):
            level = max(level, lvl)
    return level


def extract_required_years(jd_text: str) -> Optional[int]:
    patterns = [
        r'(\d+)\+?\s*years?\s+of\s+experience',
        r'(\d+)\+?\s*years?\s+experience',
        r'minimum\s+(\d+)\s+years?',
        r'at\s+least\s+(\d+)\s+years?',
        r'(\d+)\s*-\s*\d+\s+years?\s+of\s+experience',
        r'(\d+)\+\s*yrs',
    ]
    for pat in patterns:
        m = re.search(pat, jd_text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def estimate_resume_years(resume_text: str) -> Optional[float]:
    years = [int(y) for y in re.findall(r'\b(20[0-2]\d|19[89]\d)\b', resume_text)]
    if not years:
        return None
    span = max(years) - min(years)
    return float(min(span, 40)) if span > 0 else None


def extract_job_titles(text: str) -> Set[str]:
    text_lower = text.lower()
    return {role for role, variants in JOB_TITLES.items()
            if any(v in text_lower for v in variants)}


def detect_power_verbs(text: str) -> Tuple[List[str], int]:
    tokens = set(re.findall(r'\b\w+\b', text.lower()))
    found  = sorted(tokens & POWER_VERBS)
    return found, len(found)


def check_metrics(text: str) -> Tuple[bool, int]:
    patterns = [
        r'\d+\s*%', r'\d+\s*x\b',
        r'\$\s*\d+[\w]*', r'₹\s*\d+[\w]*',
        r'\d+\s*(million|billion|thousand|k|m)\b',
        r'\b\d{4,}\b', r'team\s+of\s+\d+',
        r'\d+\+?\s*(users|clients|customers|requests)',
        r'reduced\s+by\s+\d+', r'improved\s+by\s+\d+',
    ]
    hits = []
    for pat in patterns:
        hits.extend(re.findall(pat, text.lower()))
    return bool(hits), len(hits)


def check_format(text: str) -> Dict[str, object]:
    wc = len(text.split())
    return {
        "has_email":         bool(re.search(r'[\w\.\-\+]+@[\w\.-]+\.\w{2,}', text)),
        "has_phone":         bool(re.search(r'[\+\d][\d\s\-\(\)]{8,}', text)),
        "has_linkedin":      bool(re.search(r'linkedin', text, re.IGNORECASE)),
        "has_github":        bool(re.search(r'github', text, re.IGNORECASE)),
        "word_count":        wc,
        "length_status":     "Good (1–2 pages)" if 250 <= wc <= 1400
                             else ("Too Short" if wc < 250 else "Too Long"),
    }


# ══════════════════════════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════════════════════════

def weighted_skill_score(
    resume_skills: Dict[str, int],
    jd_skills:     Dict[str, int],
) -> Tuple[float, List[str], List[str], List[str]]:
    if not jd_skills:
        return 0.0, [], [], sorted(resume_skills.keys())

    max_count = max(jd_skills.values()) or 1
    weights   = {s: 0.5 + 0.5 * (c / max_count) for s, c in jd_skills.items()}

    jd_names  = set(jd_skills.keys())
    rs_names  = set(resume_skills.keys())
    matched   = sorted(jd_names & rs_names)
    missing   = sorted(jd_names - rs_names)
    extra     = sorted(rs_names - jd_names)

    matched_w = sum(weights[s] for s in matched)
    total_w   = sum(weights.values()) or 1
    pct       = round(min((matched_w / total_w) * 100, 100.0), 1)

    return pct, matched, missing, extra


def score_sections(sections: Dict[str, bool]) -> Tuple[float, List[str], List[str]]:
    present = [s for s, v in sections.items() if v]
    missing = [s for s, v in sections.items() if not v]
    earned  = sum(SECTION_WEIGHTS.get(s, 0) for s in present)
    return float(earned), present, missing


def experience_score(resume_text: str, jd_text: str) -> Tuple[float, Optional[int], Optional[float]]:
    required = extract_required_years(jd_text)
    actual   = estimate_resume_years(resume_text)
    if required is None:
        return 0.75, None, actual
    if actual is None:
        return 0.50, required, None
    if actual >= required:
        return 1.0, required, actual
    return round(actual / required, 2), required, actual


def title_alignment(resume_text: str, jd_text: str) -> Tuple[float, List[str], List[str]]:
    jd_titles     = extract_job_titles(jd_text)
    resume_titles = extract_job_titles(resume_text)
    if not jd_titles:
        return 1.0, [], []
    overlap = jd_titles & resume_titles
    score   = 1.0 if overlap else (0.6 if (
        set(" ".join(jd_titles).split()) & set(" ".join(resume_titles).split())
    ) else 0.2)
    return score, sorted(jd_titles), sorted(resume_titles)


def compute_ats_score(
    skill_pct:    float,
    section_pct:  float,
    exp_score:    float,
    verb_count:   int,
    has_metrics:  bool,
    metric_count: int,
    fmt:          Dict,
    t_score:      float,
) -> Tuple[float, Dict[str, str]]:

    skill_pts   = min(skill_pct * 0.45, 45.0)
    section_pts = min(section_pct * 0.20, 20.0)
    exp_pts     = round(exp_score * 15.0, 1)
    fmt_pts     = min(
        (3 if fmt.get("has_email")  else 0) +
        (3 if fmt.get("has_phone")  else 0) +
        (2 if fmt.get("has_linkedin") else 0) +
        (1 if fmt.get("has_github") else 0) +
        (1 if fmt.get("length_status") == "Good (1–2 pages)" else 0),
        10.0
    )
    verb_pts    = float(min(verb_count, 5))
    metric_pts  = min(float(metric_count), 5.0) if has_metrics else 0.0
    raw         = skill_pts + section_pts + exp_pts + fmt_pts + verb_pts + metric_pts
    total       = round(min(raw * (0.85 + 0.15 * t_score), 100.0), 1)

    breakdown = {
        "Skill Match       (max 45 pts)": f"{round(skill_pts,   1)} pts",
        "Section Structure (max 20 pts)": f"{round(section_pts, 1)} pts",
        "Experience Match  (max 15 pts)": f"{round(exp_pts,     1)} pts",
        "Format & Contact  (max 10 pts)": f"{round(fmt_pts,     1)} pts",
        "Power Verbs       (max  5 pts)": f"{round(verb_pts,    1)} pts",
        "Quantified Impact (max  5 pts)": f"{round(metric_pts,  1)} pts",
    }
    return total, breakdown


def ats_grade(score: float) -> str:
    if score >= 85: return "Excellent"
    if score >= 70: return "Good"
    if score >= 55: return "Average"
    if score >= 40: return "Below Average"
    return "Poor"


def ats_emoji(score: float) -> str:
    if score >= 85: return "✅"
    if score >= 70: return "🟢"
    if score >= 55: return "🟡"
    if score >= 40: return "🟠"
    return "🔴"


# ══════════════════════════════════════════════════════════════════
#  MASTER FUNCTION
# ══════════════════════════════════════════════════════════════════

def analyze_resume(
    resume_text:     str,
    job_description: str,
    candidate_name:  Optional[str] = None,
) -> Dict:

    # Skills
    resume_skills = extract_skills(resume_text)
    jd_skills     = extract_skills(job_description)
    skill_pct, matched_skills, missing_skills, extra_skills = weighted_skill_score(resume_skills, jd_skills)

    # Keyword density for JD-required skills only
    density = {
        skill: {"found_in_resume": resume_skills.get(skill, 0), "mentioned_in_jd": jd_skills[skill]}
        for skill in sorted(jd_skills.keys())
    }

    # Sections
    sections                        = detect_sections(resume_text)
    section_pct, present_secs, missing_secs = score_sections(sections)

    # Experience
    exp_sc, req_yrs, res_yrs        = experience_score(resume_text, job_description)

    # Education
    edu_req  = detect_education_level(job_description)
    edu_res  = detect_education_level(resume_text)

    # Title alignment
    t_score, jd_titles, res_titles  = title_alignment(resume_text, job_description)

    # Power verbs
    verbs, verb_count               = detect_power_verbs(resume_text)

    # Metrics
    has_metrics, metric_count       = check_metrics(resume_text)

    # Format
    fmt                             = check_format(resume_text)

    # ATS Score
    score, breakdown                = compute_ats_score(
        skill_pct, section_pct, exp_sc,
        verb_count, has_metrics, metric_count,
        fmt, t_score,
    )

    return {

        # ── Candidate ─────────────────────────────────────────────
        "candidate": {
            "name":  candidate_name or "Not Provided",
        },

        # ── ATS Score ─────────────────────────────────────────────
        "ats_score": {
            "score":      score,
            "out_of":     100,
            "grade":      ats_grade(score),
            "status":     ats_emoji(score) + "  " + ats_grade(score),
            "breakdown":  breakdown,
        },

        # ── Skill Analysis ────────────────────────────────────────
        "skill_analysis": {
            "match_percentage":        f"{skill_pct}%",
            "total_skills_in_jd":      len(jd_skills),
            "skills_matched":          len(matched_skills),
            "skills_missing":          len(missing_skills),
            "matched_skills":          matched_skills,
            "missing_skills":          missing_skills,
            "additional_skills_in_resume": extra_skills,
        },

        # ── Keyword Density ───────────────────────────────────────
        "keyword_density": {
            "description": "How many times each JD-required skill appears in your resume",
            "skills":      density,
        },

        # ── Section Analysis ──────────────────────────────────────
        "section_analysis": {
            "sections_present":  present_secs,
            "sections_missing":  missing_secs,
            "completeness":      f"{round((len(present_secs) / len(ATS_SECTIONS)) * 100)}%",
        },

        # ── Experience ────────────────────────────────────────────
        "experience": {
            "required_by_jd":       f"{req_yrs} years" if req_yrs else "Not specified",
            "detected_in_resume":   f"~{int(res_yrs)} years" if res_yrs else "Not detected",
            "match_status":         (
                "Meets Requirement ✅"     if exp_sc >= 1.0  else
                "Partially Meets ⚠️"      if exp_sc >= 0.6  else
                "Below Requirement ❌"    if req_yrs        else
                "No Requirement Stated ℹ️"
            ),
            "match_score":          f"{round(exp_sc * 100)}%",
        },

        # ── Education ─────────────────────────────────────────────
        "education": {
            "required_by_jd":       EDU_LABELS.get(edu_req, "Not Specified"),
            "detected_in_resume":   EDU_LABELS.get(edu_res, "Not Detected"),
            "match_status":         (
                "Meets Requirement ✅"  if edu_res >= edu_req else
                "Below Requirement ❌"
            ),
        },

        # ── Job Title ─────────────────────────────────────────────
        "job_title_alignment": {
            "roles_in_jd":          jd_titles if jd_titles else ["Not specified"],
            "roles_in_resume":      res_titles if res_titles else ["Not detected"],
            "alignment_score":      f"{round(t_score * 100)}%",
            "status":               (
                "Strong Match ✅"   if t_score >= 0.9 else
                "Partial Match ⚠️" if t_score >= 0.5 else
                "Weak Match ❌"
            ),
        },

        # ── Writing Quality ───────────────────────────────────────
        "writing_quality": {
            "power_verbs_used":      verbs,
            "power_verb_count":      verb_count,
            "verb_strength":         (
                "Strong ✅"   if verb_count >= 8 else
                "Moderate ⚠️" if verb_count >= 4 else
                "Weak ❌"
            ),
            "quantified_achievements": has_metrics,
            "metric_count":          metric_count,
            "metrics_strength":      (
                "Strong ✅"   if metric_count >= 5 else
                "Moderate ⚠️" if metric_count >= 2 else
                "Weak ❌"
            ),
        },

        # ── Format & Contact ──────────────────────────────────────
        "format_and_contact": {
            "email_present":    "Yes ✅" if fmt.get("has_email")    else "No ❌",
            "phone_present":    "Yes ✅" if fmt.get("has_phone")    else "No ❌",
            "linkedin_present": "Yes ✅" if fmt.get("has_linkedin") else "No ❌",
            "github_present":   "Yes ✅" if fmt.get("has_github")   else "No ❌",
            "word_count":       fmt.get("word_count"),
            "length_status":    fmt.get("length_status"),
        },

    }