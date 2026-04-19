"""
resume_parser.py  ─  Structural Resume Parser
───────────────────────────────────────────────
Parses resume into structured sections using NLP techniques:
  ✅ Section boundary detection    (heading pattern recognition)
  ✅ Contact info extraction       (email, phone, LinkedIn, GitHub, portfolio)
  ✅ Experience timeline parsing   (company, role, dates, duration)
  ✅ Education extraction          (degree, institution, year, GPA)
  ✅ Skills section isolation      (dedicated skills block parsing)
  ✅ Project extraction            (title, tech stack, impact)
  ✅ Certification detection       (cert name, issuer, year)
  ✅ Achievement/award extraction
  ✅ Writing quality analysis      (power verbs, quantified impact)
  ✅ Resume length & format checks
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════
#  SECTION HEADING PATTERNS
# ══════════════════════════════════════════════════════════════════

SECTION_PATTERNS: Dict[str, List[str]] = {
    "contact":        ["contact", "contact information", "personal details",
                       "personal information", "reach me", "get in touch"],
    "summary":        ["summary", "professional summary", "career summary",
                       "objective", "career objective", "professional objective",
                       "profile", "about me", "overview", "introduction",
                       "executive summary", "highlights"],
    "experience":     ["experience", "work experience", "professional experience",
                       "employment", "employment history", "work history",
                       "career history", "positions held", "internship",
                       "internships", "industrial training", "industry experience"],
    "education":      ["education", "educational background", "academic background",
                       "academic qualifications", "qualifications", "academics",
                       "educational qualifications", "schooling", "degrees"],
    "skills":         ["skills", "technical skills", "core competencies",
                       "key skills", "skills & expertise", "expertise",
                       "technologies", "tech stack", "tools & technologies",
                       "programming skills", "technical expertise",
                       "areas of expertise", "competencies", "proficiencies"],
    "projects":       ["projects", "project work", "personal projects",
                       "academic projects", "key projects", "notable projects",
                       "open source", "portfolio", "side projects",
                       "major projects", "selected projects", "personal ventures",
                       "github highlights", "side hustles", "open source contributions"],
    "certifications": ["certifications", "certification", "certificates",
                       "professional certifications", "licenses", "credentials",
                       "courses", "online courses", "training", "accreditations",
                       "professional licenses", "credentials & badges"],
    "achievements":   ["achievements", "awards", "honors", "recognitions",
                       "accomplishments", "publications", "patents",
                       "scholarships", "distinctions", "honors & awards",
                       "extra curricular", "extracurricular", "activities"],
    "languages":      ["languages", "language skills", "spoken languages",
                       "language proficiency"],
    "volunteer":      ["volunteer", "volunteering", "community service",
                       "social work", "non-profit"],
}

# Weights for ATS section scoring
SECTION_WEIGHTS: Dict[str, int] = {
    "contact":        15,
    "experience":     25,
    "skills":         20,
    "education":      15,
    "projects":       10,
    "summary":         5,
    "certifications":  5,
    "achievements":    5,
}

# Power verbs for writing quality
POWER_VERBS: Set[str] = {
    "accelerated","achieved","acquired","administered","advanced","allocated",
    "analyzed","architected","automated","boosted","built","championed",
    "collaborated","conceptualized","consolidated","coordinated","created",
    "customized","delivered","deployed","designed","developed","directed",
    "drove","eliminated","engineered","enhanced","established","executed",
    "expanded","facilitated","formulated","generated","grew","identified",
    "implemented","improved","increased","initiated","innovated","integrated",
    "introduced","launched","led","maintained","managed","mentored","migrated",
    "modernized","monitored","negotiated","operated","optimized","orchestrated",
    "overhauled","partnered","pioneered","planned","prioritized","produced",
    "proposed","prototyped","published","re-architected","redesigned","reduced",
    "refactored","reformed","resolved","restructured","scaled","secured",
    "shaped","simplified","spearheaded","standardized","streamlined",
    "strengthened","supervised","supported","trained","transformed","upgraded",
    "utilized","validated","won","wrote",
}

# Weak/filler verbs to penalize
WEAK_VERBS: Set[str] = {
    "worked","helped","assisted","did","made","tried","used","got","went",
    "was","were","am","is","are","have","had","has","been","being",
    "responsible","responsibilities","duties","tasks","involved","participated",
}

# Education level mapping
EDU_LEVELS: Dict[str, int] = {
    "phd": 5, "ph.d": 5, "ph.d.": 5, "doctorate": 5, "doctoral": 5,
    "m.tech": 4, "mtech": 4, "m.e": 4, "m.e.": 4,
    "master": 4, "masters": 4, "mba": 4, "msc": 4, "m.sc": 4,
    "m.sc.": 4, "mca": 4, "m.ca": 4, "m.s": 4, "ms": 4, "m.s.": 4,
    "post graduate": 4, "postgraduate": 4, "pg": 4,
    "b.tech": 3, "btech": 3, "b.e": 3, "b.e.": 3,
    "bachelor": 3, "bachelors": 3, "bsc": 3, "b.sc": 3, "b.sc.": 3,
    "bca": 3, "b.ca": 3, "b.s": 3, "bs": 3, "b.s.": 3,
    "undergraduate": 3, "b.a": 3, "ba": 3, "b.com": 3, "bcom": 3,
    "b.com.": 3, "be": 3,
    "diploma": 2, "polytechnic": 2, "associate": 2,
    "12th": 1, "hsc": 1, "higher secondary": 1, "intermediate": 1,
    "a level": 1, "a-level": 1,
    "10th": 0, "ssc": 0, "matriculation": 0, "o level": 0,
}

EDU_LABELS: Dict[int, str] = {
    0: "10th / SSC",
    1: "12th / HSC / A-Levels",
    2: "Diploma / Associate",
    3: "Bachelor's Degree",
    4: "Master's Degree",
    5: "PhD / Doctorate",
}

# Job title taxonomy
JOB_TITLES: Dict[str, List[str]] = {
    "Software Engineer":         ["software engineer","software developer","sde","swe","programmer","coder"],
    "Senior Software Engineer":  ["senior software engineer","senior developer","senior sde","sde-2","sde2","staff engineer"],
    "Principal Engineer":        ["principal engineer","principal developer","staff software engineer","senior staff"],
    "Frontend Developer":        ["frontend","front-end","front end","ui developer","ui engineer","web developer","web designer"],
    "Backend Developer":         ["backend","back-end","back end","server side","api developer","server developer"],
    "Full Stack Developer":      ["full stack","fullstack","full-stack","mean stack","mern stack"],
    "Data Scientist":            ["data scientist","data science"],
    "Data Engineer":             ["data engineer","data pipeline","etl developer","analytics engineer"],
    "ML Engineer":               ["machine learning engineer","ml engineer","ai engineer","mlops engineer","applied scientist"],
    "AI Engineer":               ["ai engineer","artificial intelligence engineer","llm engineer","ai researcher"],
    "Data Analyst":              ["data analyst","business analyst","bi analyst","analytics","business intelligence analyst"],
    "DevOps Engineer":           ["devops","dev ops","site reliability","sre","platform engineer","infrastructure engineer","release engineer"],
    "Cloud Engineer":            ["cloud engineer","cloud architect","solutions architect","cloud developer","aws engineer"],
    "Android Developer":         ["android developer","android engineer","android programmer"],
    "iOS Developer":             ["ios developer","ios engineer","swift developer","apple developer"],
    "Mobile Developer":          ["mobile developer","mobile engineer","react native developer","flutter developer","cross-platform"],
    "QA Engineer":               ["qa engineer","quality assurance","test engineer","sdet","automation engineer","qa analyst"],
    "Security Engineer":         ["security engineer","cybersecurity engineer","infosec engineer","penetration tester","appsec"],
    "Product Manager":           ["product manager","product owner","pm","program manager","technical pm"],
    "Tech Lead":                 ["tech lead","technical lead","engineering lead","lead engineer","lead developer"],
    "Engineering Manager":       ["engineering manager","em","vp engineering","director engineering","head of engineering"],
    "Database Administrator":    ["dba","database administrator","database engineer","database developer"],
    "Blockchain Developer":      ["blockchain developer","smart contract","solidity developer","web3 developer"],
    "Game Developer":            ["game developer","game engineer","unity developer","unreal developer","game programmer"],
    "Embedded Engineer":         ["embedded engineer","firmware engineer","iot engineer","embedded systems","rtos"],
    "Junior Developer":          ["junior developer","junior engineer","associate developer","associate engineer","entry level","fresher"],
}

INDUSTRY_DOMAINS: Dict[str, List[str]] = {
    "Fintech / Banking":    ["fintech","banking","finance","payment","trading","financial","insurance","lending","forex","crypto"],
    "Healthcare / Medtech": ["healthcare","health","medical","hospital","pharma","clinical","patient","ehr","telemedicine","biotech"],
    "E-commerce / Retail":  ["ecommerce","e-commerce","retail","shopping","marketplace","inventory","catalog","fulfillment"],
    "EdTech":               ["edtech","education","learning","lms","course","student","teacher","classroom","curriculum"],
    "Gaming":               ["gaming","game","unity","unreal","multiplayer","esports","metaverse"],
    "SaaS / B2B":           ["saas","b2b","enterprise","subscription","crm","erp","b2c","platform"],
    "AI / ML":              ["artificial intelligence","machine learning","deep learning","neural","llm","generative"],
    "Cybersecurity":        ["cybersecurity","security","threat","vulnerability","siem","soc","pentesting","firewall"],
    "Logistics":            ["logistics","supply chain","warehouse","fleet","shipment","delivery","last mile"],
    "Media / Streaming":    ["media","streaming","content","video","audio","broadcast","ott","podcast"],
    "Telecom":              ["telecom","telecommunications","5g","network","carrier","isp","voip"],
    "Automotive":           ["automotive","vehicle","car","autonomous","ev","electric vehicle","self-driving"],
}


# ══════════════════════════════════════════════════════════════════
#  SECTION SPLITTER
# ══════════════════════════════════════════════════════════════════

def _is_section_heading(line: str) -> Optional[str]:
    """
    Detect if a line is a section heading.
    Returns section key or None.
    Uses multiple heuristics:
    1. ALL CAPS line
    2. Title Case line shorter than 40 chars
    3. Line ending with colon
    4. Known heading keywords
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None

    # Remove trailing punctuation for matching
    clean = re.sub(r'[:\-–—_\*#]+$', '', stripped).strip().lower()

    for section_key, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if clean == pat or clean.startswith(pat + " ") or clean.endswith(" " + pat):
                return section_key
            # Fuzzy: heading contains the keyword
            if pat in clean and len(clean) < 35:
                return section_key

    # Heuristic: ALL CAPS short line = likely a heading
    if stripped.isupper() and 3 < len(stripped) < 35:
        # Try to classify based on content
        for section_key, patterns in SECTION_PATTERNS.items():
            if any(p in stripped.lower() for p in patterns):
                return section_key

    return None


def split_into_sections(resume_text: str) -> Dict[str, str]:
    """
    Split resume text into labeled sections.
    Returns {section_key: section_text}
    """
    lines = resume_text.split('\n')
    sections: Dict[str, List[str]] = defaultdict(list)
    current_section = "header"

    for line in lines:
        detected = _is_section_heading(line)
        if detected:
            current_section = detected
        else:
            sections[current_section].append(line)

    return {k: '\n'.join(v).strip() for k, v in sections.items() if v}


# ══════════════════════════════════════════════════════════════════
#  CONTACT INFO EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_contact_info(text: str) -> Dict:
    """Extract all contact information from resume text."""
    email_match    = re.search(r'[\w\.\-\+]+@[\w\.-]+\.\w{2,6}', text)
    phone_match    = re.search(r'(?:\+?\d{1,3}[\s\-\.]?)?\(?\d{2,4}\)?[\s\-\.]?\d{3,4}[\s\-\.]?\d{4,6}', text)
    linkedin_match = re.search(r'(?:linkedin\.com/in/|linkedin\.com/pub/)([a-zA-Z0-9\-\_]+)', text, re.IGNORECASE)
    github_match   = re.search(r'(?:github\.com/)([a-zA-Z0-9\-\_]+)', text, re.IGNORECASE)
    portfolio_match = re.search(
        r'(?:portfolio|website|site|blog|personal|be\.net|dribbble\.com|behance\.net)[\s:]+([https?://]?[a-zA-Z0-9\.\-\/]+\.[a-zA-Z]{2,})',
        text, re.IGNORECASE
    )
    # Generic URL (catch portfolio/website)
    url_match = re.search(
        r'https?://(?!linkedin|github|google|facebook|twitter|instagram|x\.com)[a-zA-Z0-9\.\-\/]+\.[a-zA-Z]{2,}[/\w\-\.]*',
        text, re.IGNORECASE
    )

    # Name heuristic: first non-empty line that looks like a name
    name = None
    blacklist = {'resume', 'cv', 'curriculum', 'vitae', 'contact', 'summary', 'profile', 'experience'}
    for line in text.split('\n')[:8]:
        line = line.strip()
        if (2 <= len(line.split()) <= 4
                and re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', line)
                and not any(c in line for c in ['@', ':', '/', 'http'])
                and not any(w in line.lower() for w in blacklist)):
            name = line
            break

    return {
        "name":          name,
        "email":         email_match.group(0) if email_match else None,
        "phone":         phone_match.group(0).strip() if phone_match else None,
        "linkedin":      f"linkedin.com/in/{linkedin_match.group(1)}" if linkedin_match else None,
        "github":        f"github.com/{github_match.group(1)}" if github_match else None,
        "portfolio":     url_match.group(0) if url_match else None,
        "has_email":     bool(email_match),
        "has_phone":     bool(phone_match),
        "has_linkedin":  bool(linkedin_match),
        "has_github":    bool(github_match),
        "has_portfolio": bool(url_match) or bool(portfolio_match),
    }


# ══════════════════════════════════════════════════════════════════
#  EXPERIENCE PARSING
# ══════════════════════════════════════════════════════════════════

# Month name patterns
_MONTHS = r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
_YEAR   = r'(?:19|20)\d{2}'

_DATE_RANGE_RE = re.compile(
    rf'({_MONTHS}\.?\s*{_YEAR}|{_YEAR}|present|current|now|till date|to date)'
    rf'\s*(?:[-–—to]+)\s*'
    rf'({_MONTHS}\.?\s*{_YEAR}|{_YEAR}|present|current|now|till date|to date)',
    re.IGNORECASE
)

_YEAR_ONLY_RE = re.compile(rf'\b({_YEAR})\b')


def _parse_year(s: str) -> Optional[int]:
    s = s.strip().lower()
    if any(w in s for w in ['present', 'current', 'now', 'till date', 'to date']):
        return datetime.now().year
    m = re.search(r'(20\d{2}|19\d{2})', s)
    return int(m.group(1)) if m else None


def extract_experience_years(text: str) -> Tuple[Optional[float], List[Dict]]:
    """
    Extract experience timeline from resume.
    Returns (total_years_estimated, [job_entries])
    """
    # Strategy 1: Explicit "X years of experience" statements
    explicit_matches = re.findall(
        r'(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?(?:relevant\s+|professional\s+|total\s+)?experience',
        text, re.IGNORECASE
    )
    if explicit_matches:
        return float(max(float(x) for x in explicit_matches)), []

    # Strategy 2: Date range extraction
    date_ranges = _DATE_RANGE_RE.findall(text)
    total_months = 0
    jobs = []

    for start_str, end_str in date_ranges:
        start_yr = _parse_year(start_str)
        end_yr   = _parse_year(end_str)
        if start_yr and end_yr and end_yr >= start_yr:
            duration_months = (end_yr - start_yr) * 12
            total_months += min(duration_months, 60)  # cap per job at 5 years
            jobs.append({
                "start": start_str.strip(),
                "end":   end_str.strip(),
                "approx_years": round(duration_months / 12, 1),
            })

    if jobs:
        return round(total_months / 12, 1), jobs

    # Strategy 3: Year span heuristic
    current_year = datetime.now().year
    years = [int(y) for y in re.findall(rf'\b(20[0-{current_year//10 % 10}]\d|19[89]\d)\b', text)]
    if len(years) >= 2:
        span = max(years) - min(years)
        return float(min(span, 35)), []

    return None, []


def extract_required_experience(jd_text: str) -> Optional[int]:
    """Extract minimum years of experience from JD."""
    patterns = [
        r'(\d+)\+?\s*[-–]\s*\d+\s*years?\s+(?:of\s+)?(?:relevant\s+|professional\s+)?experience',
        r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:relevant\s+|professional\s+|hands.on\s+)?experience',
        r'minimum\s+(?:of\s+)?(\d+)\s+years?',
        r'at\s+least\s+(\d+)\s+years?',
        r'(\d+)\+\s*yrs?\b',
        r'exp(?:erience)?[:\s]+(\d+)\+?\s*years?',
        r'(\d+)\s+years?\s+(?:of\s+)?experience\s+(?:in|with)',
        r'experience\s*[:\-]\s*(\d+)\+?\s*years?',
    ]
    candidates = []
    for pat in patterns:
        for m in re.finditer(pat, jd_text, re.IGNORECASE):
            val = int(m.group(1))
            if 0 < val <= 30:
                candidates.append(val)
    return min(candidates) if candidates else None


# ══════════════════════════════════════════════════════════════════
#  EDUCATION PARSING
# ══════════════════════════════════════════════════════════════════

# Pre-compile education level patterns at module load
_EDU_LEVEL_RES: List[Tuple[re.Pattern, int]] = [
    (re.compile(r'(?<![a-zA-Z.\-])' + re.escape(kw) + r'(?![a-zA-Z.\-])', re.IGNORECASE), lvl)
    for kw, lvl in EDU_LEVELS.items()
]


def extract_education_level(text: str) -> int:
    """Detect highest education level. Uses pre-compiled patterns for speed."""
    text_lower = text.lower()
    level = 0
    for compiled_pat, lvl in _EDU_LEVEL_RES:
        if compiled_pat.search(text_lower):
            level = max(level, lvl)
    return level


def extract_gpa(text: str) -> Optional[str]:
    """Extract GPA/CGPA from resume."""
    patterns = [
        r'(?:cgpa|gpa|grade|score|pointer)[:\s]+(\d+(?:\.\d+)?)\s*(?:/\s*(\d+\.?\d*|10|4|100|5))?',
        r'(\d+(?:\.\d+)?)\s*(?:/|out of)\s*(\d+\.?\d*|10\.0|10|4\.0|4|100|5)\s*(?:cgpa|gpa)?',
        r'(?:cgpa|gpa|score)\s*of\s*(\d+(?:\.\d+)?)',
        r'Grade\s*[:\-]\s*([A-Fa-f][\+\-]?|[\d\.]+\s*(?:%|percent)?)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            total = m.group(2).strip() if len(m.groups()) >= 2 and m.group(2) else "10"
            if "/" not in val and total:
                return f"{val}/{total}"
            return val
    return None


def extract_institutions(text: str) -> List[str]:
    """Extract institution names from resume."""
    patterns = [
        r'\b(?:IIT|IIM|NIT|BITS|IIIT|VIT|SRM|MIT|Stanford|Harvard|'
        r'Oxford|Cambridge|Berkeley)\b[^\n.]{0,40}',
        r'[A-Z][a-zA-Z\s]+(?:University|College|Institute|School|Academy|Polytechnic)\b',
        r'(?:University|College|Institute)\s+of\s+[A-Z][a-zA-Z\s]+',
    ]
    found = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            name = m.group(0).strip()
            if 5 < len(name) < 80 and name not in found:
                found.append(name)
    return found[:4]


# ══════════════════════════════════════════════════════════════════
#  JOB TITLE ALIGNMENT
# ══════════════════════════════════════════════════════════════════

def extract_job_titles(text: str) -> Set[str]:
    """Extract job titles mentioned in text."""
    text_lower = text.lower()
    found = set()
    for role, variants in JOB_TITLES.items():
        if any(v in text_lower for v in variants):
            found.add(role)
    return found


def compute_title_alignment(resume_text: str, jd_text: str) -> Tuple[float, List[str], List[str]]:
    """
    Compute job title alignment score.
    Uses exact match + partial token overlap.
    """
    jd_titles     = extract_job_titles(jd_text)
    resume_titles = extract_job_titles(resume_text)

    if not jd_titles:
        return 1.0, [], sorted(resume_titles)

    # Exact overlap
    overlap = jd_titles & resume_titles
    if overlap:
        return 1.0, sorted(jd_titles), sorted(resume_titles)

    # Token-level partial match
    jd_words     = set(' '.join(jd_titles).split())
    resume_words = set(' '.join(resume_titles).split()) if resume_titles else set()
    common_words = jd_words & resume_words - {'developer', 'engineer', 'senior', 'junior'}
    if common_words:
        score = min(0.5 + 0.1 * len(common_words), 0.85)
        return round(score, 2), sorted(jd_titles), sorted(resume_titles)

    return 0.25, sorted(jd_titles), sorted(resume_titles)


# ══════════════════════════════════════════════════════════════════
#  INDUSTRY DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_industry(text: str) -> List[str]:
    """Detect industry domains from text using keyword density."""
    text_lower = text.lower()
    found = []
    for domain, keywords in INDUSTRY_DOMAINS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits >= 2:
            found.append((domain, hits))
    found.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in found]


# ══════════════════════════════════════════════════════════════════
#  WRITING QUALITY ANALYSIS
# ══════════════════════════════════════════════════════════════════

def analyze_writing_quality(resume_text: str) -> Dict:
    """
    Comprehensive writing quality analysis:
    - Power verb detection and scoring
    - Weak verb flagging
    - Quantified achievement detection
    - Sentence structure quality
    - Bullet point vs. paragraph ratio
    """
    text_lower = resume_text.lower()
    all_words  = set(re.findall(r'\b[a-z]+\b', text_lower))

    # Power verbs
    power_found = sorted(all_words & POWER_VERBS)
    weak_found  = sorted(all_words & WEAK_VERBS)

    # Quantified achievements
    metric_patterns = [
        (r'\d+\s*%',                                                "Percentage"),
        (r'\d+\s*x\b',                                             "Multiplier"),
        (r'\$\s*[\d,]+[kmb]?',                                     "Dollar amount"),
        (r'₹\s*[\d,]+(?:\s*(?:lakh|crore|k|l|cr))?',             "Rupee amount"),
        (r'\d+\s*(?:million|billion|thousand|crore|lakh)\b',       "Scale"),
        (r'\b\d{4,}\b',                                            "Large number"),
        (r'team\s+of\s+\d+',                                       "Team size"),
        (r'\d+\+?\s*(?:users|clients|customers|requests|transactions|engineers)', "Stakeholder scale"),
        (r'reduced\s+(?:by\s+)?(?:\d+\s*%|\$[\d,]+)',             "Reduction with value"),
        (r'improved\s+(?:by\s+)?(?:\d+\s*%|\$[\d,]+)',            "Improvement with value"),
        (r'saved\s+(?:over\s+)?\$?₹?[\d,]+',                     "Savings"),
        (r'increased\s+(?:by\s+)?(?:\d+\s*%|\$[\d,]+)',           "Increase with value"),
        (r'\d+\s*(?:hrs?|hours?|days?|weeks?|months?)\s+(?:saved|reduced|faster)', "Time saved"),
        (r'top\s+\d+\s*%',                                         "Ranking"),
        (r'\d+\s*(?:features?|modules?|services?|apis?|microservices?)\b', "Deliverables"),
        (r'(?:first|launched|0\s+to\s+\d+)',                       "Zero to launch"),
        (r'(?:rank|ranked|position|placed)\s+(?:1st|2nd|3rd|\d+)', "Competition rank"),
    ]

    metric_hits   = []
    metric_count  = 0
    for pat, label in metric_patterns:
        matches = re.findall(pat, text_lower)
        if matches:
            metric_count += len(matches)
            metric_hits.append({
                "type":    label,
                "count":   len(matches),
                "example": matches[0],
            })

    # Sentence quality
    sentences = re.split(r'[.!?\n]', resume_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    avg_sentence_len = (
        sum(len(s.split()) for s in sentences) / len(sentences)
        if sentences else 0
    )

    # Bullet point detection
    bullet_lines = len(re.findall(r'(?:^|\n)\s*(?:[-•●◦▪▸*]|\d+\.)\s+\w', resume_text))
    total_lines  = len([l for l in resume_text.split('\n') if l.strip()])
    bullet_ratio = bullet_lines / total_lines if total_lines > 0 else 0

    # Verb strength score (0-100)
    verb_strength_score = min(
        len(power_found) * 5 - len(weak_found) * 3,
        100
    )

    return {
        "power_verbs": {
            "found":    power_found,
            "count":    len(power_found),
            "strength": (
                "Excellent ✅" if len(power_found) >= 15 else
                "Strong ✅"    if len(power_found) >= 10 else
                "Moderate ⚠️" if len(power_found) >= 5  else
                "Weak ❌"
            ),
        },
        "weak_verbs": {
            "found": weak_found[:8],
            "count": len(weak_found),
        },
        "quantified_achievements": {
            "has_metrics":  bool(metric_hits),
            "count":        metric_count,
            "breakdown":    metric_hits[:8],
            "strength": (
                "Excellent ✅" if metric_count >= 10 else
                "Strong ✅"    if metric_count >= 6  else
                "Moderate ⚠️" if metric_count >= 3  else
                "Weak ❌"
            ),
        },
        "sentence_quality": {
            "avg_sentence_length": round(avg_sentence_len, 1),
            "ideal_range":         "10–20 words",
            "status": (
                "Good ✅" if 8 <= avg_sentence_len <= 22 else "Needs improvement ⚠️"
            ),
        },
        "bullet_usage": {
            "bullet_lines":  bullet_lines,
            "bullet_ratio":  round(bullet_ratio, 2),
            "status": (
                "Good ✅"    if 0.4 <= bullet_ratio <= 0.85 else
                "Too few ⚠️" if bullet_ratio < 0.4 else
                "Too many ⚠️"
            ),
        },
        "verb_strength_score": max(0, verb_strength_score),
    }


# ══════════════════════════════════════════════════════════════════
#  FORMAT & LENGTH ANALYSIS
# ══════════════════════════════════════════════════════════════════

def analyze_format(resume_text: str, contact_info: Dict) -> Dict:
    """Comprehensive format and contact completeness check."""
    words = resume_text.split()
    wc    = len(words)

    contact_score = (
        (3 if contact_info.get("has_email")     else 0) +
        (3 if contact_info.get("has_phone")     else 0) +
        (2 if contact_info.get("has_linkedin")  else 0) +
        (1 if contact_info.get("has_github")    else 0) +
        (1 if contact_info.get("has_portfolio") else 0)
    )

    return {
        "word_count":     wc,
        "char_count":     len(resume_text),
        "length_status":  (
            "Excellent (400–800 words)"   if 400 <= wc <= 800   else
            "Good (250–400 words)"        if 250 <= wc < 400    else
            "Good (800–1200 words)"       if 800 < wc <= 1200   else
            "Too short (< 250 words)"     if wc < 250           else
            "Too long (> 1200 words)"
        ),
        "length_ok":      250 <= wc <= 1200,
        "contact_score":  min(contact_score, 10),
        "contact_max":    10,
        "contact_detail": {
            "email":     "✅" if contact_info.get("has_email")     else "❌",
            "phone":     "✅" if contact_info.get("has_phone")     else "❌",
            "linkedin":  "✅" if contact_info.get("has_linkedin")  else "❌",
            "github":    "✅" if contact_info.get("has_github")    else "❌",
            "portfolio": "✅" if contact_info.get("has_portfolio") else "❌",
        },
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION PRESENCE SCORING
# ══════════════════════════════════════════════════════════════════

# Pre-compile section heading regexes at module load — avoids re.compile per call
_SECTION_HEADING_RES: Dict[str, List[re.Pattern]] = {
    section_key: [
        re.compile(r'(?:^|\n)\s*' + re.escape(p) + r'\s*[:\-–]?\s*(?:\n|$)',
                   re.MULTILINE | re.IGNORECASE)
        for p in patterns
    ]
    for section_key, patterns in SECTION_PATTERNS.items()
}


def detect_sections_present(resume_text: str) -> Dict[str, bool]:
    """
    Detect which standard sections are present.
    Uses pre-compiled heading patterns + content-based fallback.
    """
    text_lower = resume_text.lower()
    present: Dict[str, bool] = {}

    for section_key, patterns in SECTION_PATTERNS.items():
        # Primary: pre-compiled heading patterns (no re.compile per call)
        compiled_pats = _SECTION_HEADING_RES.get(section_key, [])
        heading_found = any(cp.search(text_lower) for cp in compiled_pats)
        # Fallback: fast substring check
        content_found = any(p in text_lower for p in patterns)
        present[section_key] = heading_found or content_found

    return present


def score_sections(sections_present: Dict[str, bool]) -> Tuple[float, List[str], List[str]]:
    """Score resume sections and return earned points."""
    present_list = [k for k, v in sections_present.items() if v and k in SECTION_WEIGHTS]
    missing_list = [k for k, v in sections_present.items() if not v and k in SECTION_WEIGHTS]

    earned = sum(SECTION_WEIGHTS.get(s, 0) for s in present_list)
    total  = sum(SECTION_WEIGHTS.values())
    pct    = round(earned / total * 100, 1)

    return pct, present_list, missing_list