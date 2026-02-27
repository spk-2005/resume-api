"""
resume_analyzer.py  ─  v4.0.0
──────────────────────────────
Maximum accuracy ATS engine using:
  ✅ Weighted skill matching      (JD frequency = importance)
  ✅ Semantic skill aliases       (reactjs = react, k8s = kubernetes)
  ✅ Phrase-safe extraction       (won't match 'r' inside 'docker')
  ✅ N-gram JD keyword extraction (catches skills not in our DB)
  ✅ TF-IDF style keyword scoring (rare important words score higher)
  ✅ Experience years parsing     (extracts from both JD and resume)
  ✅ Education level matching     (B.Tech, M.Tech, MBA, PhD)
  ✅ Job title alignment          (role match scoring)
  ✅ Soft skills detection        (leadership, communication, etc.)
  ✅ Industry domain detection    (tech, finance, healthcare, etc.)
  ✅ Resume completeness scoring  (8 standard sections)
  ✅ Writing quality analysis     (power verbs + quantified metrics)
  ✅ Contact info verification    (email, phone, LinkedIn, GitHub)
  ✅ JD match score (separate from ATS) — pure resume vs JD overlap

ATS Score Breakdown (Total = 100 pts):
  ┌──────────────────────────────┬──────────┐
  │ Component                    │ Max Pts  │
  ├──────────────────────────────┼──────────┤
  │ Skill Match (weighted)       │   40     │
  │ JD Keyword Coverage          │   20     │
  │ Section Completeness         │   15     │
  │ Experience Match             │   10     │
  │ Format & Contact             │   10     │
  │ Writing Quality              │    5     │
  └──────────────────────────────┴──────────┘
"""

import re
import math
from collections import Counter
from typing import Optional, Set, Dict, List, Tuple


# ══════════════════════════════════════════════════════════════════
#  SKILL KNOWLEDGE BASE
#  Format: (canonical_name, [surface_forms_to_match])
#  Rules:  Multi-word entries MUST come before their single-word parts
# ══════════════════════════════════════════════════════════════════

SKILL_PATTERNS: List[Tuple[str, List[str]]] = [

    # ── Multi-word skills (order matters — longest first) ──────────
    ("natural language processing", ["natural language processing", "nlp"]),
    ("machine learning",            ["machine learning"]),
    ("deep learning",               ["deep learning"]),
    ("computer vision",             ["computer vision"]),
    ("reinforcement learning",      ["reinforcement learning", "rl"]),
    ("large language models",       ["large language model", "llm", "llms", "large language models"]),
    ("generative ai",               ["generative ai", "gen ai", "genai"]),
    ("data science",                ["data science"]),
    ("data engineering",            ["data engineering"]),
    ("data analysis",               ["data analysis", "data analytics"]),
    ("data visualization",          ["data visualization", "data viz"]),
    ("business intelligence",       ["business intelligence", "bi"]),
    ("big data",                    ["big data"]),
    ("feature engineering",         ["feature engineering"]),
    ("model deployment",            ["model deployment", "model serving"]),
    ("react native",                ["react native"]),
    ("next.js",                     ["next.js", "nextjs", "next js"]),
    ("vue.js",                      ["vue.js", "vuejs", "vue js"]),
    ("node.js",                     ["node.js", "nodejs", "node js"]),
    ("express.js",                  ["express.js", "expressjs", "express js"]),
    ("spring boot",                 ["spring boot"]),
    ("ruby on rails",               ["ruby on rails", "rails"]),
    ("asp.net",                     ["asp.net", "asp net", "dotnet", ".net"]),
    ("amazon web services",         ["amazon web services", "aws"]),
    ("microsoft azure",             ["microsoft azure", "azure"]),
    ("google cloud",                ["google cloud", "gcp", "google cloud platform"]),
    ("github actions",              ["github actions", "gha"]),
    ("gitlab ci",                   ["gitlab ci", "gitlab ci/cd"]),
    ("apache spark",                ["apache spark", "pyspark"]),
    ("apache kafka",                ["apache kafka", "kafka"]),
    ("apache airflow",              ["apache airflow", "airflow"]),
    ("apache hadoop",               ["apache hadoop", "hadoop"]),
    ("ci/cd",                       ["ci/cd", "ci cd", "continuous integration", "continuous deployment", "continuous delivery"]),
    ("rest api",                    ["rest api", "restful api", "restful"]),
    ("system design",               ["system design"]),
    ("high level design",           ["high level design", "hld"]),
    ("low level design",            ["low level design", "lld"]),
    ("design patterns",             ["design patterns"]),
    ("solid principles",            ["solid principles", "solid"]),
    ("unit testing",                ["unit testing", "unit tests"]),
    ("integration testing",         ["integration testing"]),
    ("test driven development",     ["test driven development", "tdd", "test-driven"]),
    ("behavior driven development", ["behavior driven development", "bdd"]),
    ("code review",                 ["code review"]),
    ("object oriented programming", ["object oriented", "oop", "object-oriented programming"]),
    ("functional programming",      ["functional programming"]),
    ("scikit-learn",                ["scikit-learn", "sklearn", "scikit learn"]),
    ("power bi",                    ["power bi", "powerbi"]),
    ("google analytics",            ["google analytics"]),
    ("problem solving",             ["problem solving", "problem-solving"]),
    ("time management",             ["time management"]),
    ("critical thinking",           ["critical thinking"]),
    ("cross functional",            ["cross functional", "cross-functional"]),
    ("stakeholder management",      ["stakeholder management"]),
    ("project management",          ["project management"]),

    # ── Programming Languages ──────────────────────────────────────
    ("python",         ["python"]),
    ("javascript",     ["javascript", "js"]),
    ("typescript",     ["typescript", "ts"]),
    ("java",           ["java"]),
    ("c++",            ["c++", "cpp"]),
    ("c#",             ["c#", "csharp", "c sharp"]),
    ("golang",         ["golang", "go lang"]),
    ("rust",           ["rust"]),
    ("ruby",           ["ruby"]),
    ("php",            ["php"]),
    ("swift",          ["swift"]),
    ("kotlin",         ["kotlin"]),
    ("scala",          ["scala"]),
    ("r",              ["\\br\\b"]),
    ("matlab",         ["matlab"]),
    ("bash",           ["bash", "shell scripting", "shell script"]),
    ("powershell",     ["powershell"]),
    ("dart",           ["dart"]),
    ("elixir",         ["elixir"]),
    ("haskell",        ["haskell"]),
    ("clojure",        ["clojure"]),
    ("julia",          ["julia"]),
    ("lua",            ["lua"]),
    ("assembly",       ["assembly", "asm"]),

    # ── Frontend ───────────────────────────────────────────────────
    ("react",          ["react", "reactjs", "react.js"]),
    ("angular",        ["angular", "angularjs"]),
    ("vue",            ["vue", "vuejs"]),
    ("svelte",         ["svelte"]),
    ("html",           ["html", "html5"]),
    ("css",            ["css", "css3"]),
    ("jquery",         ["jquery"]),
    ("bootstrap",      ["bootstrap"]),
    ("tailwind",       ["tailwind", "tailwindcss"]),
    ("sass",           ["sass", "scss"]),
    ("webpack",        ["webpack"]),
    ("vite",           ["vite"]),
    ("redux",          ["redux", "redux toolkit"]),
    ("graphql",        ["graphql"]),
    ("webrtc",         ["webrtc"]),

    # ── Backend ────────────────────────────────────────────────────
    ("fastapi",        ["fastapi"]),
    ("django",         ["django"]),
    ("flask",          ["flask"]),
    ("express",        ["express"]),
    ("spring",         ["spring", "spring framework"]),
    ("laravel",        ["laravel"]),
    ("grpc",           ["grpc"]),
    ("graphql",        ["graphql"]),
    ("celery",         ["celery"]),
    ("rabbitmq",       ["rabbitmq", "rabbit mq"]),

    # ── Data & AI ──────────────────────────────────────────────────
    ("pandas",         ["pandas"]),
    ("numpy",          ["numpy"]),
    ("tensorflow",     ["tensorflow", "tf"]),
    ("pytorch",        ["pytorch", "torch"]),
    ("keras",          ["keras"]),
    ("xgboost",        ["xgboost"]),
    ("lightgbm",       ["lightgbm"]),
    ("catboost",       ["catboost"]),
    ("huggingface",    ["huggingface", "hugging face", "transformers"]),
    ("langchain",      ["langchain", "lang chain"]),
    ("openai",         ["openai", "open ai", "chatgpt api", "gpt-4", "gpt4"]),
    ("dbt",            ["dbt"]),
    ("tableau",        ["tableau"]),
    ("looker",         ["looker"]),
    ("mlflow",         ["mlflow"]),
    ("mlops",          ["mlops", "ml ops"]),
    ("spark",          ["spark"]),
    ("flink",          ["flink", "apache flink"]),

    # ── Cloud & DevOps ─────────────────────────────────────────────
    ("docker",         ["docker"]),
    ("kubernetes",     ["kubernetes", "k8s"]),
    ("terraform",      ["terraform"]),
    ("ansible",        ["ansible"]),
    ("jenkins",        ["jenkins"]),
    ("linux",          ["linux", "ubuntu", "centos", "debian", "rhel"]),
    ("nginx",          ["nginx"]),
    ("apache",         ["apache", "apache http"]),
    ("prometheus",     ["prometheus"]),
    ("grafana",        ["grafana"]),
    ("datadog",        ["datadog"]),
    ("splunk",         ["splunk"]),
    ("helm",           ["helm"]),
    ("istio",          ["istio"]),
    ("argocd",         ["argocd", "argo cd"]),
    ("pulumi",         ["pulumi"]),
    ("vault",          ["vault", "hashicorp vault"]),

    # ── Databases ──────────────────────────────────────────────────
    ("postgresql",     ["postgresql", "postgres", "psql"]),
    ("mysql",          ["mysql"]),
    ("mongodb",        ["mongodb", "mongo"]),
    ("redis",          ["redis"]),
    ("elasticsearch",  ["elasticsearch", "elastic search", "opensearch"]),
    ("cassandra",      ["cassandra", "apache cassandra"]),
    ("sqlite",         ["sqlite"]),
    ("dynamodb",       ["dynamodb", "dynamo"]),
    ("firebase",       ["firebase", "firestore"]),
    ("neo4j",          ["neo4j"]),
    ("oracle",         ["oracle", "oracle db"]),
    ("sql server",     ["sql server", "mssql", "microsoft sql"]),
    ("snowflake",      ["snowflake"]),
    ("bigquery",       ["bigquery", "big query"]),
    ("redshift",       ["redshift", "amazon redshift"]),
    ("pinecone",       ["pinecone"]),
    ("weaviate",       ["weaviate"]),
    ("clickhouse",     ["clickhouse"]),
    ("supabase",       ["supabase"]),

    # ── Mobile ─────────────────────────────────────────────────────
    ("flutter",        ["flutter"]),
    ("android",        ["android"]),
    ("ios",            ["ios", "iphone", "ipad"]),
    ("swift",          ["swift"]),
    ("kotlin",         ["kotlin"]),

    # ── Security ───────────────────────────────────────────────────
    ("oauth",          ["oauth", "oauth2", "oauth 2.0"]),
    ("jwt",            ["jwt", "json web token"]),
    ("ssl/tls",        ["ssl", "tls", "ssl/tls", "https"]),
    ("cybersecurity",  ["cybersecurity", "cyber security", "infosec", "information security"]),
    ("penetration testing", ["penetration testing", "pen testing", "pentesting"]),
    ("sso",            ["sso", "single sign-on", "saml"]),

    # ── Tools & Practices ─────────────────────────────────────────
    ("git",            ["git"]),
    ("github",         ["github"]),
    ("gitlab",         ["gitlab"]),
    ("bitbucket",      ["bitbucket"]),
    ("jira",           ["jira"]),
    ("confluence",     ["confluence"]),
    ("agile",          ["agile"]),
    ("scrum",          ["scrum"]),
    ("kanban",         ["kanban"]),
    ("devops",         ["devops", "dev ops"]),
    ("microservices",  ["microservices", "microservice", "micro services"]),
    ("serverless",     ["serverless", "lambda", "cloud functions"]),
    ("websockets",     ["websockets", "websocket", "ws"]),
    ("message queue",  ["message queue", "mq", "event driven"]),

    # ── Office & Analytics ────────────────────────────────────────
    ("excel",          ["excel", "ms excel", "microsoft excel"]),
    ("sql",            ["sql", "mysql", "t-sql", "pl/sql"]),
    ("nosql",          ["nosql", "no-sql"]),

    # ── Soft Skills ───────────────────────────────────────────────
    ("communication",  ["communication", "verbal communication", "written communication"]),
    ("leadership",     ["leadership", "team lead", "tech lead", "people management"]),
    ("teamwork",       ["teamwork", "team player", "team work", "collaboration"]),
    ("mentoring",      ["mentoring", "mentorship", "coaching"]),
    ("presentation",   ["presentation", "public speaking"]),
]

# Pre-compile all regex patterns for speed
_SKILL_REGEXES: Dict[str, List[re.Pattern]] = {}
for _canonical, _forms in SKILL_PATTERNS:
    _pats = []
    for _form in _forms:
        if _form.startswith("\\b"):
            _pats.append(re.compile(_form, re.IGNORECASE))
        else:
            _pats.append(re.compile(
                r'(?<![a-zA-Z0-9\+\#\-])' + re.escape(_form) + r'(?![a-zA-Z0-9\+\#\-])',
                re.IGNORECASE
            ))
    _SKILL_REGEXES[_canonical] = _pats

# ── Stopwords for keyword extraction ──────────────────────────────
STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "this",
    "that", "these", "those", "it", "its", "we", "you", "your", "our",
    "their", "they", "he", "she", "his", "her", "who", "which", "what",
    "when", "where", "how", "why", "not", "no", "nor", "so", "yet",
    "both", "either", "neither", "each", "any", "all", "more", "most",
    "other", "such", "than", "then", "also", "just", "only", "very",
    "well", "good", "great", "new", "work", "team", "role", "job",
    "position", "candidate", "experience", "ability", "skills", "strong",
    "excellent", "required", "preferred", "plus", "including", "related",
    "using", "working", "looking", "seeking", "responsible", "knowledge",
    "understanding", "familiarity", "proficiency", "proficient", "ability",
    "able", "make", "use", "used", "get", "set", "build", "built",
    "year", "years", "month", "months", "day", "days", "time", "etc",
    "eg", "ie", "e.g", "i.e", "per", "as", "if", "into", "over",
    "after", "before", "during", "while", "since", "about", "above",
    "across", "between", "through", "without", "within", "along",
    "following", "across", "behind", "beyond", "plus", "except",
    "up", "out", "around", "down", "off", "again", "further", "once",
}


# ══════════════════════════════════════════════════════════════════
#  SECTIONS KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════

ATS_SECTIONS: Dict[str, List[str]] = {
    "Contact Info":     ["email", "phone", "linkedin", "github", "mobile", "contact", "@", "tel:", "mailto:"],
    "Summary":          ["summary", "objective", "profile", "about me", "overview",
                         "career objective", "professional summary", "about", "introduction"],
    "Work Experience":  ["experience", "work experience", "employment", "work history",
                         "professional experience", "internship", "career history",
                         "positions held", "employment history"],
    "Education":        ["education", "academic", "degree", "university", "college",
                         "bachelor", "master", "phd", "b.tech", "m.tech", "b.e", "m.e",
                         "b.sc", "m.sc", "mba", "bca", "mca", "schooling", "qualification"],
    "Skills":           ["skills", "technical skills", "core competencies",
                         "expertise", "technologies", "tech stack", "tools",
                         "competencies", "key skills", "areas of expertise"],
    "Projects":         ["projects", "project work", "portfolio", "open source",
                         "personal projects", "academic projects", "side projects"],
    "Certifications":   ["certification", "certificate", "certified", "credential",
                         "course", "training", "accreditation", "license", "licensure"],
    "Achievements":     ["achievement", "award", "honor", "recognition",
                         "accomplishment", "scholarship", "distinction",
                         "prize", "publication", "patent"],
}

SECTION_WEIGHTS: Dict[str, int] = {
    "Contact Info":    15,
    "Work Experience": 25,
    "Skills":          20,
    "Education":       15,
    "Projects":        10,
    "Summary":          5,
    "Certifications":   5,
    "Achievements":     5,
}


# ══════════════════════════════════════════════════════════════════
#  POWER VERBS
# ══════════════════════════════════════════════════════════════════

POWER_VERBS: Set[str] = {
    "developed", "designed", "built", "implemented", "deployed", "led",
    "managed", "created", "architected", "optimized", "improved", "reduced",
    "increased", "delivered", "launched", "automated", "integrated", "migrated",
    "scaled", "mentored", "collaborated", "analyzed", "engineered", "maintained",
    "resolved", "streamlined", "established", "accelerated", "spearheaded",
    "orchestrated", "pioneered", "revamped", "negotiated", "coordinated",
    "generated", "transformed", "enhanced", "restructured", "formulated",
    "conceptualized", "executed", "facilitated", "identified", "initiated",
    "introduced", "partnered", "produced", "proposed", "secured", "shaped",
    "simplified", "standardized", "strengthened", "succeeded", "supervised",
    "supported", "trained", "upgraded", "utilized", "validated", "wrote",
}


# ══════════════════════════════════════════════════════════════════
#  EDUCATION LEVELS
# ══════════════════════════════════════════════════════════════════

EDU_LEVELS: Dict[str, int] = {
    "phd": 5, "ph.d": 5, "doctorate": 5, "doctoral": 5,
    "m.tech": 4, "mtech": 4, "m.e": 4, "me": 4,
    "master": 4, "masters": 4, "mba": 4, "msc": 4, "m.sc": 4,
    "mca": 4, "m.ca": 4, "m.s": 4, "ms": 4,
    "b.tech": 3, "btech": 3, "b.e": 3, "be": 3,
    "bachelor": 3, "bachelors": 3, "bsc": 3, "b.sc": 3,
    "bca": 3, "b.ca": 3, "b.s": 3, "bs": 3, "undergraduate": 3,
    "b.a": 3, "ba": 3, "b.com": 3, "bcom": 3,
    "diploma": 2, "polytechnic": 2,
    "12th": 1, "hsc": 1, "higher secondary": 1, "intermediate": 1,
    "10th": 0, "ssc": 0, "matriculation": 0,
}

EDU_LABELS: Dict[int, str] = {
    0: "10th / SSC",
    1: "12th / HSC",
    2: "Diploma",
    3: "Bachelor's Degree",
    4: "Master's Degree",
    5: "PhD / Doctorate",
}


# ══════════════════════════════════════════════════════════════════
#  JOB TITLE TAXONOMY
# ══════════════════════════════════════════════════════════════════

JOB_TITLES: Dict[str, List[str]] = {
    "Software Engineer":         ["software engineer", "software developer", "sde", "swe", "programmer"],
    "Senior Software Engineer":  ["senior software engineer", "senior developer", "senior sde", "sde-2", "sde2"],
    "Frontend Developer":        ["frontend", "front-end", "front end", "ui developer", "ui engineer"],
    "Backend Developer":         ["backend", "back-end", "back end", "server side", "api developer"],
    "Full Stack Developer":      ["full stack", "fullstack", "full-stack"],
    "Data Scientist":            ["data scientist", "data science"],
    "Data Engineer":             ["data engineer", "data pipeline", "etl developer"],
    "ML Engineer":               ["machine learning engineer", "ml engineer", "ai engineer", "mlops engineer"],
    "AI Engineer":               ["ai engineer", "artificial intelligence engineer", "llm engineer"],
    "Data Analyst":              ["data analyst", "business analyst", "bi analyst", "analytics engineer"],
    "DevOps Engineer":           ["devops", "dev ops", "site reliability", "sre", "platform engineer", "infrastructure engineer"],
    "Cloud Engineer":            ["cloud engineer", "cloud architect", "solutions architect", "cloud developer"],
    "Android Developer":         ["android developer", "android engineer"],
    "iOS Developer":             ["ios developer", "ios engineer", "swift developer"],
    "Mobile Developer":          ["mobile developer", "mobile engineer", "react native developer", "flutter developer"],
    "QA Engineer":               ["qa engineer", "quality assurance", "test engineer", "sdet", "automation engineer"],
    "Security Engineer":         ["security engineer", "cybersecurity engineer", "infosec engineer", "penetration tester"],
    "Product Manager":           ["product manager", "product owner", "pm"],
    "Scrum Master":              ["scrum master", "agile coach"],
    "Tech Lead":                 ["tech lead", "technical lead", "engineering lead"],
    "Engineering Manager":       ["engineering manager", "em", "vp engineering"],
    "Database Administrator":    ["dba", "database administrator", "database engineer"],
    "Blockchain Developer":      ["blockchain developer", "smart contract", "solidity"],
    "Game Developer":            ["game developer", "game engineer", "unity developer", "unreal developer"],
    "Embedded Engineer":         ["embedded engineer", "firmware engineer", "iot engineer"],
}


# ══════════════════════════════════════════════════════════════════
#  INDUSTRY DOMAINS
# ══════════════════════════════════════════════════════════════════

INDUSTRY_DOMAINS: Dict[str, List[str]] = {
    "Fintech / Banking":    ["fintech", "banking", "finance", "payment", "trading", "financial", "insurance", "lending"],
    "Healthcare / Medtech": ["healthcare", "health", "medical", "hospital", "pharma", "clinical", "patient", "ehr"],
    "E-commerce / Retail":  ["ecommerce", "e-commerce", "retail", "shopping", "marketplace", "inventory", "catalog"],
    "EdTech":               ["edtech", "education", "learning", "lms", "course", "student", "teacher"],
    "Gaming":               ["gaming", "game", "unity", "unreal", "multiplayer"],
    "SaaS / B2B":           ["saas", "b2b", "enterprise", "subscription", "crm", "erp"],
    "AI / ML":              ["artificial intelligence", "machine learning", "deep learning", "neural", "llm"],
    "Cybersecurity":        ["cybersecurity", "security", "threat", "vulnerability", "siem", "soc"],
    "Logistics":            ["logistics", "supply chain", "warehouse", "fleet", "shipment", "delivery"],
    "Media / Streaming":    ["media", "streaming", "content", "video", "audio", "broadcast"],
}


# ══════════════════════════════════════════════════════════════════
#  SKILL EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_skills(text: str) -> Dict[str, int]:
    """
    Extracts skills from text with occurrence counts.
    Uses boundary-safe regex. Multi-word skills consume spans
    so sub-words don't get double-counted.
    Returns {canonical_skill_name: occurrence_count}
    """
    text_lower = text.lower()
    found: Dict[str, int] = {}
    consumed: List[Tuple[int, int]] = []

    for canonical, _ in SKILL_PATTERNS:
        patterns = _SKILL_REGEXES.get(canonical, [])
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


# ══════════════════════════════════════════════════════════════════
#  JD KEYWORD EXTRACTION  (catches skills not in our DB)
#  Uses TF-IDF inspired scoring to find important keywords
# ══════════════════════════════════════════════════════════════════

def extract_jd_keywords(jd_text: str, top_n: int = 40) -> Dict[str, float]:
    """
    Extract important keywords from JD using TF-IDF inspired scoring.
    - Removes stopwords
    - Weights words by position (earlier = more important in JDs)
    - Weights words by frequency
    - Boosts words in ALL CAPS or Title Case (usually skill names)
    Returns {keyword: importance_score}
    """
    # Split into sentences to get position weights
    sentences = re.split(r'[.\n!?;]', jd_text)
    word_scores: Dict[str, float] = {}
    total_words = 0

    for sent_idx, sentence in enumerate(sentences):
        # Position weight: earlier sentences matter more
        position_weight = 1.0 / (1.0 + sent_idx * 0.05)

        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\+\#\./\-]{1,30}\b', sentence)
        for word in words:
            w_lower = word.lower()
            if w_lower in STOPWORDS or len(w_lower) < 2:
                continue

            # Case boost: PYTHON or Python > python
            case_boost = 1.3 if word[0].isupper() else 1.0
            if word.isupper() and len(word) > 1:
                case_boost = 1.5

            score = position_weight * case_boost
            word_scores[w_lower] = word_scores.get(w_lower, 0) + score
            total_words += 1

    # Also extract 2-grams and 3-grams
    all_words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\+\#\.]{1,20}\b', jd_text.lower())
    filtered  = [w for w in all_words if w not in STOPWORDS and len(w) > 2]

    for i in range(len(filtered) - 1):
        bigram = filtered[i] + " " + filtered[i+1]
        if len(bigram) > 5:
            word_scores[bigram] = word_scores.get(bigram, 0) + 0.8

    for i in range(len(filtered) - 2):
        trigram = filtered[i] + " " + filtered[i+1] + " " + filtered[i+2]
        if len(trigram) > 8:
            word_scores[trigram] = word_scores.get(trigram, 0) + 0.5

    # Sort and return top N
    sorted_kw = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_kw[:top_n])


def jd_keyword_coverage(resume_text: str, jd_keywords: Dict[str, float]) -> Tuple[float, List[str], List[str]]:
    """
    Check what % of important JD keywords appear in the resume.
    Uses weighted coverage (high-importance keywords count more).
    Returns (coverage_percentage, covered_keywords, missing_keywords)
    """
    text_lower = resume_text.lower()
    covered, missing = [], []
    covered_weight = 0.0
    total_weight   = sum(jd_keywords.values()) or 1.0

    for kw, weight in jd_keywords.items():
        # Use word boundary check for single words, substring for phrases
        if " " in kw:
            found = kw in text_lower
        else:
            found = bool(re.search(r'(?<![a-zA-Z0-9])' + re.escape(kw) + r'(?![a-zA-Z0-9])', text_lower))

        if found:
            covered.append(kw)
            covered_weight += weight
        else:
            missing.append(kw)

    pct = round(min((covered_weight / total_weight) * 100, 100.0), 1)
    return pct, covered[:20], missing[:20]


# ══════════════════════════════════════════════════════════════════
#  JD MATCH SCORE  (separate from ATS — pure content similarity)
# ══════════════════════════════════════════════════════════════════

def compute_jd_match_score(
    resume_text: str,
    jd_text:     str,
    matched_skills: List[str],
    total_jd_skills: int,
    kw_coverage: float,
) -> float:
    """
    Pure JD ↔ Resume match score (0–100).
    Measures how much of the JD content is reflected in the resume.

    Formula:
      - Skill overlap:   50% weight
      - Keyword coverage: 30% weight
      - Text similarity:  20% weight (Jaccard on bigrams)
    """
    # Skill overlap component
    skill_score = (len(matched_skills) / max(total_jd_skills, 1)) * 100

    # Keyword coverage component (already computed)
    kw_score = kw_coverage

    # Jaccard similarity on word sets
    def tokenize(t: str) -> Set[str]:
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]{2,}\b', t.lower())
        return set(w for w in words if w not in STOPWORDS)

    r_words = tokenize(resume_text)
    j_words = tokenize(jd_text)
    intersection = r_words & j_words
    union        = r_words | j_words
    jaccard      = (len(intersection) / len(union) * 100) if union else 0

    match_score = (skill_score * 0.50) + (kw_score * 0.30) + (jaccard * 0.20)
    return round(min(match_score, 100.0), 1)


# ══════════════════════════════════════════════════════════════════
#  WEIGHTED SKILL SCORING
# ══════════════════════════════════════════════════════════════════

def weighted_skill_score(
    resume_skills: Dict[str, int],
    jd_skills:     Dict[str, int],
) -> Tuple[float, List[str], List[str], List[str]]:
    """
    Weighted skill match. Skills mentioned more in JD are more important.
    Weight = 0.5 + 0.5 × (count / max_count) → range [0.5, 1.0]
    """
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


# ══════════════════════════════════════════════════════════════════
#  SECTION DETECTION & SCORING
# ══════════════════════════════════════════════════════════════════

def detect_sections(text: str) -> Dict[str, bool]:
    text_lower = text.lower()
    return {
        sec: any(kw in text_lower for kw in kws)
        for sec, kws in ATS_SECTIONS.items()
    }


def score_sections(sections: Dict[str, bool]) -> Tuple[float, List[str], List[str]]:
    present = [s for s, v in sections.items() if v]
    missing = [s for s, v in sections.items() if not v]
    earned  = sum(SECTION_WEIGHTS.get(s, 0) for s in present)
    return float(earned), present, missing


# ══════════════════════════════════════════════════════════════════
#  EXPERIENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════

def extract_required_years(jd_text: str) -> Optional[int]:
    """Extract minimum years of experience required by JD."""
    patterns = [
        r'(\d+)\+?\s*-\s*\d+\s*years?\s+of\s+(?:relevant\s+)?experience',
        r'(\d+)\+?\s*years?\s+of\s+(?:relevant\s+|professional\s+)?experience',
        r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:relevant\s+)?experience',
        r'minimum\s+(?:of\s+)?(\d+)\s+years?',
        r'at\s+least\s+(\d+)\s+years?',
        r'(\d+)\+\s*yrs?\b',
        r'exp(?:erience)?[:\s]+(\d+)\+?\s*years?',
    ]
    candidates = []
    for pat in patterns:
        for m in re.finditer(pat, jd_text, re.IGNORECASE):
            candidates.append(int(m.group(1)))
    return min(candidates) if candidates else None


def estimate_resume_years(resume_text: str) -> Optional[float]:
    """
    Estimate experience from resume by finding year ranges.
    Also detects explicit 'X years of experience' statements.
    """
    # Check explicit statements first
    explicit = re.findall(
        r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience',
        resume_text, re.IGNORECASE
    )
    if explicit:
        return float(max(int(x) for x in explicit))

    # Fall back to year range detection
    years = [int(y) for y in re.findall(r'\b(20[0-2]\d|19[89]\d)\b', resume_text)]
    if not years:
        return None
    span = max(years) - min(years)
    return float(min(span, 40)) if span > 0 else None


def experience_score(resume_text: str, jd_text: str) -> Tuple[float, Optional[int], Optional[float]]:
    required = extract_required_years(jd_text)
    actual   = estimate_resume_years(resume_text)

    if required is None:
        return 0.80, None, actual      # No requirement = neutral, slight positive
    if actual is None:
        return 0.55, required, None   # Can't detect years = slight negative
    if actual >= required:
        # Bonus for significantly exceeding requirement (up to 1.0)
        bonus = min((actual - required) * 0.02, 0.1)
        return min(1.0 + bonus, 1.0), required, actual
    return round(actual / required, 3), required, actual


# ══════════════════════════════════════════════════════════════════
#  EDUCATION ANALYSIS
# ══════════════════════════════════════════════════════════════════

def detect_education_level(text: str) -> int:
    text_lower = text.lower()
    level = 0
    for kw, lvl in EDU_LEVELS.items():
        if re.search(r'(?<![a-zA-Z])' + re.escape(kw) + r'(?![a-zA-Z])', text_lower):
            level = max(level, lvl)
    return level


def detect_institutions(resume_text: str) -> List[str]:
    """Try to extract university/college names from resume."""
    patterns = [
        r'(?:university|college|institute|iit|nit|bits|iiit|vit|srm)\s+(?:of\s+)?[a-zA-Z\s,]+',
        r'[A-Z][a-zA-Z\s]+(?:University|College|Institute|School|Academy)',
    ]
    found = []
    for pat in patterns:
        for m in re.finditer(pat, resume_text, re.IGNORECASE):
            name = m.group(0).strip()
            if 5 < len(name) < 60:
                found.append(name)
    return list(set(found))[:3]


# ══════════════════════════════════════════════════════════════════
#  JOB TITLE ALIGNMENT
# ══════════════════════════════════════════════════════════════════

def extract_job_titles(text: str) -> Set[str]:
    text_lower = text.lower()
    return {role for role, variants in JOB_TITLES.items()
            if any(v in text_lower for v in variants)}


def title_alignment(resume_text: str, jd_text: str) -> Tuple[float, List[str], List[str]]:
    jd_titles     = extract_job_titles(jd_text)
    resume_titles = extract_job_titles(resume_text)

    if not jd_titles:
        return 1.0, [], list(sorted(resume_titles))

    overlap = jd_titles & resume_titles
    if overlap:
        return 1.0, sorted(jd_titles), sorted(resume_titles)

    # Partial match — shared words
    jd_words     = set(" ".join(jd_titles).split())
    resume_words = set(" ".join(resume_titles).split()) if resume_titles else set()
    common       = jd_words & resume_words
    if common:
        return 0.65, sorted(jd_titles), sorted(resume_titles)

    return 0.25, sorted(jd_titles), sorted(resume_titles)


# ══════════════════════════════════════════════════════════════════
#  INDUSTRY DOMAIN DETECTION
# ══════════════════════════════════════════════════════════════════

def detect_industry(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for domain, keywords in INDUSTRY_DOMAINS.items():
        if sum(1 for kw in keywords if kw in text_lower) >= 2:
            found.append(domain)
    return found


# ══════════════════════════════════════════════════════════════════
#  WRITING QUALITY
# ══════════════════════════════════════════════════════════════════

def detect_power_verbs(text: str) -> Tuple[List[str], int]:
    tokens = set(re.findall(r'\b\w+\b', text.lower()))
    found  = sorted(tokens & POWER_VERBS)
    return found, len(found)


def check_metrics(text: str) -> Tuple[bool, int, List[str]]:
    """
    Detect quantified achievements with specific pattern matching.
    Returns (has_metrics, count, examples)
    """
    patterns = [
        (r'\d+\s*%',                                          "percentage"),
        (r'\d+\s*x\b',                                        "multiplier"),
        (r'\$\s*[\d,]+[kmb]?',                                "dollar amount"),
        (r'₹\s*[\d,]+[lkc]?',                                 "rupee amount"),
        (r'\d+\s*(?:million|billion|thousand|k)\b',           "scale"),
        (r'\b\d{4,}\b',                                       "large number"),
        (r'team\s+of\s+\d+',                                  "team size"),
        (r'\d+\+?\s*(?:users|clients|customers|requests|transactions)', "user scale"),
        (r'reduced\s+(?:by\s+)?\d+',                         "reduction"),
        (r'improved\s+(?:by\s+)?\d+',                        "improvement"),
        (r'saved\s+(?:over\s+)?\$?₹?\d+',                   "savings"),
        (r'increased\s+(?:by\s+)?\d+',                       "increase"),
        (r'\d+\s*(?:hrs?|hours?|days?|weeks?)\s+(?:saved|reduced)', "time saved"),
        (r'top\s+\d+\s*%',                                    "ranking"),
        (r'\d+\s*(?:features?|modules?|services?|apis?)\b',   "deliverables"),
    ]
    hits, examples = 0, []
    text_lower = text.lower()
    for pat, label in patterns:
        matches = re.findall(pat, text_lower)
        if matches:
            hits += len(matches)
            if len(examples) < 5:
                examples.append(f"{label}: {matches[0]}")
    return bool(hits), hits, examples


# ══════════════════════════════════════════════════════════════════
#  FORMAT & CONTACT ANALYSIS
# ══════════════════════════════════════════════════════════════════

def check_format(text: str) -> Dict[str, object]:
    wc = len(text.split())
    # Extract actual email if found
    email_match = re.search(r'[\w\.\-\+]+@[\w\.-]+\.\w{2,}', text)
    phone_match = re.search(r'[\+\d][\d\s\-\(\)\.]{8,15}\d', text)

    return {
        "has_email":         bool(email_match),
        "email_found":       email_match.group(0) if email_match else None,
        "has_phone":         bool(phone_match),
        "has_linkedin":      bool(re.search(r'linkedin\.com|linkedin\.in|/in/', text, re.IGNORECASE)),
        "has_github":        bool(re.search(r'github\.com|github\.io', text, re.IGNORECASE)),
        "has_portfolio":     bool(re.search(r'portfolio|personal\s+site|my\s+website', text, re.IGNORECASE)),
        "word_count":        wc,
        "length_status":     (
            "Optimal (400–800 words)"  if 400 <= wc <= 800  else
            "Acceptable (250–400 words)" if 250 <= wc < 400 else
            "Acceptable (800–1200 words)" if 800 < wc <= 1200 else
            "Too Short (< 250 words)"  if wc < 250 else
            "Too Long (> 1200 words)"
        ),
        "length_ok":         250 <= wc <= 1200,
    }


# ══════════════════════════════════════════════════════════════════
#  ATS SCORE COMPUTATION
# ══════════════════════════════════════════════════════════════════

def compute_ats_score(
    skill_pct:    float,
    kw_coverage:  float,
    section_pct:  float,
    exp_score:    float,
    verb_count:   int,
    has_metrics:  bool,
    metric_count: int,
    fmt:          Dict,
    t_score:      float,
) -> Tuple[float, Dict[str, str], Dict[str, float]]:
    """
    ATS Score = 100 pts total

    Component              Max    Formula
    ──────────────────────────────────────────────────────────────
    Skill Match             40    weighted_skill_pct × 0.40
    JD Keyword Coverage     20    kw_coverage × 0.20
    Section Completeness    15    section_pct × 0.15
    Experience Match        10    exp_score × 10
    Format & Contact        10    per-check points (max 10)
    Writing Quality          5    verbs (max 3) + metrics (max 2)

    Title alignment applies a soft multiplier (0.80 – 1.00) at end.
    """

    skill_pts   = min(skill_pct   * 0.40, 40.0)
    kw_pts      = min(kw_coverage * 0.20, 20.0)
    section_pts = min(section_pct * 0.15, 15.0)
    exp_pts     = round(min(exp_score * 10.0, 10.0), 1)

    fmt_pts = min(
        (3 if fmt.get("has_email")    else 0) +
        (3 if fmt.get("has_phone")    else 0) +
        (2 if fmt.get("has_linkedin") else 0) +
        (1 if fmt.get("has_github")   else 0) +
        (1 if fmt.get("length_ok")    else 0),
        10.0
    )

    verb_pts   = float(min(verb_count // 3, 3))   # 3pts max: 1pt per 3 verbs
    metric_pts = min(float(metric_count) * 0.4, 2.0) if has_metrics else 0.0
    writing_pts = verb_pts + metric_pts

    raw_total = skill_pts + kw_pts + section_pts + exp_pts + fmt_pts + writing_pts

    # Title alignment multiplier: 0.80 (no match) to 1.00 (perfect match)
    multiplier = 0.80 + (0.20 * t_score)
    total      = round(min(raw_total * multiplier, 100.0), 1)

    breakdown_display = {
        "Skill Match           (max 40 pts)": f"{round(skill_pts,   1)} / 40",
        "JD Keyword Coverage   (max 20 pts)": f"{round(kw_pts,      1)} / 20",
        "Section Completeness  (max 15 pts)": f"{round(section_pts, 1)} / 15",
        "Experience Match      (max 10 pts)": f"{round(exp_pts,     1)} / 10",
        "Format & Contact      (max 10 pts)": f"{round(fmt_pts,     1)} / 10",
        "Writing Quality       (max  5 pts)": f"{round(writing_pts, 1)} / 5",
        "Title Multiplier                  ": f"× {round(multiplier, 2)}",
    }

    breakdown_raw = {
        "skill_match":    round(skill_pts,   1),
        "kw_coverage":    round(kw_pts,      1),
        "sections":       round(section_pts, 1),
        "experience":     round(exp_pts,     1),
        "format":         round(fmt_pts,     1),
        "writing":        round(writing_pts, 1),
        "multiplier":     round(multiplier,  2),
        "total":          total,
    }

    return total, breakdown_display, breakdown_raw


def ats_grade(score: float) -> str:
    if score >= 85: return "Excellent"
    if score >= 72: return "Good"
    if score >= 58: return "Average"
    if score >= 42: return "Below Average"
    return "Poor"


def ats_badge(score: float) -> str:
    if score >= 85: return "✅ Excellent"
    if score >= 72: return "🟢 Good"
    if score >= 58: return "🟡 Average"
    if score >= 42: return "🟠 Below Average"
    return "🔴 Poor"


def match_badge(score: float) -> str:
    if score >= 80: return "✅ Strong Match"
    if score >= 60: return "🟢 Good Match"
    if score >= 40: return "🟡 Partial Match"
    if score >= 20: return "🟠 Weak Match"
    return "🔴 Poor Match"


# ══════════════════════════════════════════════════════════════════
#  MASTER FUNCTION
# ══════════════════════════════════════════════════════════════════

def analyze_resume(
    resume_text:     str,
    job_description: str,
    candidate_name:  Optional[str] = None,
) -> Dict:
    """
    Complete ATS analysis of resume vs job description.
    Returns a fully structured, professional report.
    """

    # ── 1. Skill Extraction & Matching ────────────────────────────
    resume_skills = extract_skills(resume_text)
    jd_skills     = extract_skills(job_description)

    skill_pct, matched_skills, missing_skills, extra_skills = weighted_skill_score(
        resume_skills, jd_skills
    )

    # Skill detail: for each JD skill show resume count + JD count
    skill_detail = {
        skill: {
            "in_resume":   resume_skills.get(skill, 0),
            "in_jd":       jd_skills[skill],
            "importance":  round(0.5 + 0.5 * (jd_skills[skill] / (max(jd_skills.values()) or 1)), 2),
            "status":      "✅ Matched" if skill in matched_skills else "❌ Missing",
        }
        for skill in sorted(jd_skills.keys())
    }

    # ── 2. JD Keyword Extraction & Coverage ───────────────────────
    jd_keywords    = extract_jd_keywords(job_description, top_n=40)
    kw_coverage, covered_kws, missing_kws = jd_keyword_coverage(resume_text, jd_keywords)

    # ── 3. JD Match Score ─────────────────────────────────────────
    jd_match = compute_jd_match_score(
        resume_text, job_description,
        matched_skills, len(jd_skills), kw_coverage
    )

    # ── 4. Section Analysis ───────────────────────────────────────
    sections                         = detect_sections(resume_text)
    section_pct, present_secs, missing_secs = score_sections(sections)

    # ── 5. Experience Analysis ────────────────────────────────────
    exp_sc, req_yrs, res_yrs = experience_score(resume_text, job_description)

    # ── 6. Education Analysis ─────────────────────────────────────
    edu_req  = detect_education_level(job_description)
    edu_res  = detect_education_level(resume_text)
    edu_match = edu_res >= edu_req

    # ── 7. Job Title Alignment ────────────────────────────────────
    t_score, jd_titles, res_titles = title_alignment(resume_text, job_description)

    # ── 8. Industry Domain ────────────────────────────────────────
    jd_industry     = detect_industry(job_description)
    resume_industry = detect_industry(resume_text)

    # ── 9. Power Verbs ────────────────────────────────────────────
    verbs, verb_count = detect_power_verbs(resume_text)

    # ── 10. Metrics ───────────────────────────────────────────────
    has_metrics, metric_count, metric_examples = check_metrics(resume_text)

    # ── 11. Format Check ──────────────────────────────────────────
    fmt = check_format(resume_text)

    # ── 12. ATS Score ─────────────────────────────────────────────
    ats_score, breakdown_display, breakdown_raw = compute_ats_score(
        skill_pct   = skill_pct,
        kw_coverage = kw_coverage,
        section_pct = section_pct,
        exp_score   = exp_sc,
        verb_count  = verb_count,
        has_metrics = has_metrics,
        metric_count= metric_count,
        fmt         = fmt,
        t_score     = t_score,
    )

    # ── 13. Overall Compatibility Summary ─────────────────────────
    overall_compatibility = round((ats_score * 0.60 + jd_match * 0.40), 1)

    # ══════════════════════════════════════════════════════════════
    #  FINAL RESPONSE
    # ══════════════════════════════════════════════════════════════

    return {

        # ────────────────────────────────────────────────────────────
        # CANDIDATE
        # ────────────────────────────────────────────────────────────
        "candidate": {
            "name": candidate_name or "Not Provided",
        },

        # ────────────────────────────────────────────────────────────
        # OVERALL SCORES  (the headline numbers)
        # ────────────────────────────────────────────────────────────
        "scores": {
            "ats_score": {
                "value":       ats_score,
                "out_of":      100,
                "grade":       ats_grade(ats_score),
                "badge":       ats_badge(ats_score),
                "description": "How well this resume will perform in ATS systems",
            },
            "jd_match_score": {
                "value":       jd_match,
                "out_of":      100,
                "badge":       match_badge(jd_match),
                "description": "How closely the resume content matches this specific job description",
            },
            "overall_compatibility": {
                "value":       overall_compatibility,
                "out_of":      100,
                "description": "Combined ATS + JD match score (60% ATS, 40% JD match)",
            },
        },

        # ────────────────────────────────────────────────────────────
        # SCORE BREAKDOWN
        # ────────────────────────────────────────────────────────────
        "score_breakdown": {
            "components":   breakdown_display,
            "raw_values":   breakdown_raw,
        },

        # ────────────────────────────────────────────────────────────
        # SKILL ANALYSIS
        # ────────────────────────────────────────────────────────────
        "skill_analysis": {
            "summary": {
                "match_percentage":    f"{skill_pct}%",
                "total_skills_in_jd":  len(jd_skills),
                "matched_count":       len(matched_skills),
                "missing_count":       len(missing_skills),
                "extra_in_resume":     len(extra_skills),
            },
            "matched_skills":          matched_skills,
            "missing_skills":          missing_skills,
            "additional_resume_skills": extra_skills[:15],
            "skill_detail":            skill_detail,
        },

        # ────────────────────────────────────────────────────────────
        # JD KEYWORD ANALYSIS
        # ────────────────────────────────────────────────────────────
        "jd_keyword_analysis": {
            "coverage_percentage":  f"{kw_coverage}%",
            "keywords_covered":     covered_kws,
            "keywords_missing":     missing_kws,
            "description":          "Important keywords from JD found/missing in resume",
        },

        # ────────────────────────────────────────────────────────────
        # SECTION ANALYSIS
        # ────────────────────────────────────────────────────────────
        "section_analysis": {
            "completeness":      f"{round((len(present_secs) / len(ATS_SECTIONS)) * 100)}%",
            "sections_present":  present_secs,
            "sections_missing":  missing_secs,
            "section_scores":    {
                sec: {"present": val, "weight": SECTION_WEIGHTS.get(sec, 0)}
                for sec, val in sections.items()
            },
        },

        # ────────────────────────────────────────────────────────────
        # EXPERIENCE
        # ────────────────────────────────────────────────────────────
        "experience": {
            "required_by_jd":      f"{req_yrs} year{'s' if req_yrs != 1 else ''}" if req_yrs else "Not specified",
            "detected_in_resume":  f"~{int(res_yrs)} year{'s' if int(res_yrs) != 1 else ''}" if res_yrs else "Not detected",
            "match_score":         f"{round(exp_sc * 100)}%",
            "match_status": (
                "Exceeds Requirement ✅"    if res_yrs and req_yrs and res_yrs > req_yrs else
                "Meets Requirement ✅"      if exp_sc >= 1.0 else
                "Partially Meets ⚠️"       if exp_sc >= 0.6 else
                "Below Requirement ❌"     if req_yrs else
                "No Requirement Stated ℹ️"
            ),
        },

        # ────────────────────────────────────────────────────────────
        # EDUCATION
        # ────────────────────────────────────────────────────────────
        "education": {
            "required_by_jd":       EDU_LABELS.get(edu_req, "Not Specified"),
            "detected_in_resume":   EDU_LABELS.get(edu_res, "Not Detected"),
            "match_status":         "Meets Requirement ✅" if edu_match else "Below Requirement ❌",
            "education_gap":        max(0, edu_req - edu_res),
        },

        # ────────────────────────────────────────────────────────────
        # JOB TITLE ALIGNMENT
        # ────────────────────────────────────────────────────────────
        "job_title_alignment": {
            "alignment_score":    f"{round(t_score * 100)}%",
            "status": (
                "Strong Match ✅"   if t_score >= 0.9 else
                "Partial Match ⚠️" if t_score >= 0.5 else
                "Weak Match ❌"
            ),
            "roles_detected_in_jd":     jd_titles if jd_titles else ["Not specified"],
            "roles_detected_in_resume": res_titles if res_titles else ["Not detected"],
        },

        # ────────────────────────────────────────────────────────────
        # INDUSTRY DOMAIN
        # ────────────────────────────────────────────────────────────
        "industry_domain": {
            "jd_industry":              jd_industry if jd_industry else ["General / Not specified"],
            "resume_industry_signals":  resume_industry if resume_industry else ["General / Not detected"],
            "domain_alignment":         "✅ Aligned" if (set(jd_industry) & set(resume_industry)) else
                                        "⚠️ Partial" if (jd_industry and resume_industry) else
                                        "ℹ️ Not Determined",
        },

        # ────────────────────────────────────────────────────────────
        # WRITING QUALITY
        # ────────────────────────────────────────────────────────────
        "writing_quality": {
            "power_verbs": {
                "count":    verb_count,
                "strength": "Strong ✅" if verb_count >= 10 else "Moderate ⚠️" if verb_count >= 5 else "Weak ❌",
                "verbs_found": verbs,
            },
            "quantified_achievements": {
                "has_metrics":   has_metrics,
                "count":         metric_count,
                "strength":      "Strong ✅" if metric_count >= 6 else "Moderate ⚠️" if metric_count >= 3 else "Weak ❌",
                "examples":      metric_examples,
            },
        },

        # ────────────────────────────────────────────────────────────
        # FORMAT & CONTACT
        # ────────────────────────────────────────────────────────────
        "format_and_contact": {
            "email":     "✅ Found" if fmt.get("has_email")    else "❌ Missing",
            "phone":     "✅ Found" if fmt.get("has_phone")    else "❌ Missing",
            "linkedin":  "✅ Found" if fmt.get("has_linkedin") else "❌ Missing",
            "github":    "✅ Found" if fmt.get("has_github")   else "❌ Missing",
            "portfolio": "✅ Found" if fmt.get("has_portfolio") else "❌ Missing",
            "word_count":    fmt.get("word_count"),
            "length_status": fmt.get("length_status"),
        },

    }