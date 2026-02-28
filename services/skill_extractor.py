from functools import lru_cache
"""
skill_extractor.py  ─  Advanced Skill Entity Recognition
──────────────────────────────────────────────────────────
Uses multiple extraction strategies:
  1. Exact phrase matching with boundary safety
  2. Alias/synonym normalization
  3. Context-aware boosting (skills near "required", "must" score higher)
  4. Negation detection (filters "no experience with X", "not required")
  5. Co-occurrence scoring (skills that appear near each other cluster)
  6. Importance weighting from JD frequency + position
"""

import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from services.nlp_core import tokenize, TECH_SYNONYMS, STOPWORDS


# ══════════════════════════════════════════════════════════════════
#  MASTER SKILL KNOWLEDGE BASE
#  Format: (canonical, [aliases], category, base_importance)
# ══════════════════════════════════════════════════════════════════

SKILL_DB: List[Tuple[str, List[str], str, float]] = [

    # ── AI / ML ────────────────────────────────────────────────────
    ("natural language processing", ["natural language processing","nlp","text processing","text analytics"], "AI/ML", 0.95),
    ("machine learning",            ["machine learning","ml"],                                                "AI/ML", 0.90),
    ("deep learning",               ["deep learning","dl","neural network","neural networks"],                "AI/ML", 0.90),
    ("computer vision",             ["computer vision","image recognition","object detection"],               "AI/ML", 0.88),
    ("reinforcement learning",      ["reinforcement learning","rl"],                                          "AI/ML", 0.85),
    ("large language models",       ["large language model","llm","llms","gpt","chatgpt","large language models"], "AI/ML", 0.92),
    ("generative ai",               ["generative ai","gen ai","genai","stable diffusion","dall-e"],           "AI/ML", 0.88),
    ("feature engineering",         ["feature engineering","feature extraction","feature selection"],         "AI/ML", 0.80),
    ("model deployment",            ["model deployment","model serving","model inference"],                   "AI/ML", 0.82),
    ("mlops",                       ["mlops","ml ops","ml pipeline","model monitoring"],                      "AI/ML", 0.85),
    ("data science",                ["data science","data scientist"],                                        "AI/ML", 0.88),
    ("scikit-learn",                ["scikit-learn","sklearn","scikit learn"],                                "AI/ML", 0.82),
    ("tensorflow",                  ["tensorflow","tf","tensorflow2","tf2"],                                  "AI/ML", 0.85),
    ("pytorch",                     ["pytorch","torch","pytorch lightning"],                                  "AI/ML", 0.85),
    ("keras",                       ["keras"],                                                                "AI/ML", 0.80),
    ("xgboost",                     ["xgboost","xgb"],                                                       "AI/ML", 0.78),
    ("lightgbm",                    ["lightgbm","lgbm"],                                                     "AI/ML", 0.78),
    ("catboost",                    ["catboost"],                                                             "AI/ML", 0.75),
    ("huggingface",                 ["huggingface","hugging face","transformers","hf transformers"],          "AI/ML", 0.85),
    ("langchain",                   ["langchain","lang chain","langgraph"],                                   "AI/ML", 0.85),
    ("openai",                      ["openai","open ai","chatgpt api","gpt-4","gpt4","gpt-3","gpt3","openai api"], "AI/ML", 0.87),
    ("pandas",                      ["pandas","pd"],                                                          "AI/ML", 0.80),
    ("numpy",                       ["numpy","np"],                                                           "AI/ML", 0.78),
    ("matplotlib",                  ["matplotlib","seaborn","plotly","visualization"],                        "AI/ML", 0.72),
    ("mlflow",                      ["mlflow","ml flow"],                                                     "AI/ML", 0.78),

    # ── Languages ─────────────────────────────────────────────────
    ("python",      ["python","python3","python2"],                     "Language", 0.95),
    ("javascript",  ["javascript","js","ecmascript","es6","es2015"],    "Language", 0.93),
    ("typescript",  ["typescript","ts"],                                "Language", 0.88),
    ("java",        ["java","java8","java11","java17","spring java"],    "Language", 0.90),
    ("c++",         ["c++","cpp","c plus plus"],                        "Language", 0.87),
    ("c#",          ["c#","csharp","c sharp","dotnet c#"],              "Language", 0.85),
    ("golang",      ["golang","go","go lang"],                          "Language", 0.85),
    ("rust",        ["rust","rust lang"],                               "Language", 0.83),
    ("ruby",        ["ruby"],                                           "Language", 0.80),
    ("php",         ["php","php7","php8"],                              "Language", 0.78),
    ("swift",       ["swift","swift ui","swiftui"],                     "Language", 0.82),
    ("kotlin",      ["kotlin","kotlin coroutines"],                     "Language", 0.82),
    ("scala",       ["scala","akka"],                                   "Language", 0.80),
    ("r",           ["\\br\\b","r programming","r language"],           "Language", 0.78),
    ("matlab",      ["matlab","octave"],                                "Language", 0.72),
    ("bash",        ["bash","shell","shell scripting","shell script","sh"], "Language", 0.78),
    ("powershell",  ["powershell","ps1"],                               "Language", 0.72),
    ("dart",        ["dart"],                                           "Language", 0.75),
    ("elixir",      ["elixir","phoenix"],                               "Language", 0.72),
    ("haskell",     ["haskell"],                                        "Language", 0.70),
    ("julia",       ["julia"],                                          "Language", 0.70),
    ("lua",         ["lua"],                                            "Language", 0.68),
    ("assembly",    ["assembly","asm","x86","x64"],                     "Language", 0.68),
    ("solidity",    ["solidity","vyper","smart contract"],              "Language", 0.75),
    ("html",        ["html","html5"],                                   "Language", 0.78),
    ("css",         ["css","css3"],                                     "Language", 0.78),
    ("sql",         ["sql","t-sql","pl/sql","ansi sql","tsql"],         "Language", 0.85),

    # ── Frontend ──────────────────────────────────────────────────
    ("react",          ["react","reactjs","react.js","react hooks","react js"],            "Frontend", 0.90),
    ("angular",        ["angular","angularjs","angular2","angular js"],                    "Frontend", 0.85),
    ("vue",            ["vue","vuejs","vue.js","vue3","nuxt"],                             "Frontend", 0.83),
    ("next.js",        ["next.js","nextjs","next js"],                                     "Frontend", 0.85),
    ("svelte",         ["svelte","sveltekit"],                                             "Frontend", 0.78),
    ("jquery",         ["jquery"],                                                         "Frontend", 0.70),
    ("bootstrap",      ["bootstrap","bootstrap5"],                                         "Frontend", 0.72),
    ("tailwind",       ["tailwind","tailwindcss","tailwind css"],                          "Frontend", 0.78),
    ("sass",           ["sass","scss","less"],                                             "Frontend", 0.72),
    ("webpack",        ["webpack","webpack5"],                                             "Frontend", 0.73),
    ("vite",           ["vite","vitejs"],                                                  "Frontend", 0.75),
    ("redux",          ["redux","redux toolkit","zustand","mobx"],                         "Frontend", 0.78),
    ("graphql",        ["graphql","apollo graphql","relay"],                               "Frontend", 0.82),
    ("webrtc",         ["webrtc","websocket","web socket"],                                "Frontend", 0.75),
    ("react native",   ["react native"],                                                   "Mobile", 0.85),
    ("responsive design", ["responsive design","responsive web","mobile first"],           "Frontend", 0.78),

    # ── Backend ───────────────────────────────────────────────────
    ("fastapi",      ["fastapi","fast api"],                                "Backend", 0.88),
    ("django",       ["django","django rest framework","drf","django orm"], "Backend", 0.85),
    ("flask",        ["flask","flask-restful"],                             "Backend", 0.82),
    ("node.js",      ["node.js","nodejs","node js","node"],                 "Backend", 0.87),
    ("express",      ["express","expressjs","express.js"],                  "Backend", 0.83),
    ("spring boot",  ["spring boot","spring framework","spring mvc"],       "Backend", 0.85),
    ("asp.net",      ["asp.net","asp net","dotnet","net core"],             "Backend", 0.83),
    ("laravel",      ["laravel","lumen"],                                   "Backend", 0.78),
    ("ruby on rails",["ruby on rails","rails","ror"],                       "Backend", 0.80),
    ("grpc",         ["grpc","protocol buffers","protobuf"],                "Backend", 0.80),
    ("celery",       ["celery","async tasks","task queue"],                 "Backend", 0.75),
    ("rabbitmq",     ["rabbitmq","rabbit mq","amqp"],                       "Backend", 0.78),
    ("rest api",     ["rest api","restful api","restful","rest","api design"], "Backend", 0.87),
    ("microservices",["microservices","microservice","service mesh","micro services"], "Backend", 0.85),
    ("serverless",   ["serverless","lambda","cloud functions","faas"],      "Backend", 0.80),

    # ── Cloud & DevOps ────────────────────────────────────────────
    ("amazon web services", ["amazon web services","aws","ec2","s3","lambda","rds","eks","ecs","cloudformation"], "Cloud", 0.92),
    ("microsoft azure",     ["microsoft azure","azure","azure devops","azure ad","aks"],                         "Cloud", 0.88),
    ("google cloud",        ["google cloud","gcp","google cloud platform","gke","bigquery cloud","cloud run"],    "Cloud", 0.87),
    ("docker",      ["docker","dockerfile","docker compose","containerization"],    "DevOps", 0.90),
    ("kubernetes",  ["kubernetes","k8s","kubectl","helm","eks","gke","aks"],        "DevOps", 0.90),
    ("terraform",   ["terraform","terraform cloud","iac","infrastructure as code"], "DevOps", 0.88),
    ("ansible",     ["ansible","ansible playbook"],                                 "DevOps", 0.82),
    ("jenkins",     ["jenkins","jenkins pipeline","jenkinsfile"],                   "DevOps", 0.80),
    ("github actions", ["github actions","gha","workflow yaml"],                   "DevOps", 0.83),
    ("gitlab ci",   ["gitlab ci","gitlab ci/cd","gitlab pipeline"],                "DevOps", 0.80),
    ("ci/cd",       ["ci/cd","ci cd","continuous integration","continuous deployment","continuous delivery","cicd"], "DevOps", 0.88),
    ("linux",       ["linux","ubuntu","centos","debian","rhel","bash scripting","unix"], "DevOps", 0.83),
    ("nginx",       ["nginx","nginx proxy"],                                         "DevOps", 0.78),
    ("prometheus",  ["prometheus","alertmanager"],                                   "DevOps", 0.78),
    ("grafana",     ["grafana","dashboards","metrics"],                              "DevOps", 0.75),
    ("datadog",     ["datadog","new relic","monitoring"],                            "DevOps", 0.75),
    ("argocd",      ["argocd","argo cd","gitops","flux"],                            "DevOps", 0.78),
    ("istio",       ["istio","service mesh","envoy"],                                "DevOps", 0.75),
    ("pulumi",      ["pulumi"],                                                      "DevOps", 0.72),

    # ── Databases ─────────────────────────────────────────────────
    ("postgresql",     ["postgresql","postgres","psql","pg"],                        "Database", 0.85),
    ("mysql",          ["mysql","mariadb"],                                          "Database", 0.82),
    ("mongodb",        ["mongodb","mongo","mongoose"],                               "Database", 0.85),
    ("redis",          ["redis","redis cache","redisearch"],                         "Database", 0.83),
    ("elasticsearch",  ["elasticsearch","elastic search","opensearch","kibana"],     "Database", 0.82),
    ("cassandra",      ["cassandra","apache cassandra","scylladb"],                  "Database", 0.80),
    ("dynamodb",       ["dynamodb","dynamo"],                                        "Database", 0.80),
    ("firebase",       ["firebase","firestore","realtime database"],                 "Database", 0.78),
    ("neo4j",          ["neo4j","graph database","cypher"],                          "Database", 0.75),
    ("oracle",         ["oracle","oracle db","pl/sql oracle"],                       "Database", 0.78),
    ("sql server",     ["sql server","mssql","microsoft sql","t-sql server"],        "Database", 0.78),
    ("snowflake",      ["snowflake"],                                                "Database", 0.83),
    ("bigquery",       ["bigquery","big query","bq"],                                "Database", 0.83),
    ("redshift",       ["redshift","amazon redshift"],                               "Database", 0.80),
    ("pinecone",       ["pinecone","vector database","weaviate","qdrant","chroma"],  "Database", 0.80),
    ("clickhouse",     ["clickhouse","druid","presto","trino"],                      "Database", 0.78),
    ("sqlite",         ["sqlite","sqlite3"],                                         "Database", 0.68),
    ("nosql",          ["nosql","no-sql","document store"],                          "Database", 0.78),

    # ── Data Engineering ──────────────────────────────────────────
    ("apache spark",   ["apache spark","spark","pyspark","spark sql"],               "DataEng", 0.88),
    ("apache kafka",   ["apache kafka","kafka","kafka streams","kafka connect"],      "DataEng", 0.87),
    ("apache airflow", ["apache airflow","airflow","airflow dag"],                   "DataEng", 0.85),
    ("apache hadoop",  ["apache hadoop","hadoop","hdfs","mapreduce","hive"],         "DataEng", 0.80),
    ("dbt",            ["dbt","data build tool"],                                    "DataEng", 0.82),
    ("data pipeline",  ["data pipeline","etl","elt","data warehouse","data lake"],   "DataEng", 0.85),
    ("data visualization", ["data visualization","tableau","power bi","looker","data viz","dashboards"], "DataEng", 0.78),
    ("data analysis",  ["data analysis","data analytics","business analytics"],      "DataEng", 0.82),
    ("big data",       ["big data","distributed computing","large scale data"],      "DataEng", 0.80),

    # ── Mobile ────────────────────────────────────────────────────
    ("flutter",  ["flutter","dart flutter"],         "Mobile", 0.85),
    ("android",  ["android","android sdk","android studio","android development"], "Mobile", 0.83),
    ("ios",      ["ios","xcode","swift ios","objective-c"], "Mobile", 0.83),

    # ── Security ──────────────────────────────────────────────────
    ("cybersecurity",       ["cybersecurity","cyber security","infosec","information security"], "Security", 0.85),
    ("oauth",               ["oauth","oauth2","oauth 2.0","openid connect","oidc"],              "Security", 0.80),
    ("jwt",                 ["jwt","json web token"],                                            "Security", 0.78),
    ("ssl/tls",             ["ssl","tls","ssl/tls","https","ssl certificate"],                   "Security", 0.78),
    ("penetration testing", ["penetration testing","pen testing","pentesting","ethical hacking"], "Security", 0.82),
    ("sso",                 ["sso","single sign-on","saml","active directory","ldap"],           "Security", 0.78),

    # ── Engineering Practices ─────────────────────────────────────
    ("system design",           ["system design","distributed systems","scalable systems"],              "Practice", 0.90),
    ("design patterns",         ["design patterns","gang of four","gof patterns"],                       "Practice", 0.83),
    ("solid principles",        ["solid principles","solid","clean code","clean architecture"],           "Practice", 0.82),
    ("object oriented programming", ["object oriented","oop","object-oriented programming","oops"],      "Practice", 0.82),
    ("functional programming",  ["functional programming","fp","immutable","pure functions"],             "Practice", 0.78),
    ("data structures",         ["data structures","algorithms","dsa","data structure","complexity"],     "Practice", 0.85),
    ("unit testing",            ["unit testing","unit tests","pytest","jest","mocha","junit"],            "Practice", 0.80),
    ("integration testing",     ["integration testing","e2e testing","end to end testing","cypress","selenium"], "Practice", 0.78),
    ("test driven development", ["test driven","tdd","test-driven development"],                         "Practice", 0.78),
    ("code review",             ["code review","peer review","pull request","pr review"],                "Practice", 0.75),
    ("agile",                   ["agile","agile methodology","agile development"],                        "Practice", 0.80),
    ("scrum",                   ["scrum","sprint","scrum master","daily standup"],                        "Practice", 0.78),
    ("devops",                  ["devops","dev ops","site reliability","sre"],                            "Practice", 0.83),
    ("git",                     ["git","github","gitlab","bitbucket","version control","vcs"],            "Practice", 0.83),
    ("api design",              ["api design","api development","api gateway","swagger","openapi"],       "Practice", 0.82),
    ("performance optimization",["performance optimization","performance tuning","caching","latency"],    "Practice", 0.80),
    ("high level design",       ["high level design","hld","system architecture"],                       "Practice", 0.83),
    ("low level design",        ["low level design","lld","class design","detailed design"],              "Practice", 0.80),

    # ── Tools ─────────────────────────────────────────────────────
    ("git",         ["git","git flow","git commands"],                              "Tool", 0.82),
    ("jira",        ["jira","jira board","atlassian jira"],                         "Tool", 0.72),
    ("confluence",  ["confluence","wiki","documentation"],                          "Tool", 0.68),
    ("postman",     ["postman","insomnia","api testing tool"],                      "Tool", 0.70),
    ("vs code",     ["vs code","visual studio code","vscode"],                      "Tool", 0.65),

    # ── Soft Skills ───────────────────────────────────────────────
    ("communication",        ["communication","verbal communication","written communication","interpersonal"], "Soft", 0.75),
    ("leadership",           ["leadership","team lead","tech lead","people management","lead developer"],     "Soft", 0.80),
    ("problem solving",      ["problem solving","problem-solving","analytical thinking","troubleshooting"],   "Soft", 0.78),
    ("collaboration",        ["collaboration","teamwork","team player","cross-functional","team work"],        "Soft", 0.75),
    ("mentoring",            ["mentoring","mentorship","coaching","knowledge transfer"],                       "Soft", 0.72),
    ("project management",   ["project management","project planning","delivery","stakeholder"],               "Soft", 0.78),
    ("critical thinking",    ["critical thinking","analytical","problem analysis"],                           "Soft", 0.72),
    ("time management",      ["time management","deadline","prioritization","multitasking"],                   "Soft", 0.68),
]

# ── Build compiled regex patterns ────────────────────────────────
_COMPILED: Dict[str, List] = {}
_SKILL_META: Dict[str, Dict] = {}

for _canonical, _aliases, _category, _importance in SKILL_DB:
    _pats = []
    for _alias in _aliases:
        if _alias.startswith("\\b"):
            _pats.append(re.compile(_alias, re.IGNORECASE))
        else:
            # Escape special regex chars but preserve the intent
            _escaped = re.escape(_alias)
            # Replace escaped spaces with flexible whitespace
            _escaped = _escaped.replace(r'\ ', r'[\s\-]+')
            _pats.append(re.compile(
                r'(?<![a-zA-Z0-9\+\#\-])' + _escaped + r'(?![a-zA-Z0-9\+\#\-])',
                re.IGNORECASE
            ))
    _COMPILED[_canonical] = _pats
    _SKILL_META[_canonical] = {"category": _category, "importance": _importance}

# Negation patterns — if these precede a skill mention, skip it
_NEGATION_RE = re.compile(
    r'(?:no|not|without|lack|lacking|never|unable|don\'t|dont|doesn\'t|doesnt|'
    r'won\'t|wont|hasn\'t|hasnt|haven\'t|havent|nor)\s+(?:\w+\s+){0,3}',
    re.IGNORECASE
)

# Context boosters — phrases near these indicate a required skill
_REQUIRED_CONTEXT_RE = re.compile(
    r'(?:required|must have|must be|mandatory|essential|strong|proficient|expert|'
    r'experience with|expertise in|knowledge of|familiar with|hands.on|looking for)',
    re.IGNORECASE
)


# ══════════════════════════════════════════════════════════════════
#  CORE EXTRACTION FUNCTION
# ══════════════════════════════════════════════════════════════════

def extract_skills_advanced(
    text: str,
    context_boost: bool = True,
) -> Dict[str, Dict]:
    """
    Advanced skill extraction with:
    - Negation detection
    - Context-aware importance boosting
    - Position weighting
    - Returns {canonical: {count, positions, importance, category, context_score}}
    """
    text_lower = text.lower()
    doc_len    = len(text_lower)

    # Build negation spans to skip
    negation_spans = [(m.start(), m.end()) for m in _NEGATION_RE.finditer(text_lower)]
    # Build required-context spans for boosting
    required_spans = [(m.start(), min(m.end()+80, doc_len)) for m in _REQUIRED_CONTEXT_RE.finditer(text_lower)]

    found: Dict[str, Dict] = {}
    consumed: List[Tuple[int, int]] = []

    for canonical, _ in [(s[0], s[1]) for s in SKILL_DB]:
        patterns = _COMPILED.get(canonical, [])
        meta     = _SKILL_META.get(canonical, {})
        matches_info = []

        for pat in patterns:
            for m in pat.finditer(text_lower):
                span = (m.start(), m.end())

                # Skip if span already consumed by a longer match
                if any(s <= span[0] and span[1] <= e for s, e in consumed):
                    continue

                # Skip if inside a negation context
                if any(ns <= span[0] <= ne for ns, ne in negation_spans):
                    continue

                matches_info.append({
                    "start":    span[0],
                    "end":      span[1],
                    "position": span[0] / max(doc_len, 1),
                    "in_required_context": any(rs <= span[0] <= re_ for rs, re_ in required_spans),
                })

        if matches_info:
            # Position weight: earlier mentions = more relevant
            avg_position = sum(m["position"] for m in matches_info) / len(matches_info)
            position_weight = 1.0 - 0.3 * avg_position  # 1.0 at top, 0.7 at bottom

            # Context boost
            req_count = sum(1 for m in matches_info if m["in_required_context"])
            context_score = min(1.0 + 0.3 * req_count, 1.5) if context_boost else 1.0

            found[canonical] = {
                "count":           len(matches_info),
                "positions":       [m["start"] for m in matches_info],
                "position_weight": round(position_weight, 3),
                "context_score":   round(context_score, 3),
                "category":        meta.get("category", "Other"),
                "base_importance": meta.get("importance", 0.7),
                "effective_score": round(
                    meta.get("importance", 0.7) * position_weight * context_score, 4
                ),
            }
            # Mark all positions as consumed
            for m in matches_info:
                consumed.append((m["start"], m["end"]))

    return found


@lru_cache(maxsize=128)
def get_skill_counts(text: str) -> Dict[str, int]:
    """Simple interface — returns {skill: count}. Cached for repeated calls with same text."""
    advanced = extract_skills_advanced(text, context_boost=False)
    return {k: v["count"] for k, v in advanced.items()}


def compute_weighted_match(
    resume_skills: Dict[str, Dict],
    jd_skills:     Dict[str, Dict],
) -> Tuple[float, List[str], List[str], List[str], Dict]:
    """
    Compute weighted skill match between resume and JD.

    JD skill importance = base_importance × context_score × frequency_weight
    Match score = Σ(matched_skill_importance) / Σ(all_jd_skill_importance)

    Returns:
      (match_pct, matched, missing, extra, skill_report)
    """
    if not jd_skills:
        return 0.0, [], [], list(resume_skills.keys()), {}

    # Compute JD skill weights
    jd_weights: Dict[str, float] = {}
    for skill, info in jd_skills.items():
        freq_weight = min(0.5 + 0.5 * (info["count"] / max(v["count"] for v in jd_skills.values())), 1.0)
        jd_weights[skill] = info["base_importance"] * info["context_score"] * freq_weight

    total_weight = sum(jd_weights.values()) or 1.0

    jd_set     = set(jd_skills.keys())
    resume_set = set(resume_skills.keys())
    matched    = sorted(jd_set & resume_set)
    missing    = sorted(jd_set - resume_set)
    extra      = sorted(resume_set - jd_set)

    matched_weight = sum(jd_weights.get(s, 0) for s in matched)
    match_pct      = round(min(matched_weight / total_weight * 100, 100.0), 1)

    # Build detailed skill report for each JD skill
    skill_report = {}
    for skill in sorted(jd_set):
        resume_info = resume_skills.get(skill, {})
        jd_info     = jd_skills[skill]
        weight      = jd_weights.get(skill, 0)

        skill_report[skill] = {
            "status":           "✅ Matched" if skill in matched else "❌ Missing",
            "importance":       round(weight, 3),
            "jd_mentions":      jd_info["count"],
            "resume_mentions":  resume_info.get("count", 0),
            "jd_required":      jd_info.get("context_score", 1.0) > 1.1,
            "category":         jd_info.get("category", "Other"),
            "resume_position":  "Early" if resume_info.get("position_weight", 0) > 0.85 else
                                "Middle" if resume_info.get("position_weight", 0) > 0.70 else
                                "Late" if resume_info else "Not Found",
        }

    return match_pct, matched, missing, extra, skill_report


def categorize_skills(skills: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Group skills by category."""
    categories: Dict[str, List[str]] = defaultdict(list)
    for skill, info in skills.items():
        categories[info.get("category", "Other")].append(skill)
    return dict(categories)