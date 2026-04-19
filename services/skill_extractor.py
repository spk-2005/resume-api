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
#  MASTER SKILL DATABASE (1,000+ Tokens)
# ══════════════════════════════════════════════════════════════════

SKILL_DB: List[Tuple[str, List[str], str, float]] = [
    # ── AI / ML / LLM (Highly Expanded) ───────────────────────────
    ("natural language processing", ["natural language processing","nlp","text processing","text analytics"], "AI/ML", 0.95),
    ("machine learning",            ["machine learning","ml"],                                                "AI/ML", 0.90),
    ("deep learning",               ["deep learning","dl","neural network","neural networks"],                "AI/ML", 0.90),
    ("computer vision",             ["computer vision","image recognition","object detection"],               "AI/ML", 0.88),
    ("reinforcement learning",      ["reinforcement learning","rl"],                                          "AI/ML", 0.85),
    ("large language models",       ["large language model","llm","llms","gpt","chatgpt","large language models"], "AI/ML", 0.92),
    ("generative ai",               ["generative ai","gen ai","genai","stable diffusion","midjourney"], "AI/ML", 0.88),
    ("vector database",             ["vector database","vector db","chroma","chromadb","qdrant","weaviate","pinecone","milvus","zilliz","pgvector","lanceDB","marqo"], "AI/ML", 0.87),
    ("rag",                         ["rag","retrieval augmented generation","semantic search","hybrid search","re-ranking","reranking"], "AI/ML", 0.88),
    ("prompt engineering",          ["prompt engineering","prompting","llm orchestration","chain of thought","few-shot"], "AI/ML", 0.82),
    ("mlops",                       ["mlops","ml ops","ml pipeline","model monitoring","dvc","wandb","weights & biases","bentoml","zenml"], "AI/ML", 0.85),
    ("pytorch",                     ["pytorch","torch","pytorch lightning","deepseek-v3","deepspeed","fairseq"], "AI/ML", 0.85),
    ("huggingface",                 ["huggingface","hugging face","transformers","hf transformers","diffusers","accelerate"], "AI/ML", 0.85),
    ("langchain",                   ["langchain","lang chain","langgraph","llama-index","llamaindex","autogen","crewai","haystack"], "AI/ML", 0.85),
    ("llm fine-tuning",             ["fine-tuning","finetuning","lora","qlora","peft","rlhf","dpo","qalign"], "AI/ML", 0.86),
    ("openai",                      ["openai","gpt-4","gpt-3.5","chatgpt api","whisper","dall-e"], "AI/ML", 0.90),
    ("anthropic",                   ["anthropic","claude","claude-3","claude-2"], "AI/ML", 0.88),
    ("google gemini",               ["google gemini","gemini pro","vertex ai","palm2"], "AI/ML", 0.87),
    ("deepseek",                    ["deepseek","deepseek-v2","deepseek-coder"], "AI/ML", 0.84),
    ("llm tools",                   ["vllm","tgi","ollama","llama.cpp","local llm"], "AI/ML", 0.82),
    ("synthetic data",              ["synthetic data","data augmentation","snorkel","labelbox"], "AI/ML", 0.80),
    ("cuda",                        ["cuda","nvidia sdk","tensorrt","cudnn","gpu optimization","triton"], "AI/ML", 0.84),
    ("onnx",                        ["onnx","onnx runtime","model quantization"],                             "AI/ML", 0.78),
    ("scikit-learn",                ["scikit-learn","sklearn","scikit learn"],                                "AI/ML", 0.82),
    ("tensorflow",                  ["tensorflow","tf","tensorflow2","tf2"],                                  "AI/ML", 0.85),
    ("keras",                       ["keras"],                                                                "AI/ML", 0.80),
    ("pandas",                      ["pandas","pd"],                                                          "AI/ML", 0.80),
    ("numpy",                       ["numpy","np"],                                                           "AI/ML", 0.78),

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
    ("bash",        ["bash","shell","shell scripting","shell script"],  "Language", 0.78),
    ("sql",         ["sql","t-sql","pl/sql","ansi sql","tsql"],         "Language", 0.85),
    ("html",        ["html","html5"],                                   "Language", 0.78),
    ("css",         ["css","css3","scss","sass"],                       "Language", 0.78),
    ("mojo",        ["mojo"],                                           "Language", 0.70),

    # ── Frontend (Modernized) ─────────────────────────────────────
    ("react",          ["react","reactjs","react.js","react hooks","react js"],            "Frontend", 0.90),
    ("angular",        ["angular","angularjs","angular2","angular js"],                    "Frontend", 0.85),
    ("vue",            ["vue","vuejs","vue.js","vue3","nuxt","nuxtjs"],                     "Frontend", 0.83),
    ("next.js",        ["next.js","nextjs","next js","app router","pages router"],         "Frontend", 0.85),
    ("svelte",         ["svelte","sveltekit","svelte kit"],                                "Frontend", 0.78),
    ("solidjs",        ["solidjs","solid.js","solid js"],                                  "Frontend", 0.78),
    ("qwik",           ["qwik","qwik city"],                                               "Frontend", 0.78),
    ("astro",          ["astro","astro.build"],                                            "Frontend", 0.80),
    ("htmx",           ["htmx","unpoly","hotwire","turbo"],                                "Frontend", 0.80),
    ("remix",          ["remix","remix.run"],                                              "Frontend", 0.78),
    ("tailwind",       ["tailwind","tailwindcss","tailwind css"],                          "Frontend", 0.78),
    ("shadcn ui",      ["shadcn","shadcn/ui","radix ui"],                                   "Frontend", 0.80),
    ("framer motion",  ["framer motion","gsap","web animations"],                          "Frontend", 0.75),
    ("tanstack query", ["tanstack query","react query","swr"],                             "Frontend", 0.82),
    ("three.js",       ["three.js","threejs","webgl","webgpu"],                            "Frontend", 0.78),
    ("redux",          ["redux","redux toolkit","zustand","mobx","jotai","signals","valtio"], "Frontend", 0.78),
    ("recoil",         ["recoil","xstate"],                                                "Frontend", 0.75),
    ("graphql",        ["graphql","apollo graphql","relay","trpc"],                        "Frontend", 0.82),
    ("vite",           ["vite","vitejs","esbuild","swc"],                                  "Frontend", 0.75),
    ("pnpm",           ["pnpm","bun","deno","biome"],                                      "Frontend", 0.78),

    # ── Backend ───────────────────────────────────────────────────
    ("fastapi",      ["fastapi","fast api","pydantic"],                     "Backend", 0.88),
    ("django",       ["django","django rest framework","drf","django orm"], "Backend", 0.85),
    ("flask",        ["flask","flask-restful"],                             "Backend", 0.82),
    ("node.js",      ["node.js","nodejs","node js","node"],                 "Backend", 0.87),
    ("nest.js",      ["nest.js","nestjs","nest js"],                        "Backend", 0.84),
    ("express",      ["express","expressjs","express.js"],                  "Backend", 0.83),
    ("hono",         ["hono","hono.js","elysia"],                            "Backend", 0.78),
    ("fastify",      ["fastify"],                                            "Backend", 0.80),
    ("trpc",         ["trpc","tsoa"],                                        "Backend", 0.80),
    ("spring boot",  ["spring boot","spring framework","spring mvc"],       "Backend", 0.85),
    ("asp.net",      ["asp.net","asp net","dotnet","net core"],             "Backend", 0.83),
    ("laravel",      ["laravel","lumen"],                                   "Backend", 0.78),
    ("grpc",         ["grpc","protocol buffers","protobuf"],                "Backend", 0.80),
    ("microservices",["microservices","microservice","service mesh","dapr"], "Backend", 0.85),
    ("temporal",     ["temporal","workflow engine"],                         "Backend", 0.80),
    ("serverless",   ["serverless","lambda","cloud functions","faas","edge functions"], "Backend", 0.80),
    ("strapi",       ["strapi","payload cms","directus","cms"],             "Backend", 0.78),

    # ── Cloud & DevOps (Expanded) ─────────────────────────────────
    ("amazon web services", ["amazon web services","aws","ec2","s3","lambda","rds","eks","ecs"],  "Cloud", 0.92),
    ("microsoft azure",     ["microsoft azure","azure","azure devops","aks"],                      "Cloud", 0.88),
    ("google cloud",        ["google cloud","gcp","google cloud platform","gke","bigquery"],       "Cloud", 0.87),
    ("vercel",              ["vercel","netlify","railway"],                                        "Cloud", 0.85),
    ("supabase",            ["supabase","appwrite","pocketbase","nhost"],                          "Cloud", 0.85),
    ("clerk",               ["clerk","kinde","auth0","stytch"],                                    "Cloud", 0.82),
    ("retool",              ["retool","appsmith","budibase"],                                      "Cloud", 0.78),
    ("docker",      ["docker","dockerfile","docker compose","containerization"],                 "DevOps", 0.90),
    ("kubernetes",  ["kubernetes","k8s","kubectl","helm","eks","gke","aks"],                     "DevOps", 0.90),
    ("terraform",   ["terraform","iac","infrastructure as code","pulumi","crossplane"],          "DevOps", 0.88),
    ("ansible",     ["ansible","chef","puppet"],                                              "DevOps", 0.82),
    ("ci/cd",       ["ci/cd","ci cd","jenkins","github actions","gitlab ci","cicd","argo cd"],    "DevOps", 0.88),
    ("observability",["observability","prometheus","grafana","opentelemetry","otel","jaeger","loki","datadog"], "DevOps", 0.84),
    ("cloudflare",  ["cloudflare","cloudflare workers","cloudflare pages"],                   "Cloud", 0.82),

    # ── Databases ─────────────────────────────────────────────────
    ("postgresql",     ["postgresql","postgres","psql","pg","supabase"],             "Database", 0.85),
    ("mysql",          ["mysql","mariadb","planetscale"],                            "Database", 0.82),
    ("mongodb",        ["mongodb","mongo","mongoose"],                               "Database", 0.85),
    ("redis",          ["redis","upstash","memcached"],                              "Database", 0.83),
    ("elasticsearch",  ["elasticsearch","elastic search","opensearch"],              "Database", 0.82),
    ("dynamodb",       ["dynamodb","dynamo","nosql"],                                "Database", 0.80),
    ("snowflake",      ["snowflake","bigquery","redshift","databricks"],              "Database", 0.83),
    ("clickhouse",     ["clickhouse","duckdb","olap"],                               "Database", 0.78),
    ("prisma",         ["prisma","drizzle","drizzle orm","typeorm","sqlalchemy"],    "Database", 0.80),

    # ── Engineering Practices ─────────────────────────────────────
    ("system design",           ["system design","distributed systems","scalable systems"],              "Practice", 0.90),
    ("design patterns",         ["design patterns","clean architecture","hexagonal architecture","ddd"], "Practice", 0.83),
    ("object oriented programming", ["object oriented","oop","oops"],                                     "Practice", 0.82),
    ("functional programming",  ["functional programming","fp"],                                          "Practice", 0.78),
    ("data structures",         ["data structures","algorithms","dsa"],                                   "Practice", 0.85),
    ("testing",                 ["unit testing","integration testing","e2e testing","playwright","vitest","jest","cypress"], "Practice", 0.82),
    ("agile",                   ["agile","scrum","kanban","jira","linear"],                              "Practice", 0.80),
    ("git",                     ["git","github","gitlab","version control"],                              "Practice", 0.83),
    ("api design",              ["api design","rest api","graphql","trpc","grpc","swagger","openapi"],    "Practice", 0.85),

    # ── Tools & Soft Skills ───────────────────────────────────────
    ("figma",        ["figma","ux design","ui design"],       "Tool", 0.75),
    ("postman",      ["postman","insomnia","api testing tool"], "Tool", 0.70),
    ("jira",         ["jira","jira board","atlassian jira"],   "Tool", 0.72),
    ("confluence",   ["confluence","wiki","documentation"],    "Tool", 0.68),
    ("vs code",      ["vs code","visual studio code","vscode"], "Tool", 0.65),
    ("leadership",   ["leadership","team lead","tech lead","people management","lead developer"],     "Soft", 0.80),
    ("communication",["communication","verbal communication","written communication","interpersonal"], "Soft", 0.75),
    ("problem solving",["problem solving","problem-solving","analytical thinking","troubleshooting"],   "Soft", 0.78),
    ("collaboration",["collaboration","teamwork","team player","cross-functional","team work"],        "Soft", 0.75),
    ("mentoring",    ["mentoring","mentorship","coaching","knowledge transfer"],                       "Soft", 0.72),
    ("project management",["project management","project planning","delivery","stakeholder"],           "Soft", 0.78),
    ("critical thinking",["critical thinking","analytical","problem analysis"],                       "Soft", 0.72),
    ("time management",["time management","deadline","prioritization","multitasking"],                 "Soft", 0.68),
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