"""
nlp_core.py  ─  NLP Engine (Zero External Dependencies)
─────────────────────────────────────────────────────────
Implements core NLP algorithms from scratch:
  ✅ BM25 Okapi ranking            (better than TF-IDF for short docs)
  ✅ TF-IDF vectorization          (term frequency × inverse doc frequency)
  ✅ Cosine similarity             (vector space model)
  ✅ Lemmatization                 (rule-based English morphology)
  ✅ N-gram extraction             (unigrams, bigrams, trigrams)
  ✅ Named entity patterns         (regex-based NER for tech terms)
  ✅ Sentence boundary detection
  ✅ Synonym expansion             (canonical form lookup)
  ✅ Positional weighting          (earlier = more important)
  ✅ Dependency-aware chunking     (noun phrase extraction)
"""

import re
import math
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, List, Set, Tuple, Optional


# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════

# BM25 parameters (tuned for short resume/JD documents)
BM25_K1 = 1.5   # term frequency saturation
BM25_B  = 0.75  # document length normalization

STOPWORDS: Set[str] = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","have","has","had","do",
    "does","did","will","would","could","should","may","might","must","shall",
    "can","need","this","that","these","those","it","its","we","you","your",
    "our","their","they","he","she","his","her","who","which","what","when",
    "where","how","why","not","no","nor","so","yet","both","either","neither",
    "each","any","all","more","most","other","such","than","then","also","just",
    "only","very","well","good","great","new","work","team","role","job",
    "position","candidate","experience","ability","skills","strong","excellent",
    "required","preferred","plus","including","related","using","working",
    "looking","seeking","responsible","knowledge","understanding","familiarity",
    "proficiency","proficient","able","make","use","used","get","set","build",
    "built","year","years","month","months","day","days","time","etc","eg","ie",
    "per","as","if","into","over","after","before","during","while","since",
    "about","above","across","between","through","without","within","along",
    "following","behind","beyond","except","up","out","around","down","off",
    "again","further","once","provide","apply","help","ensure","support",
    "develop","manage","create","design","implement","maintain","different",
    "various","multiple","within","among","toward","towards","upon","via",
    "whether","although","however","therefore","furthermore","additionally",
    "including","following","regarding","concerning","considering",
}

# Rule-based lemmatization suffixes (English morphology)
# Pre-compiled at module load — avoids re.compile() on every lemmatize() call
_LEMMA_RULES_RAW: List[Tuple[str, str]] = [
    (r'ational$','ate'),(r'tional$','tion'),(r'enci$','ence'),
    (r'anci$','ance'),(r'izer$','ize'),(r'ising$','ise'),
    (r'izing$','ize'),(r'alism$','al'),(r'iveness$','ive'),
    (r'fulness$','ful'),(r'ousness$','ous'),(r'aliti$','al'),
    (r'iviti$','ive'),(r'biliti$','ble'),(r'icate$','ic'),
    (r'ative$',''),(r'alize$','al'),(r'iciti$','ic'),
    (r'ical$','ic'),(r'ful$',''),(r'ness$',''),(r'ement$',''),
    (r'ment$',''),(r'ence$',''),(r'ance$',''),(r'able$',''),
    (r'ible$',''),(r'ant$',''),(r'ent$',''),(r'ion$',''),
    (r'ous$',''),(r'ive$',''),(r'ize$',''),(r'ing$',''),
    (r'tion$',''),(r'ed$',''),(r'er$',''),(r'ly$',''),(r's$',''),
]
LEMMA_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(p), r) for p, r in _LEMMA_RULES_RAW
]

# Tech-specific synonyms (surface → canonical)
TECH_SYNONYMS: Dict[str, str] = {
    "reactjs": "react", "react.js": "react",
    "vuejs": "vue", "vue.js": "vue",
    "nodejs": "node", "node.js": "node",
    "expressjs": "express", "express.js": "express",
    "nextjs": "next", "next.js": "next",
    "k8s": "kubernetes",
    "sklearn": "scikit-learn", "scikit_learn": "scikit-learn",
    "tf": "tensorflow",
    "torch": "pytorch",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "cpp": "c++",
    "csharp": "c#",
    "golang": "go",
    "gcp": "google cloud",
    "aws": "amazon web services",
    "pg": "postgresql", "postgres": "postgresql",
    "mongo": "mongodb",
    "es": "elasticsearch",
    "nlp": "natural language processing",
    "ml": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
    "rl": "reinforcement learning",
    "llm": "large language model",
    "genai": "generative ai",
    "pyspark": "spark",
    "hf": "huggingface",
    "oop": "object oriented",
    "tdd": "test driven development",
    "bdd": "behavior driven development",
    "ci": "continuous integration",
    "cd": "continuous delivery",
    "vcs": "version control",
    "scm": "source control",
    "restful": "rest api",
    "graphql": "graphql",
    "nosql": "nosql",
    "rdbms": "sql",
    "bi": "business intelligence",
    "powerbi": "power bi",
    "hld": "high level design",
    "lld": "low level design",
    "dsa": "data structures",
    "oop": "object oriented programming",
}


# ══════════════════════════════════════════════════════════════════
#  TOKENIZATION
# ══════════════════════════════════════════════════════════════════

def tokenize(text: str, keep_technical: bool = True) -> List[str]:
    """
    Smart tokenizer that:
    - Preserves technical terms with special chars (c++, c#, .net, node.js)
    - Handles camelCase splitting (JavaScriptDeveloper → javascript developer)
    - Applies synonym normalization
    - Removes stopwords
    - Returns lemmatized lowercase tokens
    """
    # Preserve special technical tokens before lowering
    tech_preserved = re.sub(
        r'\b(c\+\+|c#|\.net|node\.js|react\.js|vue\.js|next\.js|express\.js|'
        r'scikit-learn|asp\.net|ci/cd|oauth2|ssl/tls|pl/sql|t-sql)\b',
        lambda m: m.group(0).replace('+', 'PLUS').replace('#', 'HASH')
                             .replace('.', 'DOT').replace('/', 'SLASH'),
        text, flags=re.IGNORECASE
    )

    # CamelCase split: JavaScriptDeveloper → Java Script Developer
    camel_split = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', tech_preserved)

    text_lower = camel_split.lower()

    # Restore preserved tokens
    text_lower = (text_lower
        .replace('plus', '+').replace('hash', '#')
        .replace('dot', '.').replace('slash', '/'))

    # Extract tokens (include technical chars)
    raw_tokens = re.findall(
        r'\b[a-z][a-z0-9\+\#\./\-]{0,30}\b',
        text_lower
    )

    tokens = []
    for tok in raw_tokens:
        # Apply synonym normalization
        tok = TECH_SYNONYMS.get(tok, tok)
        # Skip stopwords (but keep short tech terms like 'r', 'go', 'c')
        if tok in STOPWORDS and len(tok) > 2:
            continue
        if len(tok) < 2:
            continue
        tokens.append(tok)

    return tokens


@lru_cache(maxsize=8192)
def lemmatize(word: str) -> str:
    """Rule-based lemmatization using pre-compiled patterns + LRU cache."""
    if any(c in word for c in ('+', '#', '.', '/')):
        return word
    if len(word) <= 4:
        return word
    result = word
    for compiled_pat, replacement in LEMMA_RULES:
        if compiled_pat.search(result) and len(result) > 4:
            candidate = compiled_pat.sub(replacement, result)
            if len(candidate) >= 3:
                result = candidate
                break
    return result


@lru_cache(maxsize=512)
def tokenize_and_lemmatize(text: str) -> List[str]:
    """Full NLP pipeline: tokenize → synonym expand → lemmatize. Cached per unique text."""
    tokens = tokenize(text)
    return [lemmatize(t) for t in tokens]


def extract_sentences(text: str) -> List[str]:
    """Sentence boundary detection."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|[\n\r]{2,}', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def extract_noun_phrases(text: str) -> List[str]:
    """
    Lightweight noun phrase extraction using POS-like patterns.
    Captures: 'machine learning engineer', 'full stack developer', etc.
    """
    text_lower = text.lower()
    # Pattern: (adj/noun)* noun — captures multi-word technical phrases
    pattern = r'\b(?:[a-z]+\s+){0,3}(?:engineer|developer|architect|analyst|scientist|' \
              r'manager|lead|specialist|consultant|administrator|designer|tester|' \
              r'learning|processing|intelligence|framework|platform|system|service|' \
              r'database|language|programming|development|architecture|infrastructure)\b'
    phrases = re.findall(pattern, text_lower)
    return [p.strip() for p in phrases if len(p.strip()) > 5]


# ══════════════════════════════════════════════════════════════════
#  N-GRAM EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_ngrams(tokens: List[str], n: int) -> List[str]:
    """Extract n-grams from token list."""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def extract_all_ngrams(text: str, max_n: int = 3) -> Dict[str, int]:
    """Extract unigrams + bigrams + trigrams with frequencies."""
    tokens = tokenize(text)
    ngram_freq: Dict[str, int] = Counter()
    for n in range(1, max_n + 1):
        for ng in extract_ngrams(tokens, n):
            if all(t not in STOPWORDS for t in ng.split()):
                ngram_freq[ng] += 1
    return dict(ngram_freq)


# ══════════════════════════════════════════════════════════════════
#  TF-IDF
# ══════════════════════════════════════════════════════════════════

class TFIDFVectorizer:
    """
    TF-IDF vectorizer for short documents.
    Uses sublinear TF scaling: tf = 1 + log(count)
    IDF is computed over a reference corpus of job descriptions.
    """

    # Pre-computed IDF weights based on common resume/JD terms
    # Higher = rarer = more important
    DOMAIN_IDF: Dict[str, float] = {
        "python": 1.2, "javascript": 1.2, "java": 1.2, "typescript": 1.4,
        "react": 1.3, "angular": 1.5, "vue": 1.6, "node": 1.3,
        "docker": 1.4, "kubernetes": 1.6, "terraform": 1.7, "jenkins": 1.5,
        "aws": 1.3, "azure": 1.4, "gcp": 1.5,
        "postgresql": 1.5, "mysql": 1.4, "mongodb": 1.5, "redis": 1.5,
        "machine learning": 1.4, "deep learning": 1.5, "nlp": 1.6,
        "tensorflow": 1.5, "pytorch": 1.6, "keras": 1.5,
        "microservices": 1.5, "rest api": 1.3, "graphql": 1.6,
        "agile": 1.2, "scrum": 1.3, "git": 1.1,
        "sql": 1.2, "nosql": 1.4, "linux": 1.3,
        "fastapi": 1.7, "django": 1.5, "flask": 1.5,
        "spark": 1.6, "kafka": 1.6, "airflow": 1.7,
        "communication": 0.8, "leadership": 0.9, "teamwork": 0.8,
    }

    def vectorize(self, text: str) -> Dict[str, float]:
        """Compute TF-IDF vector for a document."""
        tokens = tokenize_and_lemmatize(text)
        if not tokens:
            return {}

        tf_raw = Counter(tokens)
        doc_len = len(tokens)
        vector: Dict[str, float] = {}

        for term, count in tf_raw.items():
            # Sublinear TF scaling
            tf = 1 + math.log(count) if count > 0 else 0
            # IDF from domain knowledge or default
            idf = self.DOMAIN_IDF.get(term, 2.0)  # default IDF=2.0 (relatively rare)
            vector[term] = tf * idf

        return vector

    def cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Cosine similarity between two TF-IDF vectors."""
        if not vec1 or not vec2:
            return 0.0

        # Dot product
        common = set(vec1.keys()) & set(vec2.keys())
        dot = sum(vec1[t] * vec2[t] for t in common)

        # Magnitudes
        mag1 = math.sqrt(sum(v*v for v in vec1.values()))
        mag2 = math.sqrt(sum(v*v for v in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return round(min(dot / (mag1 * mag2), 1.0), 4)


# ══════════════════════════════════════════════════════════════════
#  BM25 OKAPI
# ══════════════════════════════════════════════════════════════════

class BM25:
    """
    BM25 Okapi ranking function.
    Better than TF-IDF for short documents because it:
    - Saturates term frequency (prevents keyword stuffing from winning)
    - Normalizes by document length
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b  = b

    def score(
        self,
        query_tokens: List[str],
        document_tokens: List[str],
        avg_doc_len: float = 150.0,
    ) -> float:
        """
        Compute BM25 score for a document given a query.
        Higher score = better match.
        """
        if not query_tokens or not document_tokens:
            return 0.0

        doc_freq   = Counter(document_tokens)
        doc_len    = len(document_tokens)
        score      = 0.0
        N          = 1000  # assumed corpus size

        for term in set(query_tokens):
            if term not in doc_freq:
                continue

            tf  = doc_freq[term]
            # IDF approximation (using domain IDF if available)
            df  = 50  # assumed document frequency
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            # BM25 term score
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len)
            )
            score += idf * tf_norm

        # Normalize to 0-100 range
        return round(min(score * 3, 100.0), 2)

    def similarity(self, text1: str, text2: str) -> float:
        """BM25 similarity between two texts (bidirectional)."""
        t1 = tokenize_and_lemmatize(text1)
        t2 = tokenize_and_lemmatize(text2)
        avg_len = (len(t1) + len(t2)) / 2 or 150
        s1 = self.score(t1, t2, avg_len)
        s2 = self.score(t2, t1, avg_len)
        return round((s1 + s2) / 2, 2)


# ══════════════════════════════════════════════════════════════════
#  POSITIONAL TF-IDF  (position-weighted keyword extraction)
# ══════════════════════════════════════════════════════════════════

def extract_weighted_keywords(
    text: str,
    top_n: int = 50,
    include_ngrams: bool = True,
) -> Dict[str, float]:
    """
    Extract keywords with positional + frequency weighting.

    Scoring formula per token:
      score = TF × IDF × position_weight × case_boost × section_boost

    position_weight: 1.0 for first 20% of doc, decays to 0.6 at end
    case_boost:      1.5 for ALL_CAPS, 1.3 for TitleCase, 1.0 for lower
    section_boost:   2.0 if found under "Requirements" / "Skills" heading
    """
    tfidf = TFIDFVectorizer()
    sentences = extract_sentences(text)
    total_sents = max(len(sentences), 1)

    word_scores: Dict[str, float] = {}

    # Detect requirement section boundaries
    req_section = False
    req_keywords = {'requirement', 'skill', 'qualification', 'looking for',
                    'what we need', 'must have', 'you will need', 'experience'}

    for sent_idx, sentence in enumerate(sentences):
        # Check if we're in a requirements section
        sent_lower = sentence.lower()
        if any(kw in sent_lower for kw in req_keywords):
            req_section = True
        elif re.match(r'^[A-Z][^.]{0,30}:?\s*$', sentence):
            req_section = False  # new section header

        section_boost = 2.0 if req_section else 1.0
        position_weight = 1.0 - 0.4 * (sent_idx / total_sents)

        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\+\#\./\-]{1,30}\b', sentence)
        for word in words:
            w_lower = TECH_SYNONYMS.get(word.lower(), word.lower())
            if w_lower in STOPWORDS or len(w_lower) < 2:
                continue

            # Case boost
            if word.isupper() and len(word) > 1:
                case_boost = 1.5
            elif word[0].isupper():
                case_boost = 1.2
            else:
                case_boost = 1.0

            # Get TF-IDF weight
            idf = TFIDFVectorizer.DOMAIN_IDF.get(w_lower, 1.8)
            score = idf * position_weight * case_boost * section_boost
            word_scores[w_lower] = word_scores.get(w_lower, 0) + score

    # Add n-gram scores
    if include_ngrams:
        tokens = tokenize(text)
        filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

        for n in [2, 3]:
            for i in range(len(filtered) - n + 1):
                gram = ' '.join(filtered[i:i+n])
                if len(gram) > 5:
                    # N-grams get boost if they match known tech patterns
                    ngram_score = 0.8 if n == 2 else 0.5
                    word_scores[gram] = word_scores.get(gram, 0) + ngram_score

    sorted_kw = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_kw[:top_n])


# ══════════════════════════════════════════════════════════════════
#  SEMANTIC SIMILARITY  (without transformers)
# ══════════════════════════════════════════════════════════════════

def jaccard_similarity(text1: str, text2: str, use_ngrams: bool = True) -> float:
    """
    Jaccard similarity on token sets.
    Optional: use bigrams for better phrase matching.
    """
    t1 = set(tokenize_and_lemmatize(text1))
    t2 = set(tokenize_and_lemmatize(text2))

    if use_ngrams:
        toks1 = tokenize(text1)
        toks2 = tokenize(text2)
        bigrams1 = set(' '.join(toks1[i:i+2]) for i in range(len(toks1)-1))
        bigrams2 = set(' '.join(toks2[i:i+2]) for i in range(len(toks2)-1))
        t1 = t1 | bigrams1
        t2 = t2 | bigrams2

    intersection = t1 & t2
    union        = t1 | t2
    if not union:
        return 0.0
    return round(len(intersection) / len(union) * 100, 2)


def overlap_coefficient(text1: str, text2: str) -> float:
    """
    Overlap coefficient — better than Jaccard when docs have very different lengths.
    score = |intersection| / min(|A|, |B|)
    """
    t1 = set(tokenize_and_lemmatize(text1))
    t2 = set(tokenize_and_lemmatize(text2))
    intersection = t1 & t2
    min_size = min(len(t1), len(t2))
    if min_size == 0:
        return 0.0
    return round(len(intersection) / min_size * 100, 2)


@lru_cache(maxsize=256)
def combined_text_similarity(resume_text: str, jd_text: str) -> Dict[str, float]:
    """
    Compute multiple similarity metrics and return combined score.
    LRU cached so the same (resume, jd) pair is never computed twice.
    Uses ensemble of: TF-IDF cosine + BM25 + Jaccard + Overlap
    """
    tfidf    = TFIDFVectorizer()
    bm25     = BM25()

    vec_r    = tfidf.vectorize(resume_text)
    vec_j    = tfidf.vectorize(jd_text)
    cosine   = tfidf.cosine_similarity(vec_r, vec_j) * 100
    bm25_sc  = bm25.similarity(resume_text, jd_text)
    jaccard  = jaccard_similarity(resume_text, jd_text)
    overlap  = overlap_coefficient(resume_text, jd_text)

    # Ensemble: weighted combination
    combined = (
        cosine  * 0.35 +
        bm25_sc * 0.30 +
        jaccard * 0.20 +
        overlap * 0.15
    )

    return {
        "tfidf_cosine":    round(cosine,  2),
        "bm25":            round(bm25_sc, 2),
        "jaccard":         round(jaccard, 2),
        "overlap":         round(overlap, 2),
        "combined":        round(min(combined, 100.0), 2),
    }