"""
resume_analyzer.py  ─  v5.0.0  (AI/NLP Edition)
──────────────────────────────────────────────────
Master orchestrator. Pulls together all NLP modules:

  Module              Purpose
  ──────────────────────────────────────────────────────────────
  nlp_core.py         BM25, TF-IDF, cosine similarity, tokenization
  skill_extractor.py  200+ skills, negation detection, context scoring
  resume_parser.py    Section splitting, contact, experience, education

NLP Techniques Used:
  ✅ BM25 Okapi ranking           best short-doc term relevance
  ✅ TF-IDF cosine similarity     vector space model
  ✅ Jaccard + Overlap coeff      set-based similarity
  ✅ Weighted skill matching      importance from JD frequency × context
  ✅ Negation detection           filters "no experience with X"
  ✅ Required-context boosting    skills near "required", "must" → higher weight
  ✅ Positional weighting         earlier mentions = more important
  ✅ N-gram extraction            bigrams + trigrams catch multi-word skills
  ✅ Lemmatization                "developing" = "develop"
  ✅ Synonym normalization        "reactjs" = "react", "k8s" = "kubernetes"
  ✅ Section-aware parsing        skills in Skills section weighted 2×
  ✅ Ensemble scoring             4 similarity metrics combined

ATS Score (100 pts):
  ┌──────────────────────────────────┬─────────┐
  │ Component                        │ Max Pts │
  ├──────────────────────────────────┼─────────┤
  │ Skill Match (weighted+NLP)       │   38    │
  │ JD Semantic Similarity (BM25)    │   17    │
  │ Section Completeness             │   15    │
  │ Experience Match                 │   12    │
  │ Format & Contact Info            │   10    │
  │ Writing Quality                  │    5    │
  │ Title Alignment Multiplier       │  ×0.85–1.00│
  └──────────────────────────────────┴─────────┘
"""

import re
import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from services.nlp_core import (
    TFIDFVectorizer, BM25, combined_text_similarity,
    extract_weighted_keywords, jaccard_similarity,
    tokenize_and_lemmatize,
)
from services.skill_extractor import (
    extract_skills_advanced, compute_weighted_match,
    get_skill_counts, categorize_skills,
)
from services.resume_parser import (
    split_into_sections, extract_contact_info,
    extract_experience_years, extract_required_experience,
    extract_education_level, extract_gpa, extract_institutions,
    compute_title_alignment, detect_industry,
    analyze_writing_quality, analyze_format,
    detect_sections_present, score_sections,
    EDU_LABELS, SECTION_WEIGHTS,
)


# ══════════════════════════════════════════════════════════════════
#  GRADE / BADGE HELPERS
# ══════════════════════════════════════════════════════════════════

def ats_grade(score: float) -> str:
    if score >= 88: return "Outstanding"
    if score >= 78: return "Excellent"
    if score >= 65: return "Good"
    if score >= 50: return "Average"
    if score >= 35: return "Below Average"
    return "Poor"

def ats_badge(score: float) -> str:
    if score >= 88: return "🏆 Outstanding"
    if score >= 78: return "✅ Excellent"
    if score >= 65: return "🟢 Good"
    if score >= 50: return "🟡 Average"
    if score >= 35: return "🟠 Below Average"
    return "🔴 Poor"

def match_badge(score: float) -> str:
    if score >= 80: return "✅ Strong Match"
    if score >= 65: return "🟢 Good Match"
    if score >= 48: return "🟡 Partial Match"
    if score >= 30: return "🟠 Weak Match"
    return "🔴 Poor Match"

def exp_status(exp_sc: float, req_yrs, res_yrs) -> str:
    if res_yrs and req_yrs and res_yrs > req_yrs * 1.2:
        return "Exceeds Requirement ✅"
    if exp_sc >= 1.0:
        return "Meets Requirement ✅"
    if exp_sc >= 0.7:
        return "Partially Meets ⚠️"
    if req_yrs:
        return "Below Requirement ❌"
    return "No Requirement Stated ℹ️"


# ══════════════════════════════════════════════════════════════════
#  JD KEYWORD COVERAGE  (NLP-enhanced)
# ══════════════════════════════════════════════════════════════════

def compute_jd_keyword_coverage(
    resume_text: str,
    jd_keywords: Dict[str, float],
) -> Tuple[float, List[str], List[str]]:
    """
    Weighted keyword coverage with lemmatization.
    Checks both exact and lemmatized forms.
    """
    text_lower    = resume_text.lower()
    resume_tokens = set(tokenize_and_lemmatize(resume_text))

    covered, missing = [], []
    covered_weight   = 0.0
    total_weight     = sum(jd_keywords.values()) or 1.0

    for kw, weight in jd_keywords.items():
        # Multi-word: substring match
        if ' ' in kw:
            found = kw in text_lower
        else:
            # Single word: exact + lemmatized
            kw_lemma = tokenize_and_lemmatize(kw)
            found = (
                bool(re.search(r'(?<![a-zA-Z0-9])' + re.escape(kw) + r'(?![a-zA-Z0-9])', text_lower))
                or bool(set(kw_lemma) & resume_tokens)
            )

        if found:
            covered.append(kw)
            covered_weight += weight
        else:
            missing.append(kw)

    pct = round(min(covered_weight / total_weight * 100, 100.0), 1)
    return pct, covered[:25], missing[:25]


# ══════════════════════════════════════════════════════════════════
#  EXPERIENCE SCORING
# ══════════════════════════════════════════════════════════════════

def score_experience(resume_text: str, jd_text: str) -> Tuple[float, Optional[int], Optional[float]]:
    required = extract_required_experience(jd_text)
    actual, _ = extract_experience_years(resume_text)

    if required is None:
        return 0.82, None, actual  # neutral positive
    if actual is None:
        return 0.55, required, None  # can't verify

    if actual >= required:
        # Diminishing bonus for over-qualification
        over = actual - required
        bonus = min(over * 0.015, 0.12)
        return min(1.0 + bonus, 1.0), required, actual

    # Under-qualified: proportional score with floor
    ratio = actual / required
    return round(max(ratio, 0.2), 3), required, actual


# ══════════════════════════════════════════════════════════════════
#  MASTER ATS SCORE COMPUTATION
# ══════════════════════════════════════════════════════════════════

def compute_ats_score(
    skill_pct:      float,   # 0-100
    bm25_score:     float,   # 0-100  (semantic similarity)
    section_pct:    float,   # 0-100
    exp_score:      float,   # 0-1
    writing_pts:    float,   # 0-5
    format_pts:     float,   # 0-10
    title_score:    float,   # 0-1
) -> Tuple[float, Dict, Dict]:
    """
    Final ATS score with NLP-enhanced components.

    Component weights chosen based on real ATS system behavior research:
    - Skills are the #1 filter in most ATS (38 pts)
    - Semantic relevance (BM25/TF-IDF) catches what skill list misses (17 pts)
    - Section structure matters to ATS parsers (15 pts)
    - Experience is verified manually but ATS pre-filters (12 pts)
    - Format/contact for ATS parseability (10 pts)
    - Writing quality signals candidate quality (5 pts)
    - Title multiplier: missing title alignment = strong signal of mismatch
    """
    skill_pts   = round(min(skill_pct * 0.38, 38.0), 2)
    bm25_pts    = round(min(bm25_score * 0.17, 17.0), 2)
    section_pts = round(min(section_pct * 0.15, 15.0), 2)
    exp_pts     = round(min(exp_score * 12.0, 12.0), 2)
    fmt_pts     = round(min(format_pts, 10.0), 2)
    writ_pts    = round(min(writing_pts, 5.0), 2)

    raw = skill_pts + bm25_pts + section_pts + exp_pts + fmt_pts + writ_pts

    # Title alignment multiplier: 0.85–1.00
    multiplier = 0.85 + (0.15 * title_score)
    total = round(min(raw * multiplier, 100.0), 1)

    display = {
        "Skill Match (NLP-Weighted)    [max 38]": f"{skill_pts:5.1f} / 38",
        "Semantic Similarity (BM25)    [max 17]": f"{bm25_pts:5.1f} / 17",
        "Section Completeness          [max 15]": f"{section_pts:5.1f} / 15",
        "Experience Match              [max 12]": f"{exp_pts:5.1f} / 12",
        "Format & Contact Info         [max 10]": f"{fmt_pts:5.1f} / 10",
        "Writing Quality               [max  5]": f"{writ_pts:5.1f} / 5",
        "Title Alignment Multiplier          ": f"× {multiplier:.3f}",
        "─────────────────────────────────────": "─────────",
        "TOTAL ATS SCORE               [max 100]": f"{total:5.1f} / 100",
    }

    raw_values = {
        "skill_match":   skill_pts,
        "bm25_semantic": bm25_pts,
        "sections":      section_pts,
        "experience":    exp_pts,
        "format":        fmt_pts,
        "writing":       writ_pts,
        "multiplier":    round(multiplier, 3),
        "total":         total,
    }

    return total, display, raw_values


# ══════════════════════════════════════════════════════════════════
#  JD MATCH SCORE  (ensemble of 4 NLP metrics)
# ══════════════════════════════════════════════════════════════════

def _compute_jd_match_with_sim(
    resume_text:     str,
    jd_text:         str,
    matched_skills:  List[str],
    total_jd_skills: int,
    kw_coverage:     float,
    sim:             Dict,
) -> Tuple[float, Dict]:
    """JD match score using pre-computed similarity dict — avoids recomputing BM25/cosine."""
    skill_sc   = (len(matched_skills) / max(total_jd_skills, 1)) * 100
    kw_sc      = kw_coverage
    bm25_sc    = sim["bm25"]
    jaccard_sc = sim["jaccard"]
    combined = (
        skill_sc   * 0.50 +
        kw_sc      * 0.25 +
        bm25_sc    * 0.15 +
        jaccard_sc * 0.10
    )
    final = round(min(combined, 100.0), 1)
    return final, {
        "skill_overlap_score":   round(skill_sc,   1),
        "keyword_coverage":      round(kw_sc,       1),
        "bm25_semantic_score":   round(bm25_sc,     1),
        "jaccard_similarity":    round(jaccard_sc,  1),
        "tfidf_cosine":          round(sim.get("tfidf_cosine", 0), 1),
        "overlap_coefficient":   round(sim.get("overlap", 0), 1),
        "combined_nlp":          round(sim.get("combined", 0), 1),
        "final_jd_match_score":  final,
    }


def compute_jd_match(
    resume_text:     str,
    jd_text:         str,
    matched_skills:  List[str],
    total_jd_skills: int,
    kw_coverage:     float,
) -> Tuple[float, Dict]:
    """Legacy entry-point — computes similarity internally."""
    sim = combined_text_similarity(resume_text, jd_text)
    skill_sc   = (len(matched_skills) / max(total_jd_skills, 1)) * 100
    kw_sc      = kw_coverage
    bm25_sc    = sim["bm25"]
    jaccard_sc = sim["jaccard"]

    combined = (
        skill_sc   * 0.50 +
        kw_sc      * 0.25 +
        bm25_sc    * 0.15 +
        jaccard_sc * 0.10
    )
    final = round(min(combined, 100.0), 1)

    return final, {
        "skill_overlap_score":   round(skill_sc,   1),
        "keyword_coverage":      round(kw_sc,       1),
        "bm25_semantic_score":   round(bm25_sc,     1),
        "jaccard_similarity":    round(jaccard_sc,  1),
        "tfidf_cosine":          round(sim["tfidf_cosine"], 1),
        "overlap_coefficient":   round(sim["overlap"], 1),
        "final_jd_match_score":  final,
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION SKILL ANALYSIS
#  (Skills found specifically in the Skills section vs rest of resume)
# ══════════════════════════════════════════════════════════════════

def analyze_skill_placement(
    resume_sections: Dict[str, str],
    jd_skills: Dict[str, Dict],
) -> Dict:
    """
    Check if required JD skills appear in the Skills section
    (where ATS parsers weight them most heavily).
    """
    skills_section_text = resume_sections.get("skills", "")
    exp_section_text    = resume_sections.get("experience", "")

    skills_in_section = set(get_skill_counts(skills_section_text).keys()) if skills_section_text else set()
    skills_in_exp     = set(get_skill_counts(exp_section_text).keys()) if exp_section_text else set()
    jd_skill_set      = set(jd_skills.keys())

    in_skills_section   = sorted(jd_skill_set & skills_in_section)
    only_in_experience  = sorted((jd_skill_set & skills_in_exp) - skills_in_section)
    missing_everywhere  = sorted(jd_skill_set - skills_in_section - skills_in_exp)

    return {
        "jd_skills_in_skills_section":  in_skills_section,
        "jd_skills_only_in_experience": only_in_experience,
        "jd_skills_missing_entirely":   missing_everywhere,
        "placement_tip": (
            "✅ Good placement" if len(in_skills_section) >= len(jd_skill_set) * 0.6 else
            "⚠️ Add more JD skills to your Skills section for better ATS parsing"
        ),
    }


# ══════════════════════════════════════════════════════════════════
#  MASTER ANALYZE FUNCTION
# ══════════════════════════════════════════════════════════════════

def analyze_resume(
    resume_text:     str,
    job_description: str,
    candidate_name:  Optional[str] = None,
) -> Dict:
    """
    Full AI/NLP-powered ATS analysis.
    Orchestrates all modules and returns a unified structured report.
    """

    # ── STEP 1: Parse Resume Structure ───────────────────────────
    resume_sections = split_into_sections(resume_text)
    contact_info    = extract_contact_info(resume_text)

    # ── STEP 2: Skill Extraction (both sides) ─────────────────────
    resume_skills_adv = extract_skills_advanced(resume_text, context_boost=False)
    jd_skills_adv     = extract_skills_advanced(job_description, context_boost=True)

    skill_pct, matched_skills, missing_skills, extra_skills, skill_report = \
        compute_weighted_match(resume_skills_adv, jd_skills_adv)

    skill_categories_jd     = categorize_skills(jd_skills_adv)
    skill_categories_resume = categorize_skills(resume_skills_adv)

    # ── STEP 3: JD Keyword Extraction & Coverage ──────────────────
    jd_keywords  = extract_weighted_keywords(job_description, top_n=45)
    kw_coverage, covered_kws, missing_kws = compute_jd_keyword_coverage(
        resume_text, jd_keywords
    )

    # ── STEP 4: NLP Similarity Metrics ────────────────────────────
    # Compute similarity ONCE — reuse bm25 for both JD match and ATS BM25 component
    _sim = combined_text_similarity(resume_text, job_description)
    bm25_score = _sim["bm25"]
    jd_match_score, jd_match_detail = _compute_jd_match_with_sim(
        resume_text, job_description,
        matched_skills, len(jd_skills_adv), kw_coverage, _sim
    )

    # ── STEP 5: Section Analysis ──────────────────────────────────
    sections_present                    = detect_sections_present(resume_text)
    section_pct, present_secs, miss_secs = score_sections(sections_present)

    # ── STEP 6: Skill Placement Analysis ─────────────────────────
    skill_placement = analyze_skill_placement(resume_sections, jd_skills_adv)

    # ── STEP 7: Experience ────────────────────────────────────────
    exp_score, req_yrs, res_yrs = score_experience(resume_text, job_description)
    _, job_timeline             = extract_experience_years(resume_text)

    # ── STEP 8: Education ─────────────────────────────────────────
    edu_req      = extract_education_level(job_description)
    edu_res      = extract_education_level(resume_text)
    edu_match    = edu_res >= edu_req
    gpa          = extract_gpa(resume_text)
    institutions = extract_institutions(resume_text)

    # ── STEP 9: Job Title Alignment ───────────────────────────────
    title_score, jd_titles, res_titles = compute_title_alignment(
        resume_text, job_description
    )

    # ── STEP 10: Industry Domain ──────────────────────────────────
    jd_industry     = detect_industry(job_description)
    resume_industry = detect_industry(resume_text)

    # ── STEP 11: Writing Quality ──────────────────────────────────
    writing = analyze_writing_quality(resume_text)
    verb_count   = writing["power_verbs"]["count"]
    metric_count = writing["quantified_achievements"]["count"]
    writing_pts  = min(verb_count // 3 * 1.5 + (metric_count // 3) * 1.0, 5.0)

    # ── STEP 12: Format & Contact ─────────────────────────────────
    fmt      = analyze_format(resume_text, contact_info)
    fmt_pts  = float(fmt["contact_score"]) + (1.0 if fmt["length_ok"] else 0.0)
    fmt_pts  = min(fmt_pts, 10.0)

    # ── STEP 13: ATS Score ────────────────────────────────────────
    ats_score, score_display, score_raw = compute_ats_score(
        skill_pct   = skill_pct,
        bm25_score  = bm25_score,
        section_pct = section_pct,
        exp_score   = exp_score,
        writing_pts = writing_pts,
        format_pts  = fmt_pts,
        title_score = title_score,
    )

    # ── STEP 14: Overall Compatibility ───────────────────────────
    overall = round(ats_score * 0.58 + jd_match_score * 0.42, 1)

    # ── STEP 15: Smart Insights ───────────────────────────────────
    insights = _generate_insights(
        ats_score, jd_match_score, skill_pct,
        matched_skills, missing_skills,
        present_secs, miss_secs,
        exp_score, req_yrs, res_yrs,
        writing, skill_placement,
    )

    # ══════════════════════════════════════════════════════════════
    #  FINAL REPORT
    # ══════════════════════════════════════════════════════════════
    return {

        # ── Candidate ─────────────────────────────────────────────
        "candidate": {
            "name":  candidate_name or contact_info.get("name") or "Not Provided",
            "email": contact_info.get("email"),
            "phone": contact_info.get("phone"),
        },

        # ── Overall Scores ────────────────────────────────────────
        "scores": {
            "ats_score": {
                "value":       ats_score,
                "out_of":      100,
                "grade":       ats_grade(ats_score),
                "badge":       ats_badge(ats_score),
                "description": "ATS parseability + JD alignment combined score",
            },
            "jd_match_score": {
                "value":       jd_match_score,
                "out_of":      100,
                "badge":       match_badge(jd_match_score),
                "description": "Ensemble NLP similarity: skills + keywords + BM25 + Jaccard",
                "detail":      jd_match_detail,
            },
            "overall_compatibility": {
                "value":       overall,
                "out_of":      100,
                "description": "58% ATS score + 42% JD match (weighted ensemble)",
            },
        },

        # ── Score Breakdown ───────────────────────────────────────
        "score_breakdown": {
            "components_display": score_display,
            "raw_values":         score_raw,
            "nlp_signals": {
                "bm25_raw":           round(bm25_score, 2),
                "tfidf_cosine":       round(jd_match_detail.get("tfidf_cosine", 0), 2),
                "jaccard":            round(jd_match_detail.get("jaccard_similarity", 0), 2),
                "overlap_coeff":      round(jd_match_detail.get("overlap_coefficient", 0), 2),
            },
        },

        # ── Skill Analysis ────────────────────────────────────────
        "skill_analysis": {
            "summary": {
                "match_percentage":    f"{skill_pct}%",
                "total_jd_skills":     len(jd_skills_adv),
                "matched_count":       len(matched_skills),
                "missing_count":       len(missing_skills),
                "extra_in_resume":     len(extra_skills),
            },
            "matched_skills":          matched_skills,
            "missing_skills":          missing_skills,
            "extra_resume_skills":     extra_skills[:15],
            "skill_report":            skill_report,
            "jd_skills_by_category":   skill_categories_jd,
            "resume_skills_by_category": skill_categories_resume,
            "skill_placement":         skill_placement,
        },

        # ── JD Keyword Analysis ───────────────────────────────────
        "jd_keyword_analysis": {
            "coverage_percentage": f"{kw_coverage}%",
            "keywords_covered":    covered_kws,
            "keywords_missing":    missing_kws,
            "top_jd_keywords":     list(jd_keywords.keys())[:20],
            "method":              "TF-IDF positional weighting + n-gram extraction",
        },

        # ── Section Analysis ──────────────────────────────────────
        "section_analysis": {
            "completeness":        f"{round(len(present_secs)/max(len(SECTION_WEIGHTS),1)*100)}%",
            "section_score":       f"{round(section_pct, 1)}/100",
            "sections_present":    present_secs,
            "sections_missing":    miss_secs,
            "section_detail": {
                sec: {
                    "present": sections_present.get(sec, False),
                    "weight":  SECTION_WEIGHTS.get(sec, 0),
                    "status":  "✅" if sections_present.get(sec) else "❌",
                }
                for sec in SECTION_WEIGHTS
            },
        },

        # ── Experience ────────────────────────────────────────────
        "experience": {
            "required_by_jd":     f"{req_yrs} year{'s' if req_yrs != 1 else ''}" if req_yrs else "Not specified",
            "detected_in_resume": f"~{res_yrs:.1f} years" if res_yrs else "Not detected",
            "match_score":        f"{round(exp_score * 100)}%",
            "match_status":       exp_status(exp_score, req_yrs, res_yrs),
            "job_timeline":       job_timeline[:5],
        },

        # ── Education ─────────────────────────────────────────────
        "education": {
            "required_by_jd":       EDU_LABELS.get(edu_req, "Not Specified"),
            "detected_in_resume":   EDU_LABELS.get(edu_res, "Not Detected"),
            "match_status":         "Meets Requirement ✅" if edu_match else "Below Requirement ❌",
            "education_gap":        max(0, edu_req - edu_res),
            "gpa_cgpa":             gpa or "Not found",
            "institutions":         institutions,
        },

        # ── Job Title Alignment ───────────────────────────────────
        "job_title_alignment": {
            "alignment_score":    f"{round(title_score * 100)}%",
            "status": (
                "Strong Match ✅"   if title_score >= 0.9 else
                "Partial Match ⚠️" if title_score >= 0.5 else
                "Weak Match ❌"
            ),
            "roles_in_jd":         jd_titles or ["Not specified"],
            "roles_in_resume":     res_titles or ["Not detected"],
            "multiplier_applied":  f"× {round(0.85 + 0.15 * title_score, 3)}",
        },

        # ── Industry Domain ───────────────────────────────────────
        "industry_domain": {
            "jd_industry":          jd_industry or ["General / Not specified"],
            "resume_industry":      resume_industry or ["General / Not detected"],
            "domain_alignment": (
                "✅ Aligned"   if set(jd_industry) & set(resume_industry) else
                "⚠️ Partial"  if jd_industry and resume_industry else
                "ℹ️ Not Determined"
            ),
        },

        # ── Writing Quality ───────────────────────────────────────
        "writing_quality": writing,

        # ── Format & Contact ──────────────────────────────────────
        "format_and_contact": {
            **fmt["contact_detail"],
            "email_value":   contact_info.get("email"),
            "phone_value":   contact_info.get("phone"),
            "linkedin_url":  contact_info.get("linkedin"),
            "github_url":    contact_info.get("github"),
            "portfolio_url": contact_info.get("portfolio"),
            "word_count":    fmt["word_count"],
            "length_status": fmt["length_status"],
            "contact_score": f"{fmt['contact_score']}/10",
        },

        # ── AI Insights ───────────────────────────────────────────
        "ai_insights": insights,

        # ── NLP Methods Used ──────────────────────────────────────
        "analysis_metadata": {
            "version":          "5.0.0",
            "nlp_methods": [
                "BM25 Okapi ranking",
                "TF-IDF cosine similarity",
                "Jaccard similarity (with bigrams)",
                "Overlap coefficient",
                "Weighted skill matching with context boost",
                "Negation detection",
                "Positional TF-IDF keyword extraction",
                "N-gram extraction (uni/bi/trigrams)",
                "Rule-based lemmatization",
                "Synonym normalization",
                "Section-boundary parsing",
                "Experience timeline parsing",
                "Writing quality analysis",
            ],
            "skill_db_size":   "200+ skills with aliases",
            "scoring_model":   "Ensemble weighted scoring with title multiplier",
        },
    }


# ══════════════════════════════════════════════════════════════════
#  AI INSIGHTS GENERATOR
# ══════════════════════════════════════════════════════════════════

def _generate_insights(
    ats_score:       float,
    jd_match:        float,
    skill_pct:       float,
    matched_skills:  List[str],
    missing_skills:  List[str],
    present_secs:    List[str],
    miss_secs:       List[str],
    exp_score:       float,
    req_yrs,
    res_yrs,
    writing:         Dict,
    skill_placement: Dict,
) -> Dict:
    """
    Generate data-driven insights — no generic advice,
    only insights specific to this resume's actual gaps.
    """
    strengths   = []
    gaps        = []
    quick_wins  = []  # high impact, easy to fix

    # ── Strengths ─────────────────────────────────────────────────
    if skill_pct >= 75:
        strengths.append(f"Strong skill match — {skill_pct}% of required skills present")
    if "experience" in present_secs and exp_score >= 1.0:
        strengths.append("Experience meets or exceeds JD requirements")
    if writing["power_verbs"]["count"] >= 10:
        strengths.append(f"Strong action verbs — {writing['power_verbs']['count']} power verbs found")
    if writing["quantified_achievements"]["count"] >= 5:
        strengths.append(f"Good quantified impact — {writing['quantified_achievements']['count']} metrics found")
    if len(present_secs) >= 6:
        strengths.append(f"Comprehensive resume structure — {len(present_secs)}/8 sections present")

    # ── Gaps ──────────────────────────────────────────────────────
    if missing_skills:
        top_missing = missing_skills[:5]
        gaps.append(f"Missing {len(missing_skills)} JD skills — top gaps: {', '.join(top_missing)}")
    if miss_secs:
        gaps.append(f"Missing resume sections: {', '.join(miss_secs)}")
    if exp_score < 0.7 and req_yrs:
        gaps.append(f"Experience gap — JD needs {req_yrs}yr, resume shows ~{res_yrs or 0:.0f}yr")
    if writing["power_verbs"]["count"] < 5:
        gaps.append("Very few action verbs — bullets sound passive and weak")
    if writing["quantified_achievements"]["count"] < 2:
        gaps.append("No quantified impact — add numbers, percentages, and metrics")
    if writing["weak_verbs"]["count"] > 5:
        gaps.append(f"Too many weak verbs ({', '.join(writing['weak_verbs']['found'][:4])}) — replace with action verbs")

    # ── Quick Wins ────────────────────────────────────────────────
    placement_missing = skill_placement.get("jd_skills_only_in_experience", [])
    if placement_missing:
        quick_wins.append(
            f"Move these skills to your Skills section for better ATS parsing: "
            f"{', '.join(placement_missing[:5])}"
        )
    if miss_secs:
        critical = [s for s in miss_secs if s in ["summary", "skills", "contact"]]
        if critical:
            quick_wins.append(f"Add missing critical sections: {', '.join(critical)}")
    if missing_skills[:3]:
        quick_wins.append(
            f"Add these high-priority missing skills if you have them: "
            f"{', '.join(missing_skills[:3])}"
        )
    if writing["quantified_achievements"]["count"] < 3:
        quick_wins.append(
            "Add 3–5 quantified achievements (%, $, team size, users) to your experience bullets"
        )

    return {
        "strengths":   strengths[:5],
        "gaps":        gaps[:5],
        "quick_wins":  quick_wins[:4],
        "summary": (
            f"ATS Score: {ats_score}/100 ({ats_grade(ats_score)}). "
            f"JD Match: {jd_match}/100. "
            f"Skill match: {skill_pct}% ({len(matched_skills)} matched, {len(missing_skills)} missing). "
            f"{len(present_secs)}/8 sections present."
        ),
    }