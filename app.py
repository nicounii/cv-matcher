from flask import Flask, render_template, request, redirect, url_for, session, send_file
from src.utils.report_generator import generate_report
import os
import uuid
import re
import sys
from dotenv import load_dotenv
from flask_session import Session

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.processing.resume_processor import (
    process_resume,
    process_jd,
    clean_for_model,
    normalize_display,
)
from src.matching.matching_engine import calculate_similarity, get_top_job_matches
from src.analysis.gemini_analysis import (
    enhanced_analysis_with_gemini,
    analyze_with_gemini,
)
from src.utils.priority_skills import get_priority_skills
from src.utils.keyword_matcher import present_missing_with_surface
from src.analysis.gemini_model_manager import gemini_manager

load_dotenv()


def startup_model_check():
    """Check model availability at startup"""
    print("🚀 CV Embed Starting...")

    model = gemini_manager.get_model()
    if model:
        model_name = gemini_manager.get_working_model_name()
        print(f"✅ Connected to Gemini: {model_name}")
    else:
        print("⚠️ No Gemini model available - using fallback mode")

    return bool(model)


app = Flask(__name__)
app.secret_key = "your_secret_key_here"
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB limit

# Flask-Session (server-side session) config
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(app.root_path, "flask_session")
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
Session(app)

gemini_available = startup_model_check()


def _escape(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _compile_terms(terms):
    terms = [t for t in (terms or []) if t]
    if not terms:
        return None
    terms = [re.escape(t) for t in sorted(set(terms), key=len, reverse=True)]
    # Use "not a word char" guards instead of \b so punctuation tokens match:
    # (?<!\w)(term)(?!\w) matches at start or after non-word, and before non-word or end.
    return re.compile(r"(?i)(?<!\w)(?:" + "|".join(terms) + r")(?!\w)")


def compute_metrics(resume_text: str, jd_text: str, analysis: dict):
    """Enhanced metrics with model information"""

    # Add model info to analysis
    model_name = gemini_manager.get_working_model_name()
    if model_name:
        analysis["model_info"] = {
            "model_used": model_name,
            "model_generation": "2.5"
            if "2.5" in model_name
            else "2.0"
            if "2.0" in model_name
            else "1.x",
        }

    # Initialize robust processors (if available)
    try:
        from test_models import RobustKeywordExtractor, KeywordValidator

        extractor = RobustKeywordExtractor()
        validator = KeywordValidator()
        # Extract keywords robustly
        robust_keywords = extractor.extract_keywords(resume_text, jd_text)
        # Validate extracted keywords
        validated_keywords = validator.validate_keywords(robust_keywords, jd_text)
    except ImportError:
        # Fallback if classes not available
        validated_keywords = {
            "jd_required_keywords": analysis.get("jd_required_keywords", []),
            "jd_optional_keywords": analysis.get("jd_optional_keywords", []),
            "technical_skills": analysis.get("technical_skills", []),
            "soft_skills": analysis.get("soft_skills", []),
        }

    # Merge with Gemini analysis (keeping AI insights for language improvements)
    analysis.update(validated_keywords)

    # Build role-aware synonyms (now more deterministic)
    from src.utils.dynamic_synonyms import get_role_synonyms

    role_syns = get_role_synonyms(
        validated_keywords.get("jd_required_keywords", []),
        validated_keywords.get("jd_optional_keywords", []),
        validated_keywords.get("technical_skills", []),
        validated_keywords.get("soft_skills", []),
        jd_text,
    )

    # Present/missing analysis with better matching
    from src.utils.keyword_matcher import present_missing_with_surface

    pr, mr, surf_req = present_missing_with_surface(
        resume_text, validated_keywords.get("jd_required_keywords", []), role_syns
    )
    po, mo, surf_opt = present_missing_with_surface(
        resume_text, validated_keywords.get("jd_optional_keywords", []), role_syns
    )
    pt, mt, surf_tech = present_missing_with_surface(
        resume_text, validated_keywords.get("technical_skills", []), role_syns
    )
    ps, ms, surf_soft = present_missing_with_surface(
        resume_text, validated_keywords.get("soft_skills", []), role_syns
    )

    found_all = sorted(list(pr | po))
    missing_all = sorted(list(mr | mo))

    # Deterministic scoring algorithm
    req_total = max(1, len(pr) + len(mr))
    opt_total = max(1, len(po) + len(mo))
    tech_total = max(1, len(pt) + len(mt))

    req_coverage = len(pr) / req_total
    opt_coverage = len(po) / opt_total
    tech_coverage = len(pt) / tech_total

    # Weighted base score
    base_score = 100 * (
        0.50 * req_coverage  # Required skills are most important
        + 0.25 * opt_coverage  # Optional skills
        + 0.20 * tech_coverage  # Technical skills
        + 0.05 * min(1.0, len(ps) / max(1, len(ps) + len(ms)))  # Soft skills bonus
    )

    # Calculate penalties
    critical_penalty = len(mr) * 8  # 8 points per missing required
    weak_language_penalty = len(analysis.get("weak_language_phrases", [])) * 3
    low_context_penalty = len(analysis.get("low_context_phrases", [])) * 2

    total_penalty = critical_penalty + weak_language_penalty + low_context_penalty
    total_penalty = min(total_penalty, base_score * 0.7)  # Cap penalty at 70% of base

    ats_score = max(0, min(100, int(base_score - total_penalty)))

    # Update analysis with robust results
    analysis.update(
        {
            "present_required": sorted(list(pr)),
            "missing_required": sorted(list(mr)),
            "present_optional": sorted(list(po)),
            "missing_optional": sorted(list(mo)),
            "present_technical": sorted(list(pt)),
            "missing_technical": sorted(list(mt)),
            "present_soft": sorted(list(ps)),
            "missing_soft": sorted(list(ms)),
            "resume_keywords_found": found_all,
            "resume_keywords_missing": missing_all,
            "smart_cv_analysis": {
                "critical_issues": len(mr),
                "improvements": len(analysis.get("weak_language_phrases", [])),
                "missing_skills": len(missing_all),
                "keywords_found": len(found_all),
            },
            "ats_score": ats_score,
            "coverage_metrics": {
                "required_coverage": round(req_coverage * 100, 1),
                "optional_coverage": round(opt_coverage * 100, 1),
                "technical_coverage": round(tech_coverage * 100, 1),
            },
        }
    )

    # Build highlighting terms
    resume_good_terms = (
        list(surf_req.values())
        + list(surf_opt.values())
        + list(surf_tech.values())
        + list(surf_soft.values())
    )

    return {"resume_good_terms": resume_good_terms}


def _highlight(text, good=None, critical=None, medium=None, low=None):
    html = _escape(text)
    repl = []

    def wrap(regex, cls):
        nonlocal html
        if not regex:
            return

        def _sub(m):
            token = m.group(0)
            ph = f"@@H{len(repl)}@@"
            repl.append((ph, f'<span class="hl {cls}">{_escape(token)}</span>'))
            return ph

        html = regex.sub(_sub, html)

    # Order matters to avoid overwriting
    wrap(_compile_terms(critical), "hl-critical")
    wrap(_compile_terms(medium), "hl-medium")
    wrap(_compile_terms(low), "hl-low")
    wrap(_compile_terms(good), "hl-good")

    for ph, markup in repl:
        html = html.replace(ph, markup)
    return html


def _ats_level(score: int) -> str:
    if score >= 80:
        return "ats-great"
    if score >= 60:
        return "ats-good"
    if score >= 40:
        return "ats-fair"
    return "ats-poor"


@app.route("/", methods=["GET"])
def index():
    session.clear()
    return render_template("index.html")


@app.route("/process_resume", methods=["POST"])
def process_resume_route():
    resume_text = ""
    if "resume_file" in request.files:
        resume_file = request.files["resume_file"]
        if resume_file.filename != "":
            file_ext = os.path.splitext(resume_file.filename)[1]
            filename = f"resume_{uuid.uuid4().hex}{file_ext}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            resume_file.save(file_path)
            resume_text = process_resume(file_path)
            os.remove(file_path)
    if not resume_text and "resume_text" in request.form:
        resume_text = request.form["resume_text"]
    if resume_text:
        session["resume_text"] = resume_text
        return redirect(url_for("upload_jd"))
    return redirect(url_for("index"))


@app.route("/upload_jd", methods=["GET"])
def upload_jd():
    if "resume_text" not in session:
        return redirect(url_for("index"))
    return render_template("upload.html")


@app.route("/process_jd", methods=["POST"])
def process_jd_route():
    jd_text = ""
    if "jd_file" in request.files:
        jd_file = request.files["jd_file"]
        if jd_file.filename != "":
            file_ext = os.path.splitext(jd_file.filename)[1]
            filename = f"jd_{uuid.uuid4().hex}{file_ext}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            jd_file.save(file_path)
            jd_text = process_jd(file_path)
            os.remove(file_path)
    if not jd_text and "jd_text" in request.form:
        jd_text = request.form["jd_text"]

    if jd_text and "resume_text" in session:
        session["jd_text"] = jd_text  # display/original

        # Similarity + job matches use cleaned copies
        resume_clean = clean_for_model(session["resume_text"])
        jd_clean = clean_for_model(jd_text)

        similarity_score = calculate_similarity(resume_clean, jd_clean)
        top_matches = get_top_job_matches(resume_clean)

        # Gemini ATS analysis uses original text (better extraction)
        analysis = analyze_with_gemini(session["resume_text"], jd_text)

        # Compute present/missing by matching directly against the resume text
        req_terms = analysis.get("jd_required_keywords", [])
        opt_terms = analysis.get("jd_optional_keywords", [])
        tech_terms = analysis.get("technical_skills", [])
        soft_terms = analysis.get("soft_skills", [])

        pr, mr, surf_req = present_missing_with_surface(
            session["resume_text"], req_terms
        )
        po, mo, surf_opt = present_missing_with_surface(
            session["resume_text"], opt_terms
        )
        pt, mt, surf_tech = present_missing_with_surface(
            session["resume_text"], tech_terms
        )
        ps, ms, surf_soft = present_missing_with_surface(
            session["resume_text"], soft_terms
        )

        # JD highlighting (green present, red required miss, blue optional miss)
        jd_hl = _highlight(
            jd_text,
            good=sorted(list(pr | po)),
            critical=sorted(list(mr)),
            low=sorted(list(mo)),
        )

        # Resume highlighting: use actual matched surfaces so users see exactly what matched
        resume_good_terms = (
            list(surf_req.values())
            + list(surf_opt.values())
            + list(surf_tech.values())
            + list(surf_soft.values())
        )
        resume_hl = _highlight(
            session["resume_text"],
            good=resume_good_terms,
            medium=[p.lower() for p in analysis.get("weak_language_phrases", [])],
            low=[p.lower() for p in analysis.get("low_context_phrases", [])],
        )

        # Deterministic counts based on the sets above
        keywords_found_all = sorted(list(pr | po))
        missing_skills_all = sorted(list(mr | mo))

        # Improvements = weak phrases that actually appear in resume
        def occurs_any(text, phrases):
            matched = set()
            for p in phrases or []:
                p2 = (p or "").strip()
                if not p2:
                    continue
                if re.search(
                    re.escape(p2), session["resume_text"], flags=re.IGNORECASE
                ):
                    matched.add(p2.lower())
            return matched

        weak_matched = occurs_any(
            session["resume_text"], analysis.get("weak_language_phrases")
        )
        low_matched = occurs_any(
            session["resume_text"], analysis.get("low_context_phrases")
        )

        improvements_count = len(weak_matched)
        missing_skills_count = len(missing_skills_all)
        keywords_found_count = len(keywords_found_all)

        # Critical issues: missing REQUIRED keywords that are high-priority in your DB
        PRIORITY = get_priority_skills()
        critical_missing = [k for k in mr if k in PRIORITY]
        critical_count = len(critical_missing)

        # Deterministic ATS score from coverage and simple penalties
        req_total = max(1, len(pr) + len(mr))
        opt_total = max(1, len(po) + len(mo))
        req_coverage = len(pr) / req_total
        opt_coverage = len(po) / opt_total
        base = 100 * (0.65 * req_coverage + 0.25 * opt_coverage)
        penalty = (
            5 * critical_count
            + 2 * max(0, improvements_count - 3)
            + 1 * len(low_matched)
        )
        ats_score = int(max(0, min(100, round(base - penalty))))

        # Update analysis so template numbers match exactly
        analysis.update(
            {
                # canonicals for chips/counts
                "present_required": sorted(list(pr)),
                "missing_required": sorted(list(mr)),
                "present_optional": sorted(list(po)),
                "missing_optional": sorted(list(mo)),
                "present_technical": sorted(list(pt)),
                "missing_technical": sorted(list(mt)),
                "present_soft": sorted(list(ps)),
                "missing_soft": sorted(list(ms)),
                "resume_keywords_found": keywords_found_all,
                "resume_keywords_missing": missing_skills_all,
                "critical_missing_required": sorted(list(set(mr) & PRIORITY)),
                "non_critical_missing_required": sorted(list(set(mr) - PRIORITY)),
                "smart_cv_analysis": {
                    "critical_issues": critical_count,
                    "improvements": improvements_count,
                    "missing_skills": missing_skills_count,
                    "keywords_found": keywords_found_count,
                },
                "ats_score": ats_score,
            }
        )

        # Save results
        session["similarity_score"] = similarity_score
        session["top_matches"] = top_matches
        session["analysis"] = analysis
        session["resume_hl"] = resume_hl
        session["jd_hl"] = jd_hl
        session["ats_level"] = _ats_level(analysis.get("ats_score", 0))

        return redirect(url_for("result"))

    return redirect(url_for("upload_jd"))


@app.route("/result", methods=["GET"])
def result():
    if "similarity_score" not in session:
        return redirect(url_for("index"))

    return render_template(
        "result.html",
        similarity_score=session["similarity_score"],
        top_matches=session["top_matches"],
        resume_text=session.get("resume_text", ""),
        jd_text=session.get("jd_text", ""),
        resume_hl=session.get("resume_hl", ""),
        jd_hl=session.get("jd_hl", ""),
        analysis=session.get("analysis", {}),
        ats_level=session.get("ats_level", "ats-poor"),
    )


if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
