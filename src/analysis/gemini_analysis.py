import os
import json
import re
from typing import Dict, Any, List
from .gemini_model_manager import gemini_manager

DEFAULT = {
    "ats_score": 0,
    "smart_cv_analysis": {
        "critical_issues": 0,
        "improvements": 0,
        "missing_skills": 0,
        "keywords_found": 0,
    },
    "jd_required_keywords": [],
    "jd_optional_keywords": [],
    "resume_keywords_found": [],
    "resume_keywords_missing": [],
    "weak_language_phrases": [],
    "low_context_phrases": [],
    "technical_skills": [],
    "soft_skills": [],
    "ats_suggestions": [],
}


def _pull_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text or "")
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return DEFAULT.copy()


def enhanced_analysis_with_gemini(resume_text: str, jd_text: str) -> Dict[str, Any]:
    """Enhanced analysis with a more sophisticated prompt."""
    model = gemini_manager.get_model()
    if not model:
        return DEFAULT.copy()

    model_name = gemini_manager.get_working_model_name()

    prompt = f"""
Act as a world-class recruitment expert with deep knowledge of Applicant Tracking Systems (ATS).

Analyze the provided RESUME and JOB DESCRIPTION and return a single JSON object with the following detailed analysis:

- "overall_summary": A brief, professional summary of the candidate's suitability for the role.
- "experience_analysis": An analysis of the candidate's years of experience compared to the job requirements. Mention if the candidate meets, exceeds, or falls short of the required experience.
- "skill_analysis": A detailed breakdown of the candidate's skills, categorizing them into "technical", "soft", and "domain-specific" skills. For each skill, indicate if it's present in the resume and relevant to the job description.
- "education_analysis": An assessment of the candidate's educational background against any requirements in the job description.
- "suitability_score": An integer score from 0 to 100 indicating the overall match between the resume and the job description.
- "report_summary": A bulleted list of the candidate's strengths and weaknesses for the role.

RESUME:
{resume_text}

JOB_DESCRIPTION:
{jd_text}
"""

    try:
        resp = model.generate_content(prompt)
        enhanced_data = _pull_json(resp.text or "")
        print(f"✅ Enhanced analysis completed with {model_name}")
    except Exception as e:
        print(f"❌ Enhanced analysis failed with {model_name}: {str(e)}")
        enhanced_data = {}

    # Combine with the original analysis
    analysis = analyze_with_gemini(resume_text, jd_text)
    analysis.update(enhanced_data)

    return analysis


def analyze_with_gemini(resume_text: str, jd_text: str) -> Dict[str, Any]:
    """Analyze with latest available Gemini model"""

    # Get the best available model
    model = gemini_manager.get_model()

    if not model:
        print("⚠️ No Gemini model available, using fallback analysis")
        return DEFAULT.copy()

    model_name = gemini_manager.get_working_model_name()
    print(f"🤖 Using Gemini model: {model_name}")

    prompt = f"""
Act as an expert ATS optimizer.

Return ONLY a single JSON object with keys:
- jd_required_keywords: array of lowercase must-have keywords/technologies from JD
- jd_optional_keywords: array of lowercase nice-to-have keywords from JD
- resume_keywords_found: array of lowercase JD keywords found in the resume
- resume_keywords_missing: array of lowercase required JD keywords not found in the resume
- weak_language_phrases: array of vague resume phrases to tighten (original casing if possible)
- low_context_phrases: array of resume phrases that need quantification/context (original casing)
- technical_skills: array of lowercase technical skills stack for the role (from JD)
- soft_skills: array of lowercase soft skills for the role (from JD)
- smart_cv_analysis: object with integer fields:
   critical_issues, improvements, missing_skills, keywords_found
- ats_score: integer 0-100 summarizing ATS compatibility
- ats_suggestions: array of short actionable suggestions (<= 8 words each)

RESUME:
{resume_text}

JOB_DESCRIPTION:
{jd_text}
"""

    try:
        resp = model.generate_content(prompt)
        data = _pull_json(resp.text or "")
        print(f"✅ Analysis completed with {model_name}")
    except Exception as e:
        print(f"❌ Analysis failed with {model_name}: {str(e)}")
        data = DEFAULT.copy()

    # Normalize and fill defaults
    out = DEFAULT.copy()
    out.update({k: data.get(k, out[k]) for k in out.keys()})

    # Add model info to response
    out["model_used"] = model_name
    out["model_available"] = True

    # Normalize data...
    sca = out.get("smart_cv_analysis") or {}
    out["smart_cv_analysis"] = {
        "critical_issues": int(sca.get("critical_issues", 0)),
        "improvements": int(sca.get("improvements", 0)),
        "missing_skills": int(
            sca.get("missing_skills", len(out.get("resume_keywords_missing", [])))
        ),
        "keywords_found": int(
            sca.get("keywords_found", len(out.get("resume_keywords_found", [])))
        ),
    }
    out["ats_score"] = int(out.get("ats_score", 0))

    # Normalize keyword lists
    for key in [
        "jd_required_keywords",
        "jd_optional_keywords",
        "resume_keywords_found",
        "resume_keywords_missing",
        "technical_skills",
        "soft_skills",
    ]:
        arr = out.get(key) or []
        out[key] = sorted(list({str(x).strip().lower() for x in arr if str(x).strip()}))

    # Keep phrase casing, remove duplicates
    for key in ["weak_language_phrases", "low_context_phrases", "ats_suggestions"]:
        arr = out.get(key) or []
        out[key] = list(dict.fromkeys([str(x).strip() for x in arr if str(x).strip()]))

    return out
