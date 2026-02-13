import os
import re
import json
import joblib
from typing import Dict, List, Iterable
from ..core.config import SKILLS_DB_PATH
from ..analysis.gemini_model_manager import gemini_manager
from functools import lru_cache

class DynamicSynonyms:
    def __init__(self):
        self.skills_db = self._load_skills_db()
        self.synonym_cache = {}
    
    @lru_cache(maxsize=1000)
    def _load_skills_db(self) -> List[str]:
        if os.path.exists(SKILLS_DB_PATH):
            try:
                data = joblib.load(SKILLS_DB_PATH)
                return [str(x).strip() for x in (data or []) if str(x).strip()]
            except Exception:
                return []
        return []
    
    def _ai_synonyms_for_terms(terms: Iterable[str], jd_text: str) -> Dict[str, List[str]]:
        """Ask Gemini for role-aware aliases using latest model"""
        
        model = gemini_manager.get_model()
        if not model:
            return {}

        terms_list = [str(t).strip().lower() for t in (terms or []) if str(t).strip()]
        if not terms_list:
            return {}

        prompt = f"""
        Return only JSON: map each canonical term (lowercase) to a short array (0-6 items) of
        lowercase synonyms/aliases/abbreviations/brand variants as commonly used for THIS job role.
        Include spacing/hyphen/dot variants and known abbreviations, but no definitions.

        TERMS:
        {terms_list}

        JOB_DESCRIPTION:
        {jd_text}
        """
        try:
            resp = model.generate_content(prompt)
            data = _safe_json(resp.text or "")
            
            # Sanitize response
            out: Dict[str, List[str]] = {}
            for k, v in (data.items() if isinstance(data, dict) else []):
                if not isinstance(k, str):
                    continue
                key = k.strip().lower()
                arr = []
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, str):
                            s = it.strip().lower()
                           if s and s != key and s not in arr:
                                arr.append(s)
                out[key] = arr[:6]
            
            model_name = gemini_manager.get_working_model_name()
            print(f"🔄 Synonyms generated with {model_name}")
            
            return out
        except Exception as e:
            print(f"❌ Synonym generation failed: {str(e)}")
            return {}
    
    def _normalize_key(self, s: str) -> str:
        """Normalize for comparison"""
        return re.sub(r'[\s\-/_.]+', '', (s or '').lower())
    
    def get_deterministic_synonyms(self, term: str) -> List[str]:
        """Get deterministic synonyms based on rules"""
        term = term.lower().strip()
        synonyms = set()
        
        # Common programming language synonyms
        lang_synonyms = {
            'javascript': ['js', 'ecmascript'],
            'typescript': ['ts'],
            'python': ['py'],
            'c++': ['cpp', 'cplusplus'],
            'c#': ['csharp', 'dotnet'],
            'postgresql': ['postgres', 'psql'],
            'mysql': ['my sql'],
            'mongodb': ['mongo'],
            'kubernetes': ['k8s'],
            'docker': ['containerization'],
            'amazon web services': ['aws'],
            'google cloud': ['gcp', 'google cloud platform'],
            'microsoft azure': ['azure'],
        }
        
        # Add known synonyms
        if term in lang_synonyms:
            synonyms.update(lang_synonyms[term])
        
        # Add variations (with/without spaces, hyphens)
        if '-' in term:
            synonyms.add(term.replace('-', ' '))
            synonyms.add(term.replace('-', ''))
        
        if ' ' in term:
            synonyms.add(term.replace(' ', '-'))
            synonyms.add(term.replace(' ', ''))
        
        # Add database variants
        normalized = self._normalize_key(term)
        for skill in self.skills_db:
            if self._normalize_key(skill) == normalized and skill.lower() != term:
                synonyms.add(skill.lower())
        
        return list(synonyms)
    
    def get_role_synonyms(
        self,
        req_terms: List[str],
        opt_terms: List[str], 
        tech_terms: List[str],
        soft_terms: List[str],
        jd_text: str
    ) -> Dict[str, List[str]]:
        """Get comprehensive synonyms with caching"""
        cache_key = hash(f"{','.join(req_terms)}{','.join(opt_terms)}{jd_text[:100]}")
        
        if cache_key in self.synonym_cache:
            return self.synonym_cache[cache_key]
        
        all_terms = []
        for bucket in (req_terms, opt_terms, tech_terms, soft_terms):
            all_terms.extend([str(t).strip().lower() for t in (bucket or []) if str(t).strip()])
        
        # Get unique terms
        unique_terms = list(dict.fromkeys(all_terms))
        
        synonyms = {}
        for term in unique_terms:
            # Start with deterministic synonyms
            term_synonyms = self.get_deterministic_synonyms(term)
            
            # Add AI synonyms if available (with fallback)
            ai_synonyms = self._get_cached_ai_synonyms(term, jd_text)
            term_synonyms.extend(ai_synonyms)
            
            # Remove duplicates and limit
            synonyms[term] = list(dict.fromkeys(term_synonyms))[:8]
        
        self.synonym_cache[cache_key] = synonyms
        return synonyms
    
    def _get_cached_ai_synonyms(self, term: str, jd_text: str) -> List[str]:
        """Get AI synonyms with caching and fallback"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return []
            
            genai.configure(api_key=api_key)
            # With this:
            def get_working_model():
                model_names = ["gemini-1.5-flash", "gemini-1.0-pro", "gemini-pro"]
    
                for model_name in model_names:
                    try:
                        return genai.GenerativeModel(
                            model_name,
                            generation_config={"temperature": 0.0, "top_p": 1.0, "top_k": 1}
                        )
                    except Exception:
                        continue
                raise Exception("No working Gemini model found")

            model = get_working_model()
            
            prompt = f"""
Return only a JSON array of 3-5 lowercase synonyms/abbreviations for "{term}" in this job context.
No definitions, no explanations, just the array.

Context: {jd_text[:300]}

Example format: ["synonym1", "synonym2", "synonym3"]
"""
            
            response = model.generate_content(prompt)
            result = json.loads(response.text.strip())
            
            if isinstance(result, list):
                return [str(s).lower().strip() for s in result[:5] if isinstance(s, str)]
            
        except Exception:
            pass
        
        return []

# Update the original function to use the class
def get_role_synonyms(req_terms, opt_terms, tech_terms, soft_terms, jd_text):
    synonym_engine = DynamicSynonyms()
    return synonym_engine.get_role_synonyms(req_terms, opt_terms, tech_terms, soft_terms, jd_text)
