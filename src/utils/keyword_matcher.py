import re
from typing import Dict, Iterable, List, Set, Tuple

# Flexible separator between tokens inside a term (space, -, _, /, .)
SEP = r'[\s\-/_.]*'

# Extend as needed. Keys must be lowercase canonical forms.
SYNONYMS: Dict[str, List[str]] = {
    # cloud
    'gcp': ['google cloud', 'google cloud platform'],
    'aws': ['amazon web services'],
    'azure devops': ['azure boards', 'ado'],
    # web/dev
    'node.js': ['nodejs', 'node js'],
    '.net': ['dotnet', 'dot net'],
    # bi/analytics
    'power bi': ['powerbi', 'pbi'],
    'ms office': ['microsoft office', 'office 365', 'office365'],
    'postgresql': ['postgres'],
    # data engineering
    'ci/cd': ['cicd', 'ci cd'],
    'google bigquery': ['bigquery', 'gcp bigquery'],
    'etl': ['extract transform load'],
    # gis/geo
    'pix4d': ['pix 4d', 'pix-4d'],
    'arcgis': ['arc gis', 'arc-gis'],
    'qgis': ['q gis', 'q-gis'],
    # security
    'iam': ['identity and access management'],
}

def canonical(term: str) -> str:
    return (term or "").strip().lower()

def _token_sep_variant(t: str) -> str:
    # Convert "power bi" -> r'power[\s\-/_.]*bi'
    tokens = re.split(r'[\s\-/_.]+', t)
    return SEP.join(map(re.escape, tokens))

def term_variants(term: str) -> List[str]:
    """
    Return canonical term + known synonyms/aliases + flexible-separator variants
    """
    t = canonical(term)
    variants = [t]
    variants.extend(SYNONYMS.get(t, []))
    # Flexible variant for multi-token terms
    for base in list(variants):
        if re.search(r'[\s\-/_.]', base):
            variants.append(_token_sep_variant(base))
            variants.append(r'\s+'.join(map(re.escape, re.split(r'[\s\-/_.]+', base))))
    # Dedup preserving order
    seen = set()
    out = []
    for v in variants:
        v2 = v.lower()
        if v2 not in seen:
            seen.add(v2)
            out.append(v)
    return out

def compile_patterns_for_term(term: str) -> List[re.Pattern]:
    """
    Build patterns for term and its variants.
    (?<!\w) and (?!\w) let punctuation tokens like 'c++' match.
    """
    patterns: List[re.Pattern] = []
    for v in term_variants(term):
        # If already looks like a pattern (contains backslashes) leave as-is
        if '\\' in v and ('\\s' in v or '\\-' in v or '\\.' in v):
            pat = r'(?i)(?<!\w)' + v + r'(?!\w)'
        else:
            pat = r'(?i)(?<!\w)' + re.escape(v) + r'(?!\w)'
        patterns.append(re.compile(pat))
    return patterns

def any_match_with_surface(text: str, patterns: Iterable[re.Pattern]) -> str:
    """
    Return the first matched surface string or "" if none.
    (?<!\\w) and (?!\\w) let punctuation tokens like 'c++' match.
    """
    for p in patterns:
        m = p.search(text)
        if m:
            return m.group(0)
    return ""

def present_missing_with_surface(text: str, terms: Iterable[str], synonyms: Dict = None) -> Tuple[Set[str], Set[str], Dict[str, str]]:
    """
    Return (present set, missing set, surfaces dict term->matched surface).
    """
    text = text or ""
    present: Set[str] = set()
    missing: Set[str] = set()
    surfaces: Dict[str, str] = {}
    for raw in terms or []:
        t = canonical(raw)
        if not t:
            continue
        patterns = compile_patterns_for_term(t)
        surface = any_match_with_surface(text, patterns)
        if surface:
            present.add(t)
            surfaces[t] = surface
        else:
            missing.add(t)
    return present, missing, surfaces

# Backward-compatible wrapper if needed elsewhere
def present_missing(text: str, terms: Iterable[str]) -> Tuple[Set[str], Set[str]]:
    p, m, _ = present_missing_with_surface(text, terms)
    return p, m
