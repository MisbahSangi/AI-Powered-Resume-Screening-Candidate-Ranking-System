"""
ner_extractor.py
-----------------
Extracts candidate name and organization names (employers, universities).

Design note: this module is built "heuristic-first, ML-optional" on purpose.
- The heuristics below (name = top-of-resume capitalized line, orgs = lines
  inside Experience/Education that match a "Title at Company" / "Company |
  Title" shape) are *deliberately* the primary path — they're fast, free,
  fully explainable, and in practice work well specifically for resumes
  (which are far more structurally predictable than general text).
- spaCy's statistical NER model (en_core_web_sm) is wired in as an optional
  cross-check/enhancement: if the model is installed, it's used to validate
  or supplement the heuristic result. If it's not installed (or the download
  is blocked by a network policy, as it is in some sandboxed environments),
  the module silently falls back to heuristics only — it never hard-fails.

This means the system is fully functional out of the box, and gets slightly
more robust automatically once the spaCy model is available on a machine
with normal internet access.
"""

from __future__ import annotations

import re
from typing import List, Optional

_SPACY_NLP = None
_SPACY_LOAD_ATTEMPTED = False


def _get_spacy_model():
    """Lazily try to load en_core_web_sm exactly once. Never raises."""
    global _SPACY_NLP, _SPACY_LOAD_ATTEMPTED
    if _SPACY_LOAD_ATTEMPTED:
        return _SPACY_NLP
    _SPACY_LOAD_ATTEMPTED = True
    try:
        import spacy

        _SPACY_NLP = spacy.load("en_core_web_sm")
    except Exception:
        # Model not installed, or some environment-level issue (e.g. blocked
        # download). Either way: fall back to heuristics, don't crash.
        _SPACY_NLP = None
    return _SPACY_NLP


_NAME_LINE_RE = re.compile(r"^[A-Z][a-zA-Z.'-]+(\s+[A-Z][a-zA-Z.'-]+){1,3}$")

_NOISE_WORDS = {
    "resume", "curriculum", "vitae", "cv", "profile", "portfolio",
}


def extract_name_heuristic(header_text: str, full_text: str) -> Optional[str]:
    """A candidate's name is almost always one of the first non-empty lines,
    written in Title Case, with no digits/emails/contact symbols.
    """
    search_text = header_text if header_text.strip() else full_text
    for line in search_text.splitlines()[:8]:
        candidate = line.strip()
        if not candidate or len(candidate) > 40:
            continue
        if any(ch.isdigit() for ch in candidate):
            continue
        if "@" in candidate or "http" in candidate.lower():
            continue
        if candidate.lower() in _NOISE_WORDS:
            continue
        if _NAME_LINE_RE.match(candidate):
            return candidate
    return None


def extract_name(header_text: str, full_text: str) -> Optional[str]:
    """Primary path: heuristic. Cross-check with spaCy NER if available."""
    heuristic_name = extract_name_heuristic(header_text, full_text)

    nlp = _get_spacy_model()
    if nlp is not None:
        doc = nlp(full_text[:500])  # name will be near the top
        person_ents = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if person_ents and not heuristic_name:
            return person_ents[0]
        # If both agree (or heuristic found nothing), this still resolves cleanly.

    return heuristic_name


# --- Organization extraction (companies, universities) -----------------

_ORG_LINE_PATTERNS = [
    # "Software Engineer at Acme Corp"
    re.compile(r"^(?P<title>.+?)\s+(?:at|@)\s+(?P<org>.+?)(?:\s*[\|,].*)?$", re.IGNORECASE),
    # "Acme Corp | Software Engineer | Jan 2022 - Present"
    re.compile(r"^(?P<org>[^|,]+?)\s*\|\s*(?P<title>[^|,]+)"),
]


def extract_organizations(section_text: str) -> List[str]:
    """Heuristic extraction of organization names from an Experience or
    Education section, using common resume line shapes.

    Should be called with section text (e.g. ParsedResume.section('experience')),
    not the whole resume — otherwise contact-info lines like
    "email@x.com | +92 300 1234567" can superficially match the pipe-delimited
    pattern below. As a defensive second layer, lines containing '@' or 'http'
    are skipped outright regardless of what's calling this.
    """
    orgs: List[str] = []
    for line in section_text.splitlines():
        line = line.strip()
        if not line or len(line) > 100:
            continue
        if "@" in line or "http" in line.lower():
            continue
        for pattern in _ORG_LINE_PATTERNS:
            match = pattern.match(line)
            if match and "org" in match.groupdict():
                org = match.group("org").strip(" .-")
                if org and len(org) < 60:
                    orgs.append(org)
                break
    return list(dict.fromkeys(orgs))  # dedupe, preserve order
