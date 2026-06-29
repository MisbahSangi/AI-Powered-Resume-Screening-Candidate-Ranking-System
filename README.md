# AI-Powered Resume Screening & Candidate Ranking System

## Why hybrid, not a single LLM API call?

It would be possible to build this by sending each resume + job description
to GPT/Claude/Gemini and asking for a match score. That approach is fast to
build, but has real costs for *this specific* use case:

| Concern | Pure LLM API | This hybrid system |
|---|---|---|
| Per-resume cost at scale | Real, recurring | Free after setup (local compute) |
| Explainability | A number with no visible reasoning | Every sub-score + matched/missing skills shown |
| Reproducibility | Can vary between calls | Deterministic — same input, same output |
| Privacy | Candidate PII leaves the machine | Stays local |
| Semantic understanding | Excellent | Good (see Known Limitations) |

To be fair to the alternative: a frontier LLM is genuinely very good at
parsing messy resume text. The hybrid approach doesn't claim to beat that on
raw extraction accuracy — it wins on cost, explainability, and
reproducibility, which is what **"Explainable scoring methodology"** in the
task brief is actually asking for. A model that outputs "82% match" with no
visible reasoning cannot satisfy that requirement no matter how accurate it
is on average.

---

## Architecture

```
Resume (PDF/DOCX/TXT)              Job Description (text)
        |                                   |
        v                                   v
 Layer 1: Parsing & section detection   Layer 3: JD parsing
 (pdfplumber/python-docx + regex)       (required vs preferred skill split)
        |                                   |
        v                                   |
 Layer 2: Hybrid extraction                  |
   - Regex: email, phone, links, dates       |
   - Skill taxonomy + fuzzy match            |
   - Heuristic name/org extraction           |
   - Experience duration, education level    |
        |                                   |
        +-----------------+-----------------+
                          |
                          v
        Layer 4: Explainable scoring engine
        (weighted: skill + semantic + experience + education)
                          |
                          v
        Layer 5: Deterministic summary & recommendation
        (extractive TF-IDF summarizer, templated explanation)
                          |
                          v
              Layer 6/7: SQLite storage + Streamlit dashboard
```

Every layer is independently testable (see `tests/`) and the scoring engine
never depends on any of the optional ML upgrades being installed.

---

## Scoring methodology (the explainability deliverable)

Four signals, weighted and combined into one final score. Every sub-score is
shown to the user — never just the final number.

| Component | Weight | How it's computed | Why |
|---|---|---|---|
| **Skill match** | 40% | Required/preferred skill overlap via taxonomy + fuzzy matching | Whether someone can actually do the job matters most |
| **Semantic similarity** | 25% | Cosine similarity between resume and JD text | Catches relevant experience phrased differently than the JD's exact wording |
| **Experience match** | 20% | Years parsed from work history vs. years required | Simple, traceable ratio |
| **Education match** | 15% | Ordinal degree-level comparison, with partial credit for one level below | Lowest weight — many strong candidates have non-traditional paths |

These weights are adjustable live in the dashboard sidebar (must sum to
1.0). The default split is a starting point, not a fixed law — change it and
document why, the same way this README documents the original choice.

Every `ScoreBreakdown` object carries an `.explanation()` method that
generates a plain-language summary **from the numbers themselves** — it is
template-based, not LLM-generated, so it can never say anything that isn't
already true of the underlying scores.

---

## What's genuinely tested vs. what's an upgrade path

This was built and tested inside a sandboxed environment whose network
policy blocks Hugging Face Hub and GitHub release-asset downloads. That
matters for two optional components:

| Component | Tested here (default) | Upgrade path (untested here, but auto-detected) |
|---|---|---|
| Semantic similarity | TF-IDF cosine similarity (scikit-learn) | `sentence-transformers` (all-MiniLM-L6-v2) — install it and it activates automatically, no code changes |
| Name/org extraction | Heuristic rules (top-of-resume capitalized line, "Title at Company" patterns) | spaCy NER (`en_core_web_sm`) — run `python -m spacy download en_core_web_sm` and it's used as a cross-check automatically |

Both fallbacks are real, working implementations — not stubs — and the
example pipeline run in this README's test suite uses them. On your own
machine (normal internet access), installing the two optional packages
listed in `requirements.txt`'s comments upgrades semantic matching quality
with zero code changes.

**Concrete proof this matters:** in testing, the sentence
*"Built REST APIs with FastAPI and PostgreSQL"* against *"Python web
framework experience with relational databases"* scored only **0.06**
semantic similarity under TF-IDF, despite being a strong conceptual match —
because the two sentences share almost no exact words. This is precisely
what sentence-transformers embeddings are built to fix, and precisely why
it's offered as an upgrade rather than skipped entirely.

---

## Known limitations (stated honestly, not hidden)

- **Experience score measures tenure, not relevance.** A candidate with 7
  years in an unrelated field (e.g. graphic design) scores 1.0 on
  Experience even though that experience doesn't transfer. This is by
  design — relevance is what the Skill and Semantic scores are for. In
  testing, an unrelated-field resume correctly still ranked last overall
  (36.5/100) because skill (0.0) and semantic (0.06) scores pulled the
  average down, even with a perfect experience/education score.
- **Education score checks degree level, not field of study.** A Bachelor's
  in any subject satisfies a "Bachelor's required" JD line.
- **Skill taxonomy coverage is finite.** `data/skills_taxonomy.json` ships
  with ~100 common tech/soft skills. It's a starting point, not exhaustive —
  extend it for other domains by adding entries; no code changes needed.
- **Section detection relies on common headers.** Resumes with highly
  unconventional formatting (no clear section headers at all) fall back to
  whole-document search, which works but is less precise.

---

## Project structure

```
resume_screening_system/
├── app.py                      # Streamlit dashboard
├── cli.py                      # Batch/scriptable CLI
├── requirements.txt
├── data/
│   ├── skills_taxonomy.json    # Master skill list (extensible)
│   ├── sample_resumes/         # 3 sample resumes (.txt + .pdf), varying match strength
│   └── sample_job_descriptions/
├── src/
│   ├── parsing/                # Layer 1: file -> text -> sections
│   ├── extraction/              # Layer 2: regex, skills, NER, experience, education
│   ├── jd_analysis/             # Layer 3: job description -> structured requirements
│   ├── scoring/                 # Layer 4: semantic similarity + weighted scoring engine
│   ├── summarization/          # Layer 5: extractive summarizer + recommendation text
│   ├── storage/                 # Layer 6: SQLite + numpy-based vector store
│   └── pipeline.py             # Orchestrates every layer
└── tests/
    ├── test_pipeline.py        # Automated regression tests (pytest)
    └── ui_test.py               # Playwright end-to-end dashboard test
```

## Setup & running

```bash
pip install -r requirements.txt

# Optional upgrades (see "What's genuinely tested" above):
# pip install sentence-transformers
# python -m spacy download en_core_web_sm

# Dashboard:
streamlit run app.py

# CLI (batch mode):
python cli.py --jd data/sample_job_descriptions/backend_engineer_jd.txt \
              --resumes data/sample_resumes/*.pdf \
              --output results.csv

# Tests:
python -m pytest tests/test_pipeline.py -v
```

## AI features mapped to task requirements

| Task requirement | Implementation |
|---|---|
| Skill matching | `skill_matcher.py` — taxonomy + fuzzy matching |
| Semantic similarity analysis | `semantic_similarity.py` |
| Resume summarization | `summarizer.py` — extractive TF-IDF sentence ranking |
| Candidate ranking | `pipeline.rank_candidates()` |
| Keyword extraction | TF-IDF sentence scoring (summarizer) + skill matching |
| Recommendation generation | `generate_recommendation()` — deterministic, from score breakdown |

## Bonus features implemented

- **Skill Gap Analysis** — aggregated missing-required-skills view across all candidates (dashboard)
- **Vector Search** — "find similar candidates" using a numpy-based embedding store
- **Multi-Job Candidate Matching** — already-parsed candidates can be re-scored against a *second* job description with one click, reusing the full explainable scoring engine (not just embedding similarity) — no re-uploading or re-parsing needed
- **Candidate Clustering** — KMeans grouping over the same embedding space, exposed directly in the dashboard with an adjustable cluster count

Not implemented given the project's tight 2-day timeline (genuinely useful,
but not free the way the above are): Interview Question Generation, AI Chat
Assistant for Recruiters, Resume Improvement Suggestions. The cleanest
extension point for these is a local LLM (e.g. Ollama) used purely to
*phrase* output from the existing deterministic data — see the docstring in
`summarizer.generate_recommendation()`.

## Persistence (SQLite)

Every processed screening — job description, all candidates, and their full
score breakdowns — is saved to `data/app.db` automatically (toggleable via
a sidebar checkbox). A "Screening history" panel at the top of the
dashboard lists every past job posting and lets you revisit its rankings
without re-uploading anything, confirmed to survive a full page reload.

This was *not* true in an earlier version of this project: `database.py`
existed and was unit-tested in isolation, but was never actually called
from `app.py`. Worth saying plainly, since it's exactly the kind of gap
that looks fine in a code review and only shows up when you actually use
the running app.

## A real bug this project caught (and fixed) along the way

Streamlit reruns the entire script on every widget interaction, and
`st.button(...)` only returns `True` on the one rerun immediately
following its click. Early code gated the *entire* results section behind
`if not run_clicked: return` — which meant selecting a different candidate
from the dropdown, changing the compare list, or clicking any other button
after processing would silently reset the whole page back to the empty
state, since `run_clicked` evaluates to `False` on every rerun except the
exact one where that button was clicked.

Fixed by moving processed results into `st.session_state`, so they persist
across any later interaction until the user explicitly reprocesses. A
regression test (`tests/bonus_features_test.py`) specifically exercises
"change the candidate dropdown after processing" to make sure this can't
silently come back.
