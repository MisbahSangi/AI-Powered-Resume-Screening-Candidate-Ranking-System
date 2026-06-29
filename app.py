from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.extraction.education_classifier import level_label
from src.extraction.skill_matcher import SkillTaxonomy
from src.jd_analysis.jd_parser import parse_job_description
from src.pipeline import build_candidate_profile, score_candidate_against_job
from src.scoring.scoring_engine import ScoringWeights
from src.scoring.semantic_similarity import active_backend
from src.storage import database as db
from src.storage.vector_store import VectorRecord, VectorStore

TAXONOMY_PATH = Path(__file__).resolve().parent / "data" / "skills_taxonomy.json"

st.set_page_config(page_title="Resume Screening & Ranking", layout="wide")


@st.cache_resource
def get_taxonomy() -> SkillTaxonomy:
    return SkillTaxonomy(TAXONOMY_PATH)


@st.cache_resource
def ensure_db_ready() -> None:
    """Runs once per app session — CREATE TABLE IF NOT EXISTS is idempotent
    and safe to call on every startup.
    """
    db.init_db()


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return Path(tmp.name)


def render_score_breakdown_chart(breakdown) -> None:
    import matplotlib.pyplot as plt

    labels = ["Skill", "Semantic", "Experience", "Education"]
    values = [
        breakdown.skill_score,
        breakdown.semantic_score,
        breakdown.experience_score,
        breakdown.education_score,
    ]
    fig, ax = plt.subplots(figsize=(4, 2.2))
    bars = ax.barh(labels, values, color="#4C72B0")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Sub-score (0-1)")
    for bar, value in zip(bars, values):
        ax.text(value + 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
    fig.tight_layout()
    st.pyplot(fig)


def main():
    st.title("AI-Powered Resume Screening & Candidate Ranking")
    st.caption(
        "Hybrid pipeline: rule-based extraction + skill-taxonomy matching + "
        "semantic similarity, with a fully explainable scoring breakdown per candidate."
    )

    with st.sidebar:
        st.header("1. Job description")
        job_title = st.text_input("Job title (for your records)", placeholder="e.g. Backend Software Engineer")
        jd_input_mode = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True)
        jd_text = ""
        if jd_input_mode == "Paste text":
            jd_text = st.text_area("Paste the job description", height=220)
        else:
            jd_file = st.file_uploader("Upload JD (.txt)", type=["txt"])
            if jd_file is not None:
                jd_text = jd_file.read().decode("utf-8", errors="ignore")

        st.header("2. Resumes")
        resume_files = st.file_uploader(
            "Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True
        )

        st.header("3. Scoring weights")
        st.caption("Must sum to 1.0 — defaults reflect the rationale in the README.")
        w_skill = st.slider("Skill match", 0.0, 1.0, 0.40, 0.05)
        w_semantic = st.slider("Semantic similarity", 0.0, 1.0, 0.25, 0.05)
        w_experience = st.slider("Experience", 0.0, 1.0, 0.20, 0.05)
        w_education = st.slider("Education", 0.0, 1.0, 0.15, 0.05)
        weight_sum = w_skill + w_semantic + w_experience + w_education
        st.caption(f"Current sum: {weight_sum:.2f}")

        save_to_history = st.checkbox("Save this run to history (SQLite)", value=True)
        run_clicked = st.button("Process & rank candidates", type="primary")

    ensure_db_ready()
    taxonomy = get_taxonomy()
    st.caption(f"Active semantic-similarity backend: **{active_backend()}**")

    with st.expander("📁 Screening history (past job postings, stored in SQLite)"):
        with db.get_connection() as conn:
            past_jobs = db.get_all_jobs(conn)
        if not past_jobs:
            st.write("No screenings saved yet — process some resumes with 'Save this run to history' checked.")
        else:
            job_labels = [
                f"#{row['id']} — {row['title'] or 'Untitled'} ({row['created_at']})" for row in past_jobs
            ]
            chosen = st.selectbox("View a past screening", job_labels, key="history_job_select")
            chosen_id = past_jobs[job_labels.index(chosen)]["id"]
            with db.get_connection() as conn:
                history_rows = db.get_rankings_for_job(conn, chosen_id)
            if history_rows:
                hist_df = pd.DataFrame([dict(r) for r in history_rows])
                hist_df["final_score"] = (hist_df["final_score"] * 100).round(1)
                st.dataframe(hist_df, width="stretch", hide_index=True)
            else:
                st.write("No candidates recorded for this job yet.")

    # --- Run the pipeline only when the button is freshly clicked, but
    # persist everything needed for the rest of the page into
    # session_state so later interactions (dropdowns, sliders, other
    # buttons) don't wipe it out — see module docstring. ---
    if run_clicked:
        if not jd_text.strip():
            st.error("Please provide a job description first.")
            return
        if not resume_files:
            st.error("Please upload at least one resume.")
            return
        if abs(weight_sum - 1.0) > 1e-6:
            st.error(f"Scoring weights must sum to 1.0 (currently {weight_sum:.2f}). Adjust the sliders.")
            return

        jd = parse_job_description(jd_text, taxonomy)
        weights = ScoringWeights(skill=w_skill, semantic=w_semantic, experience=w_experience, education=w_education)

        results = []
        profiles_for_vectors = []
        with st.spinner(f"Processing {len(resume_files)} resume(s)..."):
            for uploaded in resume_files:
                path = save_uploaded_file(uploaded)
                try:
                    profile = build_candidate_profile(path, taxonomy)
                except Exception as exc:
                    st.warning(f"Could not parse {uploaded.name}: {exc}")
                    continue
                result = score_candidate_against_job(profile, jd, weights)
                results.append((uploaded.name, result))
                profiles_for_vectors.append(profile)

        if not results:
            st.error("No resumes could be processed.")
            return

        results.sort(key=lambda pair: pair[1].breakdown.final_score, reverse=True)

        job_id = None
        if save_to_history:
            with db.get_connection() as conn:
                job_id = db.insert_job(
                    conn,
                    {
                        "title": job_title or None,
                        "raw_text": jd_text,
                        "required_skills": jd.required_skills,
                        "preferred_skills": jd.preferred_skills,
                        "required_experience_years": jd.required_experience_years,
                        "required_education_level": jd.required_education_level,
                    },
                )
                for filename, result in results:
                    p, b = result.profile, result.breakdown
                    candidate_id = db.insert_candidate(
                        conn,
                        {
                            "source_path": p.source_path,
                            "name": p.name,
                            "email": p.email,
                            "phone": p.phone,
                            "linkedin": p.linkedin,
                            "github": p.github,
                            "skills": p.skills,
                            "experience_years": p.experience_years,
                            "education_level": p.education_level,
                            "raw_text": p.raw_text,
                            "summary": p.summary,
                        },
                    )
                    db.insert_score(
                        conn,
                        {
                            "candidate_id": candidate_id,
                            "job_id": job_id,
                            "final_score": b.final_score,
                            "skill_score": b.skill_score,
                            "semantic_score": b.semantic_score,
                            "experience_score": b.experience_score,
                            "education_score": b.education_score,
                            "breakdown_json": b.explanation(),
                            "recommendation": result.recommendation,
                        },
                    )

        # Persist for every subsequent rerun, and clear any stale
        # second-job results from a previous candidate set.
        st.session_state["screening"] = {
            "results": results,
            "profiles": profiles_for_vectors,
            "jd": jd,
            "weights": weights,
            "job_id": job_id,
        }
        st.session_state.pop("second_job_rows", None)

    if "screening" not in st.session_state:
        st.info("Add a job description and one or more resumes in the sidebar, then click **Process & rank candidates**.")
        return

    screening = st.session_state["screening"]
    results = screening["results"]
    profiles_for_vectors = screening["profiles"]
    jd = screening["jd"]
    weights = screening["weights"]
    job_id = screening["job_id"]

    with st.expander("Parsed job requirements", expanded=False):
        st.write("**Required skills:**", ", ".join(sorted(jd.required_skills)) or "_none detected_")
        st.write("**Preferred skills:**", ", ".join(sorted(jd.preferred_skills)) or "_none detected_")
        st.write("**Required experience:**", f"{jd.required_experience_years:.0f}+ years")
        st.write("**Required education:**", level_label(jd.required_education_level))

    if job_id is not None:
        st.caption(f"Saved to history as job #{job_id}. View it any time under 'Screening history' above.")

    # --- Ranked table ---
    st.subheader("Ranked candidates")
    table_rows = []
    for rank, (filename, result) in enumerate(results, start=1):
        b = result.breakdown
        table_rows.append(
            {
                "Rank": rank,
                "Name": result.profile.name or filename,
                "Final score": round(b.final_score * 100, 1),
                "Skill": b.skill_score,
                "Semantic": b.semantic_score,
                "Experience (yrs)": result.profile.experience_years,
                "Education": level_label(result.profile.education_level),
                "Recommendation": result.recommendation,
            }
        )
    df = pd.DataFrame(table_rows)
    st.dataframe(df, width="stretch", hide_index=True)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("Export rankings as CSV", csv_buffer.getvalue(), "candidate_rankings.csv", "text/csv")

    # --- Skill gap analysis (bonus feature) ---
    with st.expander("Skill gap analysis across all candidates (bonus feature)"):
        all_missing = {}
        for _, result in results:
            for skill in result.breakdown.missing_required_skills:
                all_missing[skill] = all_missing.get(skill, 0) + 1
        if all_missing:
            gap_df = pd.DataFrame(
                sorted(all_missing.items(), key=lambda kv: kv[1], reverse=True),
                columns=["Missing required skill", "Number of candidates missing it"],
            )
            st.dataframe(gap_df, width="stretch", hide_index=True)
        else:
            st.write("No required-skill gaps found across the candidate pool.")

    # --- Per-candidate detail ---
    st.subheader("Candidate detail")
    names = [f"{rank}. {result.profile.name or filename}" for rank, (filename, result) in enumerate(results, start=1)]
    selected = st.selectbox("Select a candidate", names, key="candidate_select")
    idx = names.index(selected)
    filename, result = results[idx]
    profile, breakdown = result.profile, result.breakdown

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"**Email:** {profile.email or '—'}")
        st.markdown(f"**Phone:** {profile.phone or '—'}")
        st.markdown(f"**LinkedIn:** {profile.linkedin or '—'}")
        st.markdown(f"**GitHub:** {profile.github or '—'}")
        st.markdown(f"**Skills:** {', '.join(profile.skills) or '—'}")
        st.markdown(f"**Summary:** {profile.summary}")
        if profile.parse_warnings:
            st.warning(" ".join(profile.parse_warnings))
    with col2:
        render_score_breakdown_chart(breakdown)

    st.markdown("**Full explanation (deterministic, generated from the scores above):**")
    st.info(breakdown.explanation())

    # --- Compare candidates ---
    st.subheader("Compare candidates")
    compare_choices = st.multiselect(
        "Pick 2 or more candidates to compare", names, default=names[: min(2, len(names))], key="compare_select"
    )
    if len(compare_choices) >= 2:
        compare_rows = []
        for choice in compare_choices:
            i = names.index(choice)
            _, r = results[i]
            compare_rows.append(
                {
                    "Name": r.profile.name,
                    "Final score": round(r.breakdown.final_score * 100, 1),
                    "Skill": r.breakdown.skill_score,
                    "Semantic": r.breakdown.semantic_score,
                    "Experience": r.breakdown.experience_score,
                    "Education": r.breakdown.education_score,
                    "Missing skills": ", ".join(r.breakdown.missing_required_skills) or "none",
                }
            )
        st.dataframe(pd.DataFrame(compare_rows), width="stretch", hide_index=True)

    # --- Vector store setup (shared by similarity search + clustering) ---
    vector_store = None
    if len(profiles_for_vectors) >= 2:
        vector_store = VectorStore()
        vector_store.build(
            [p.raw_text for p in profiles_for_vectors],
            [VectorRecord(record_id=str(i), label=p.name or f"Candidate {i}") for i, p in enumerate(profiles_for_vectors)],
        )

    # --- Vector search (bonus feature) ---
    with st.expander("Find similar candidates (bonus: vector search)"):
        if vector_store is None:
            st.write("Upload at least 2 resumes to enable similarity search.")
        else:
            query_name = st.selectbox(
                "Find candidates similar to:", [p.name for p in profiles_for_vectors], key="similarity_select"
            )
            query_profile = next(p for p in profiles_for_vectors if p.name == query_name)
            similar = vector_store.most_similar(query_profile.raw_text, top_k=len(profiles_for_vectors))
            sim_df = pd.DataFrame(
                [(rec.label, score) for rec, score in similar if rec.label != query_name],
                columns=["Candidate", "Similarity"],
            )
            st.dataframe(sim_df, width="stretch", hide_index=True)

    # --- Candidate clustering (bonus feature) ---
    with st.expander("Group candidates into clusters (bonus: candidate clustering)"):
        if vector_store is None or len(profiles_for_vectors) < 3:
            st.write("Upload at least 3 resumes to enable clustering.")
        else:
            max_k = min(6, len(profiles_for_vectors) - 1)
            n_clusters = st.slider("Number of clusters", 2, max_k, min(3, max_k), key="cluster_slider")
            clusters = vector_store.cluster(n_clusters=n_clusters)
            if not clusters:
                st.write("Not enough candidates to form that many clusters.")
            else:
                for cluster_id, members in sorted(clusters.items()):
                    st.markdown(f"**Cluster {cluster_id + 1}:** {', '.join(members)}")
                st.caption(
                    "Clusters are formed by grouping candidates with similar overall resume "
                    "content (KMeans over the same embedding space used for similarity search) — "
                    "useful for spotting natural candidate segments (e.g. backend-heavy vs. "
                    "frontend-heavy) at a glance, without reading every resume individually."
                )

    # --- Multi-job candidate matching (bonus feature) ---
    with st.expander("Match these same candidates against a different job (bonus: multi-job matching)"):
        st.caption(
            "Already-parsed candidates above are re-scored against a second job description — "
            "no re-uploading or re-parsing needed, since extraction only has to happen once per resume."
        )
        second_jd_text = st.text_area("Paste a different job description", height=160, key="second_jd")
        if st.button("Re-score against this job", key="rescore_button"):
            if not second_jd_text.strip():
                st.error("Paste a job description first.")
            else:
                jd2 = parse_job_description(second_jd_text, taxonomy)
                second_rows = []
                for p in profiles_for_vectors:
                    r2 = score_candidate_against_job(p, jd2, weights)
                    second_rows.append(
                        {
                            "Name": p.name,
                            "Final score": round(r2.breakdown.final_score * 100, 1),
                            "Skill": r2.breakdown.skill_score,
                            "Semantic": r2.breakdown.semantic_score,
                            "Missing skills": ", ".join(r2.breakdown.missing_required_skills) or "none",
                            "Recommendation": r2.recommendation,
                        }
                    )
                second_rows.sort(key=lambda row: row["Final score"], reverse=True)
                st.session_state["second_job_rows"] = second_rows

        if "second_job_rows" in st.session_state:
            st.dataframe(pd.DataFrame(st.session_state["second_job_rows"]), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
