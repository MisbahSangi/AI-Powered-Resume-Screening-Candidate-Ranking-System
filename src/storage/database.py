from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    name TEXT,
    email TEXT,
    phone TEXT,
    linkedin TEXT,
    github TEXT,
    skills_json TEXT,
    experience_years REAL,
    education_level INTEGER,
    raw_text TEXT,
    summary TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    raw_text TEXT NOT NULL,
    required_skills_json TEXT,
    preferred_skills_json TEXT,
    required_experience_years REAL,
    required_education_level INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id INTEGER NOT NULL REFERENCES candidates(id),
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    final_score REAL,
    skill_score REAL,
    semantic_score REAL,
    experience_score REAL,
    education_score REAL,
    breakdown_json TEXT,
    recommendation TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def get_connection(db_path: Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def insert_candidate(conn: sqlite3.Connection, candidate: dict) -> int:
    cur = conn.execute(
        """INSERT INTO candidates
           (source_path, name, email, phone, linkedin, github, skills_json,
            experience_years, education_level, raw_text, summary)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            candidate.get("source_path"),
            candidate.get("name"),
            candidate.get("email"),
            candidate.get("phone"),
            candidate.get("linkedin"),
            candidate.get("github"),
            json.dumps(sorted(candidate.get("skills", []))),
            candidate.get("experience_years"),
            candidate.get("education_level"),
            candidate.get("raw_text"),
            candidate.get("summary"),
        ),
    )
    return cur.lastrowid


def insert_job(conn: sqlite3.Connection, job: dict) -> int:
    cur = conn.execute(
        """INSERT INTO jobs
           (title, raw_text, required_skills_json, preferred_skills_json,
            required_experience_years, required_education_level)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            job.get("title"),
            job.get("raw_text"),
            json.dumps(sorted(job.get("required_skills", []))),
            json.dumps(sorted(job.get("preferred_skills", []))),
            job.get("required_experience_years"),
            job.get("required_education_level"),
        ),
    )
    return cur.lastrowid


def insert_score(conn: sqlite3.Connection, score: dict) -> int:
    cur = conn.execute(
        """INSERT INTO scores
           (candidate_id, job_id, final_score, skill_score, semantic_score,
            experience_score, education_score, breakdown_json, recommendation)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            score["candidate_id"],
            score["job_id"],
            score["final_score"],
            score["skill_score"],
            score["semantic_score"],
            score["experience_score"],
            score["education_score"],
            score.get("breakdown_json"),
            score.get("recommendation"),
        ),
    )
    return cur.lastrowid


def get_rankings_for_job(conn: sqlite3.Connection, job_id: int) -> List[sqlite3.Row]:
    return conn.execute(
        """SELECT c.id as candidate_id, c.name, c.email, s.final_score,
                  s.skill_score, s.semantic_score, s.experience_score,
                  s.education_score, s.recommendation
           FROM scores s JOIN candidates c ON c.id = s.candidate_id
           WHERE s.job_id = ?
           ORDER BY s.final_score DESC""",
        (job_id,),
    ).fetchall()


def get_all_jobs(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """List every job posting ever screened, most recent first — powers the
    'past job postings' history view so a recruiter can revisit a previous
    screening run without re-uploading anything.
    """
    return conn.execute(
        """SELECT id, title, required_experience_years, required_education_level, created_at
           FROM jobs ORDER BY created_at DESC"""
    ).fetchall()
