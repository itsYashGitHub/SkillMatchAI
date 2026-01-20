import joblib
import re
from sentence_transformers import SentenceTransformer, util

# Load models
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
clf = joblib.load("resume_fit_classifier.pkl")

# -----------------------------
# SKILL DICTIONARY (EXTENDABLE)
# -----------------------------
SKILLS = [
    "java", "python", "sql", "kafka", "angular", "spring", "spring boot",
    "machine learning", "deep learning", "data science",
    "react", "node", "docker", "kubernetes",
    "aws", "azure", "gcp",
    "html", "css", "javascript",
    "mysql", "postgresql", "mongodb",
    "git", "linux", "rest api"
]

# -----------------------------
# SKILL EXTRACTION (ROBUST)
# -----------------------------
def extract_skills(text):
    text = text.lower()
    found_skills = set()

    for skill in SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text):
            found_skills.add(skill)

    return found_skills


# -----------------------------
# PREDICT FIT
# -----------------------------
def predict_fit(resume_text, job_text):
    combined_text = resume_text + " " + job_text
    embedding = bert_model.encode([combined_text])

    prediction = clf.predict(embedding)[0]
    probabilities = clf.predict_proba(embedding)[0]

    label_map = {
        0: "❌ Poor Fit",
        1: "⚠️ Average Fit",
        2: "✅ Good Fit"
    }

    return label_map[prediction], probabilities


# -----------------------------
# EXPLAINABILITY (FIXED)
# -----------------------------
def explain_match(resume_text, job_text):
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_text)

    matched = sorted(resume_skills.intersection(job_skills))
    missing = sorted(job_skills.difference(resume_skills))

    return {
        "matched_skills": matched,
        "missing_skills": missing
    }


# -----------------------------
# SIMILARITY SCORE
# -----------------------------
def similarity_score(resume_text, job_text):
    emb = bert_model.encode([resume_text, job_text], convert_to_tensor=True)
    return float(util.cos_sim(emb[0], emb[1]))
