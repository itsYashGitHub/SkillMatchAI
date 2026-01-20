import streamlit as st
import fitz  # PyMuPDF for PDF reading
from matcher import predict_fit, explain_match, similarity_score, skill_coverage, final_fit_label

st.set_page_config(page_title="SkillMatch AI", layout="wide")

st.title("🤖 SkillMatch AI")
st.subheader("AI-powered Resume ↔ Job Matching System")

# -----------------------------
# PDF READER
# -----------------------------
def read_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# -----------------------------
# INPUTS
# -----------------------------
with st.sidebar:
    st.header("📄 Upload Resume")
    resume_file = st.file_uploader("Upload PDF Resume", type=["pdf"])

    st.header("📝 Job Description")
    job_description = st.text_area("Paste Job Description", height=220)

# -----------------------------
# PROCESS
# -----------------------------
if resume_file and job_description:
    resume_text = read_pdf(resume_file)

    if st.button("🚀 Analyze Match"):
        with st.spinner("Analyzing resume..."):
            label_raw, probs = predict_fit(resume_text, job_description)
            explanation = explain_match(resume_text, job_description)
            sim_score = similarity_score(resume_text, job_description)

            matched_skills = explanation["matched_skills"]
            job_skills = set(matched_skills).union(set(explanation["missing_skills"]))
            coverage = skill_coverage(matched_skills, job_skills)

            label_final = final_fit_label(label_raw, matched_skills, job_skills)

        # -----------------------------
        # MATCH RESULT
        # -----------------------------
        st.markdown("## 📊 Match Result")
        st.markdown(f"### {label_final}")
        st.progress(int(sim_score * 100))
        st.caption(f"Semantic Similarity Score: {sim_score:.2f}")
        st.metric("Skill Coverage", f"{coverage * 100:.1f}%")

        # -----------------------------
        # CONFIDENCE (STATIC)
        # -----------------------------
        st.markdown("### 🔍 Prediction Confidence")

        st.write("❌ Poor Fit")
        st.progress(int(probs[0] * 100))

        st.write("⚠️ Average Fit")
        st.progress(int(probs[1] * 100))

        st.write("✅ Good Fit")
        st.progress(int(probs[2] * 100))

        # -----------------------------
        # EXPLAINABILITY
        # -----------------------------
        st.markdown("## 🧠 Why this result?")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ Matching Skills")
            if matched_skills:
                for skill in matched_skills:
                    st.success(skill.title())
            else:
                st.info("No strong overlaps detected.")

        with col2:
            st.markdown("### ❌ Missing Skills")
            missing_skills = explanation["missing_skills"]
            if missing_skills:
                for skill in missing_skills:
                    st.warning(skill.title())
            else:
                st.info("No major gaps detected.")

        # -----------------------------
        # DETAILS
        # -----------------------------
        with st.expander("📌 How does this work?"):
            st.write("""
            • Resume and job description are embedded using BERT  
            • A trained ML classifier predicts fit category  
            • Skill overlap provides explainability  
            • Final fit combines semantic prediction and explicit skill coverage  
            """)

elif not resume_file and not job_description:
    st.info("Please upload a resume and enter a job description to get started.")
elif not resume_file:
    st.info("Please upload a resume PDF file.")
elif not job_description:
    st.info("Please enter the job description text.")
