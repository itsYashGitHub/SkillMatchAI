import streamlit as st
import fitz
from matcher import predict_fit, explain_match, similarity_score

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
            label, probs = predict_fit(resume_text, job_description)
            explanation = explain_match(resume_text, job_description)
            sim_score = similarity_score(resume_text, job_description)

        # -----------------------------
        # MATCH RESULT
        # -----------------------------
        st.markdown("## 📊 Match Result")
        st.markdown(f"### {label}")
        st.progress(int(sim_score * 100))
        st.caption(f"Semantic Similarity Score: {sim_score:.2f}")

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
            if explanation["matched_skills"]:
                for skill in explanation["matched_skills"]:
                    st.success(skill.title())
            else:
                st.info("No strong overlaps detected.")

        with col2:
            st.markdown("### ❌ Missing Skills")
            if explanation["missing_skills"]:
                for skill in explanation["missing_skills"]:
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
            """)
