import streamlit as st
from matcher import calculate_match

st.title("SkillMatch AI - Resume & Job Description Matcher")

st.write("Paste your resume text and the job description below to get a match score.")

resume_text = st.text_area("Resume Text", height=150)
job_text = st.text_area("Job Description", height=150)

if st.button("Calculate Match"):
    if not resume_text or not job_text:
        st.warning("Please enter both resume and job description text.")
    else:
        score = calculate_match(resume_text, job_text)
        st.success(f"Match Score: {score}%")
