SkillMatch AI (Link: https://news-app-wg3rpx6bvnt6ylpgjcsqee.streamlit.app/)

Overview

SkillMatch AI is an AI-powered resume-to-job matching system designed to help recruiters and job seekers quickly assess how well a resume fits a specific job description.

This project combines:

State-of-the-art NLP with BERT embeddings for semantic understanding

A trained machine learning classifier for multi-class fit prediction (Poor / Average / Good)

Explainability features that highlight matched and missing skills between the resume and job description

A user-friendly Streamlit app with PDF resume upload and clear visual feedback

Features

PDF Resume Upload: Extract text directly from resume PDFs for easy input

Job Description Input: Paste or type job descriptions

BERT-based Semantic Similarity: Deep language model embeddings to capture contextual skill matches

ML Classifier: Predicts resume fit into 3 categories (Poor, Average, Good)

Explainability: Displays matched and missing skills using curated dictionary-based phrase matching

Interactive UI: Streamlit interface with progress bars and skill badges for clarity

Technologies Used

Python 3.13

Streamlit
 for UI

Sentence-Transformers
 (BERT embeddings)

scikit-learn
 (ML classifier)

PyMuPDF
 for PDF parsing

spaCy for NLP preprocessing (limited role)

Installation & Setup

Clone the repo:

git clone https://github.com/yourusername/SkillMatchAI.git
cd SkillMatchAI


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py

How It Works

Upload your resume in PDF format

Paste the job description text

The system extracts text from the PDF

Resume and job description are embedded using BERT

A trained classifier predicts the fit category

Skill overlaps and gaps are extracted using a curated skill dictionary

Results are displayed with semantic similarity score, fit category, confidence, and matched/missing skills

Model Training (Summary)

Trained a multi-class classifier on a curated resume dataset

Used weak supervision with cross-category negative sampling

Employed Sentence-BERT embeddings as input features

Evaluated model performance using precision, recall, and F1-score
