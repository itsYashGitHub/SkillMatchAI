import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# -----------------------------
# LOAD BERT MODEL
# -----------------------------
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("resume_dataset.csv")

df.rename(columns={
    "Resume": "resume_text",
    "Category": "category"
}, inplace=True)

df = df.dropna().reset_index(drop=True)

# -----------------------------
# CREATE POSITIVE PAIRS (SAME CATEGORY)
# -----------------------------
df["job_text"] = df.groupby("category")["resume_text"].transform(
    lambda x: x.sample(frac=1, random_state=42).values
)

# -----------------------------
# CREATE NEGATIVE PAIRS (DIFFERENT CATEGORY)
# -----------------------------
df_negative = df.sample(frac=0.6, random_state=42).copy()
df_negative["job_text"] = df.groupby("category")["job_text"].transform(
    lambda x: np.roll(x.values, 1)
)


# Combine dataset
df_full = pd.concat([df, df_negative]).reset_index(drop=True)

# -----------------------------
# WEAK LABEL GENERATION
# -----------------------------
def generate_label(resume, job):
    emb = bert_model.encode([resume, job], convert_to_tensor=True)
    score = float(util.cos_sim(emb[0], emb[1]))

    if score >= 0.70:
        return 2  # Good Fit
    elif score >= 0.40:
        return 1  # Average Fit
    else:
        return 0  # Poor Fit


df_full["label"] = df_full.apply(
    lambda row: generate_label(row["resume_text"], row["job_text"]),
    axis=1
)

# -----------------------------
# CHECK CLASS DISTRIBUTION
# -----------------------------
print("\nClass distribution:")
print(df_full["label"].value_counts())

# -----------------------------
# BERT EMBEDDINGS
# -----------------------------
texts = (df_full["resume_text"] + " " + df_full["job_text"]).tolist()
X = bert_model.encode(texts, show_progress_bar=True)
y = df_full["label"]

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# TRAIN CLASSIFIER (BALANCED)
# -----------------------------
clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

clf.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(clf, "resume_fit_classifier.pkl")
print("\nModel saved as resume_fit_classifier.pkl")
