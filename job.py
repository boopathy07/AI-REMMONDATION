import streamlit as st
import pandas as pd
import spacy
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.matcher import PhraseMatcher

st.title("AI Job Recommendation System")
st.write("Upload your resume to get job recommendations")

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

domain_keywords = {

    "AI": [
        "machine learning",
        "deep learning",
        "nlp",
        "artificial intelligence",
        "computer vision"
    ],

    "Data": [
        "data analysis",
        "pandas",
        "numpy",
        "data scientist",
        "statistics"
    ],

    "Web": [
        "react",
        "javascript",
        "html",
        "css",
        "frontend",
        "backend"
    ]
}

skill_list = [
    "python","java","sql","machine learning","deep learning",
    "nlp","pandas","numpy","tensorflow","pytorch","data analysis",
    "react","javascript","html","css","excel","power bi","tableau"
]

skill_graph = {
    "machine learning": ["python","deep learning","data analysis"],
    "deep learning": ["machine learning","tensorflow","pytorch"],
    "nlp": ["machine learning","deep learning"],
    "react": ["javascript","html","css"],
    "javascript": ["react","html","css"],
    "sql": ["database","data analysis"],
    "python": ["machine learning","pandas","data analysis"],
    "pandas": ["python","data analysis"]
}

matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp.make_doc(skill) for skill in skill_list]
matcher.add("SKILLS", patterns)

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + " "
    return text

def preprocess(text):
    doc = nlp(str(text).lower())
    tokens = [t.lemma_ for t in doc if not t.is_stop and t.is_alpha]
    return " ".join(tokens)

def extract_skills(text):
    doc = nlp(text.lower())
    matches = matcher(doc)
    skills = set()
    for match_id, start, end in matches:
        skills.add(doc[start:end].text)
    return list(skills)

def expand_skills(skills):
    expanded = set(skills)
    for s in skills:
        if s in skill_graph:
            expanded.update(skill_graph[s])
    return list(expanded)

def detect_domain(text):

    text = text.lower()

    scores = {}

    for domain, keywords in domain_keywords.items():

        score = 0

        for word in keywords:

            if word in text:
                score += 1

        scores[domain] = score

    return max(scores, key=scores.get)

def extract_experience(text):
    patterns = [
        r'(\d+)\+?\s+years?\s+of\s+experience',
        r'(\d+)\+?\s+years?\s+experience'
    ]
    for p in patterns:
        m = re.search(p, text.lower())
        if m:
            return int(m.group(1))
    return 0

def experience_score(user_exp, job_exp):
    diff = abs(user_exp - job_exp)
    return max(0, 1 - diff/10)

def preference_score(user_domain, job_domain):
    if str(user_domain).lower() == str(job_domain).lower():
        return 1
    return 0

def skill_overlap(user_skills, job_skills):
    user_set = set(user_skills)
    job_set = set(job_skills)
    if len(job_set) == 0:
        return 0
    common = user_set.intersection(job_set)
    return len(common) / len(job_set)

resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])
jobs_file = st.file_uploader("Upload Jobs Dataset CSV", type=["csv"]) 

if resume_file and jobs_file:

    jobs = pd.read_csv(jobs_file)

    jobs["clean_description"] = jobs["description"].apply(preprocess)
    jobs["job_skills"] = jobs["description"].apply(extract_skills)

    job_embeddings = model.encode(jobs["clean_description"].tolist())

    resume_text = extract_text_from_pdf(resume_file)

    resume_clean = preprocess(resume_text)

    resume_skills = extract_skills(resume_text)
    resume_skills = expand_skills(resume_skills)

    user_experience = extract_experience(resume_text)
    user_domain = detect_domain(resume_text)

    resume_embedding = model.encode(resume_clean)

    similarities = cosine_similarity([resume_embedding], job_embeddings)[0]
    jobs["similarity"] = similarities

    candidate_jobs = jobs.sort_values(by="similarity", ascending=False).head(50)

    scores = []

    for _, row in candidate_jobs.iterrows():

        S = row["similarity"]
        E = experience_score(user_experience, row["experience"])
        P = preference_score(user_domain, row["domain"])
        skill_score = skill_overlap(resume_skills, row["job_skills"])

        score = 0.5*S + 0.2*E + 0.2*skill_score + 0.1*P
        scores.append(score)

    candidate_jobs["final_score"] = scores

    recommendations = candidate_jobs.sort_values(
        by="final_score",
        ascending=False
    ).drop_duplicates(subset=["title"]).head(10)

    st.write("Detected user domain:", user_domain)
    st.write("Detected Skills:", resume_skills)
    st.write("Detected Experience:", user_experience, "years\n")

    for _, row in recommendations.iterrows():

        st.write("Job:", row["title"])
        st.write("Domain:", row["domain"])
        st.write("Similarity:", round(row["similarity"],3))
        st.write("Score:", round(row["final_score"],3))
        st.write("Required Skills:", row["skills"])
        st.write("-----")