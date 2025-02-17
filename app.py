import pandas as pd
import spacy
import torch
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from flask_cors import CORS  # Importing CORS for handling cross-origin requests

app = Flask(__name__)
CORS(app)  # Enabling CORS for the entire Flask app

# Load spaCy for text preprocessing
nlp = spacy.load("en_core_web_sm")

# Load job descriptions
job_data = pd.read_csv("data/cleaned_job_descriptions.csv")

# Load Sentence Transformer model (BERT-based)
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

import re

def preprocess_text(text):
    """Optimized preprocessing: remove special characters, keep key terms"""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    return text.strip()  # Remove extra spaces


job_data["optimized_description"] = job_data["Job Description"].apply(preprocess_text)
job_embeddings = sbert_model.encode(job_data['optimized_description'].tolist(), convert_to_tensor=True)

def match_jobs(resume_text, top_n=5):
    """Find the best matching jobs for a given resume using SBERT."""
    cleaned_resume = preprocess_text(resume_text)
    
    # Compute SBERT embedding for the resume
    resume_embedding = sbert_model.encode([cleaned_resume], convert_to_tensor=True)
    
    # Compute cosine similarity with job descriptions
    similarities = util.pytorch_cos_sim(resume_embedding, job_embeddings).squeeze(0)
    
    # Get top N matching job titles
    top_indices = torch.topk(similarities, top_n).indices.tolist()
    matches = [{"title": job_data.iloc[i]["Job Title"], "similarity": float(similarities[i])} for i in top_indices]
    
    return matches

@app.route("/match", methods=["POST"])
def match():
    """API Endpoint to match resumes with jobs."""
    data = request.get_json()
    resume_text = data.get("resume", "")
    if not resume_text:
        return jsonify({"error": "Resume text is required"}), 400
    
    matches = match_jobs(resume_text)
    return jsonify({"matches": matches})

if __name__ == "__main__":
    app.run(debug=True)
