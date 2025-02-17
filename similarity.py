import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_embeddings(texts):
    """
    Compute SBERT embeddings for a list of text inputs.
    """
    return model.encode(texts, convert_to_numpy=True)

def compute_similarity(resume_file, job_file, output_file):
    """
    Compute similarity between resumes and job descriptions.
    """
    # Load datasets
    resumes_df = pd.read_csv(resume_file)
    job_df = pd.read_csv(job_file)

    # Extract relevant columns
    resume_texts = resumes_df['Cleaned_Resume'].tolist()  # Preprocessed resumes
    job_texts = job_df['cleaned_description'].tolist()  # Preprocessed job descriptions
    job_titles = job_df['Job Title'].tolist()  # Job titles for reference
    resume_categories = resumes_df['Category'].tolist()  # Resume categories for reference

    # Compute embeddings
    resume_embeddings = compute_embeddings(resume_texts)
    job_embeddings = compute_embeddings(job_texts)

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(resume_embeddings, job_embeddings)

    # Convert similarity scores to a DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, columns=job_titles)

    # Add original categories for reference
    similarity_df.insert(0, "Resume Category", resume_categories)

    # Save results
    similarity_df.to_csv(output_file, index=False)
    print(f"Similarity scores saved to {output_file}")

# Example usage
if __name__ == "__main__":
    compute_similarity("data/Cleaned_ResumeDataSet.csv", "data/cleaned_job_descriptions.csv", "data/similarity_scores.csv")
