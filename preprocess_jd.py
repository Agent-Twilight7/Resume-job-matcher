import pandas as pd
import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """
    Preprocess the input text by tokenizing, lemmatizing, and removing stopwords.
    """
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def preprocess_job_descriptions(input_file, output_file):
    """
    Preprocess job descriptions from a CSV file and save the cleaned data to output_file.
    """
    df = pd.read_csv(input_file)
    df['cleaned_description'] = df['Job Description'].apply(preprocess_text)
    df.to_csv(output_file, index=False)
    print(f"Preprocessed job descriptions saved to {output_file}")

# Example usage
if __name__ == "__main__":
    preprocess_job_descriptions("data/job_title_des.csv", "data/cleaned_job_descriptions.csv")
