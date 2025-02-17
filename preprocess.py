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

# Load the dataset
df = pd.read_csv("data/UpdatedResumeDataSet.csv", encoding='utf-8')
# Apply preprocessing to the 'Resume' column
df['Cleaned_Resume'] = df['Resume'].apply(preprocess_text)
# Save the cleaned data
df.to_csv("data/Cleaned_ResumeDataSet.csv", index=False)
print("Preprocessing complete. Cleaned data saved to 'data/Cleaned_ResumeDataSet.csv'.")
