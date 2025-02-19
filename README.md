# Job-Resume Matcher

A web application that matches a resume to job descriptions using natural language processing (NLP) techniques and machine learning models. It analyzes resumes and compares them to a dataset of job descriptions to find the most relevant job titles based on similarity scores.

## Tech Stack

- **Frontend**: React.js, HTML, CSS  
- **Backend**: Flask  
- **NLP Model**: Sentence-BERT (SBERT) with Hugging Face  
- **Text Processing**: SpaCy, Regular Expressions  
- **Machine Learning Framework**: PyTorch  
- **API Communication**: JSON  

## Project Overview

This project leverages advanced natural language processing (NLP) techniques to compare a resume to a set of job descriptions. By utilizing pre-trained models like Sentence-BERT, the project calculates similarity scores and suggests the best job matches based on the user's resume.

## Key Features

- **Resume Matching Algorithm**: Uses Sentence-BERT for semantic search to match resumes to relevant job descriptions based on similarity.  
- **Preprocessing**: Optimized text preprocessing (removing special characters and stopwords) for better resume-job comparison.  
- **Real-Time Results**: Users can submit resumes and instantly view job recommendations with matching titles and similarity scores.  
- **User-Friendly Interface**: Simple and professional design allowing seamless interaction with the backend API.  

## Setup & Installation

### Clone the repository:
```bash
git clone https://github.com/your-username/job-resume-matcher.git
cd job-resume-matcher
```

### Install required dependencies:

#### Frontend (React.js):
```bash
cd frontend
npm install
```

#### Backend (Flask, PyTorch, Sentence-BERT):
```bash
cd backend
pip install -r requirements.txt
```

### Run the backend server:
```bash
python app.py
```

### Run the frontend React app:
```bash
npm start
```

### Access the app:
Open your browser and go to [http://localhost:3000](http://localhost:3000).

## How It Works

### User Interaction:
```
- The user pastes their resume into the provided input box on the frontend.
```

### Backend Processing:
```
- The resume is sent to the Flask backend as a POST request.
- The backend uses SpaCy for text preprocessing and applies the Sentence-BERT model to calculate semantic similarity between the user's resume and the job descriptions dataset.
```

### Result Display:
```
- The backend returns the top matching job titles along with similarity scores.
- The frontend dynamically displays the results in an easy-to-read format.
```
