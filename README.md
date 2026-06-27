# AI Resume Analyzer

An elegant, modern dark-themed web application that analyzes PDF resumes using Google's Gemini 2.5 Flash model. It extracts text from the resume, processes its layout and structure, identifies key skills, and computes an ATS score. Optionally, it compares the resume with a target Job Description to compute a Match Score, highlights strengths & weaknesses, and recommends actionable modifications to land more interviews.

## Features

- **Sleek, Dark-Themed UI**: Built using Streamlit with premium glassmorphic visual containers, dynamic badges, and clear metric dashboards.
- **PDF Extraction**: Seamless text parsing from uploaded PDF resumes.
- **ATS Compatibility Score**: Realistic calculation based on key sections, standard headings, and layout parsing.
- **Job Match Score**: Instant alignment metrics showing how well the resume matches a target Job Description.
- **Skills Analysis**: Compares skills found vs. missing/weak skills.
- **Recruiter Feedback**: Detailed list of strengths, gaps, and actionable recommendations to improve.

## Setup & Installation

### 1. Prerequisites

Make sure you have Python 3.9+ installed on your system.

### 2. Clone/Copy Project Files

Navigate to the project directory:
```bash
cd C:\Users\bthar\.gemini\antigravity\scratch\ai_resume_analyzer
```

### 3. Install Dependencies

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### 4. Configure API Key

Get a Gemini API Key from [Google AI Studio](https://aistudio.google.com/).
Create a `.env` file in the root directory and add your key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```
*(Alternatively, you can paste the API key directly into the sidebar of the web application at runtime).*

## Running the Application

Start the Streamlit development server:
```bash
streamlit run app.py
```

The application will launch in your default web browser (usually at `http://localhost:8501`).
