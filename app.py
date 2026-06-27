import os
import json
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pypdf import PdfReader
from pydantic import BaseModel, Field
from typing import List
import traceback

# Load environment variables
load_dotenv()

# Page configuration with a professional theme
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium dark theme design
st.markdown("""
<style>
    /* Dark Theme Base styling */
    .stApp {
        background-color: #0d0f17;
        color: #e2e8f0;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #151926;
        border-right: 1px solid #232d42;
    }
    
    /* Glowing main header */
    .main-title {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        text-align: center;
        font-weight: 400;
    }
    
    /* Card container styling */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Metric Score containers */
    .score-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
    }
    .score-value {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        margin: 10px 0;
    }
    .score-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Progress bar custom color */
    div[data-testid="stProgress"] > div > div > div > div {
        background-color: #6366f1;
    }
    
    /* Accordion header styling override */
    .streamlit-expanderHeader {
        background-color: rgba(30, 41, 59, 0.3) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Custom bullets for list items */
    .list-item {
        margin-bottom: 10px;
        font-size: 1rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Define Pydantic Schema for structured outputs from Gemini
class ResumeAnalysis(BaseModel):
    summary: str = Field(description="A concise professional summary of the candidate based on their resume (3-4 sentences).")
    skills_found: List[str] = Field(description="Key skills, technologies, and methodologies found in the resume.")
    missing_skills: List[str] = Field(description="Key skills or requirements from the Job Description that are missing or weak in the resume. If no Job Description is provided, list general skills that would enhance this candidate's profile.")
    ats_score: int = Field(description="ATS compatibility score (0-100) based on standard industry keyword matching, layout parsing readability, and structure.")
    match_score: int = Field(description="Matching score (0-100) indicating how well the candidate's skills and experience match the provided Job Description. If no Job Description is provided, this must be 0.")
    suggestions: List[str] = Field(description="Actionable, specific suggestions to improve the resume (e.g., formatting improvements, wording, adding projects, metrics).")
    strengths: List[str] = Field(description="Core professional strengths demonstrated in the resume.")
    weaknesses: List[str] = Field(description="Gaps or weaknesses in the candidate's profile relative to the job description, or general career gaps (e.g. lack of metrics, gaps in employment, missing certs).")

def extract_text_from_pdf(uploaded_file) -> str:
    """Extracts all text from an uploaded PDF file."""
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Failed to read PDF file: {e}")
        return ""

def analyze_resume_api(resume_text: str, job_desc: str, api_key: str) -> ResumeAnalysis:
    """Calls Gemini API with structured output configuration to analyze the resume."""
    # Ensure API key has no trailing/leading whitespace or newlines
    clean_key = api_key.strip()
    
    # Log key signature for diagnostics (safe to do since it only prints prefix and length)
    print(f"[Diagnostics] Cleaning API key. Length: {len(clean_key)}, Prefix: '{clean_key[:6]}...'")
    
    # Using the new Google GenAI SDK
    client = genai.Client(api_key=clean_key)
    
    prompt = f"""
    You are an expert ATS (Applicant Tracking System) optimizer and professional recruiter.
    Analyze the following resume text and compare it with the provided job description (if any).

    Resume Text:
    \"\"\"{resume_text}\"\"\"

    Job Description:
    \"\"\"{job_desc if job_desc else "None provided. Analyze the resume generally."}\"\"\"

    Provide the analysis strictly according to the requested JSON schema.
    Ensure your ratings are realistic:
    - The ATS Score (0-100) should reflect formatting, structure, and keyword density.
    - The Match Score (0-100) should reflect how well the candidate's skills and experience align with the Job Description. If no Job Description is provided, return 0.
    - Fill all list fields with relevant, professional points.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ResumeAnalysis,
            temperature=0.2
        )
    )
    
    # Parse the structured JSON response
    data = json.loads(response.text)
    return ResumeAnalysis(**data)

# Initialize Session State
if 'analysis' not in st.session_state:
    st.session_state.analysis = None
if 'resume_name' not in st.session_state:
    st.session_state.resume_name = None
if 'api_key_val' not in st.session_state:
    st.session_state.api_key_val = os.getenv("GEMINI_API_KEY") or ""

# Sidebar layout
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.markdown("### Settings")
    
    # API Key Configuration without key parameter to prevent locking
    api_key_input = st.text_input(
        "Gemini API Key", 
        type="password", 
        value=st.session_state.api_key_val,
        help="Get your API key from Google AI Studio: https://aistudio.google.com/"
    )
    
    # Update session state value dynamically as the user types
    if api_key_input != st.session_state.api_key_val:
        st.session_state.api_key_val = api_key_input
        st.rerun()
        
    # Reset Button
    if st.button("Reset App & Clear Key", type="secondary", use_container_width=True):
        st.session_state.analysis = None
        st.session_state.resume_name = None
        st.session_state.api_key_val = ""
        st.rerun()
        
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This app parses your resume, measures ATS compliance, and checks alignment with any target Job Description."
    )
    st.caption("Powered by Gemini 2.5 Flash")

# Main Content Header
st.markdown("<h1 class='main-title'>AI Resume Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Scan, score, and optimize your resume for applicant tracking systems in seconds</p>", unsafe_allow_html=True)

# Main Application Form / Workspace
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("### 📥 Input Panel")
    
    # 1. Resume Upload
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF format only)", 
        type=["pdf"], 
        help="Make sure the file is not password protected."
    )
    
    # 2. Job Description
    job_desc = st.text_area(
        "Target Job Description (Optional)", 
        height=250, 
        placeholder="Paste the job description here to calculate a Match Score and discover missing skills..."
    )
    
    # 3. Submit Button
    api_key = st.session_state.api_key_val
    submit_disabled = uploaded_file is None or not api_key
    
    if not api_key:
        st.warning("⚠️ Please provide a Gemini API Key in the sidebar settings to begin.")
    elif uploaded_file is None:
        st.info("💡 Upload a PDF resume to start the analysis.")
        
    analyze_btn = st.button(
        "Analyze Resume", 
        type="primary", 
        use_container_width=True, 
        disabled=submit_disabled
    )

with col2:
    st.markdown("### 📊 Analysis Results")
    
    # Run analysis
    if analyze_btn and uploaded_file and api_key:
        with st.spinner("Parsing resume and calling Gemini API..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if resume_text:
                try:
                    # Clean the API Key and send it directly to the model
                    clean_key = api_key.strip()
                    analysis_result = analyze_resume_api(resume_text, job_desc, clean_key)
                    st.session_state.analysis = analysis_result
                    st.session_state.resume_name = uploaded_file.name
                    st.success("Analysis complete!")
                except Exception as e:
                    # Print full traceback to standard error / logs for easy debugging
                    traceback.print_exc()
                    st.error(f"Error during API call: {e}")
                    st.info("Please verify that your Gemini API Key is valid and active in Google AI Studio.")
            else:
                st.error("Could not extract text from the uploaded PDF. Please make sure it contains readable text.")

    # Render results from session state (if available)
    if st.session_state.analysis:
        res = st.session_state.analysis
        
        # Color codes based on scores
        def get_score_color(score):
            if score >= 80: return "#10b981"  # Emerald Green
            if score >= 50: return "#f59e0b"  # Amber Orange
            return "#ef4444"                  # Rose Red
            
        ats_color = get_score_color(res.ats_score)
        match_color = get_score_color(res.match_score)
        
        # Display Summary cards side-by-side
        score_col1, score_col2 = st.columns(2)
        
        with score_col1:
            st.markdown(f"""
            <div class="score-container">
                <div class="score-label">ATS Compatibility</div>
                <div class="score-value" style="color: {ats_color};">{res.ats_score}</div>
                <div style="font-size: 0.8rem; color: #64748b;">industry standards</div>
            </div>
            """, unsafe_allow_html=True)
            
        with score_col2:
            if job_desc.strip():
                st.markdown(f"""
                <div class="score-container">
                    <div class="score-label">Job Match Score</div>
                    <div class="score-value" style="color: {match_color};">{res.match_score}</div>
                    <div style="font-size: 0.8rem; color: #64748b;">alignment with JD</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="score-container">
                    <div class="score-label">Job Match Score</div>
                    <div class="score-value" style="color: #64748b;">N/A</div>
                    <div style="font-size: 0.8rem; color: #64748b;">no JD provided</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Resume Summary Card
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="margin-top: 0; color: #6366f1;">Summary</h4>
            <p style="font-size: 1rem; line-height: 1.6; color: #cbd5e1;">{res.summary}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create Tabs for detailed analysis
        tab1, tab2, tab3 = st.tabs(["🛠️ Skills Overview", "⚖️ Strengths & Gaps", "📈 Recommendations"])
        
        with tab1:
            st.markdown("#### Skills Found in Resume")
            if res.skills_found:
                # Display skills as badges
                skills_html = "".join([f'<span style="background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 20px; padding: 4px 12px; margin: 4px; display: inline-block; font-size: 0.85rem;">{skill}</span>' for skill in res.skills_found])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.info("No specific skills identified.")
                
            st.markdown("<br>#### Missing / Weak Skills", unsafe_allow_html=True)
            if res.missing_skills:
                # Display missing skills as red-tinted badges
                missing_html = "".join([f'<span style="background: rgba(239, 68, 68, 0.1); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 20px; padding: 4px 12px; margin: 4px; display: inline-block; font-size: 0.85rem;">{skill}</span>' for skill in res.missing_skills])
                st.markdown(missing_html, unsafe_allow_html=True)
            else:
                st.success("No critical missing skills identified!")
                
        with tab2:
            st.markdown("#### Key Strengths")
            for strength in res.strengths:
                st.markdown(f"🟢 **{strength}**")
                
            st.markdown("<br>#### Profile Gaps / Weaknesses", unsafe_allow_html=True)
            for weakness in res.weaknesses:
                st.markdown(f"🔴 **{weakness}**")
                
        with tab3:
            st.markdown("#### Actionable Suggestions to Improve")
            for i, suggestion in enumerate(res.suggestions, 1):
                st.markdown(f"**{i}.** {suggestion}")
                
    else:
        # Placeholder visual when no resume has been analyzed yet
        st.markdown(
            """
            <div style="text-align: center; padding: 60px 20px; border: 2px dashed rgba(255,255,255,0.05); border-radius: 16px; background: rgba(30, 41, 59, 0.1);">
                <img src="https://img.icons8.com/external-flatart-icons-outline-flatarticons/64/6366f1/external-resume-job-search-flatart-icons-outline-flatarticons-2.png" style="opacity: 0.5; margin-bottom: 15px;"/>
                <p style="color: #64748b; font-size: 1.1rem;">Waiting for resume analysis...</p>
                <p style="color: #475569; font-size: 0.9rem;">Upload a resume and click "Analyze Resume" to view details</p>
            </div>
            """,
            unsafe_allow_html=True
        )
