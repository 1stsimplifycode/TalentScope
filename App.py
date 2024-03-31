import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from pdfminer.high_level import extract_text
import base64
from streamlit_tags import st_tags
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import spacy
from collections import Counter
import re

# Initialize session state for role selection
if 'role' not in st.session_state:
    st.session_state['role'] = None

def set_role(role):
    st.session_state['role'] = role

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Check if NLTK data is downloaded, if not download it
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Define skills, job roles, and experiences
SKILLS_KEYWORDS = {
    "python", "java", "sql", "javascript", "react", "node.js", "django", "flask", 
    "tensorflow", "pytorch", "design", "illustrator", "photoshop", "ui", "ux", 
    "data analysis", "machine learning", "deep learning", "statistical analysis"
}

EXPERIENCE_KEYWORDS = {
    "worked", "developed", "designed", "implemented", "created", "managed"
}

JOB_SKILLS_MAP = {
    "data scientist": {
        "python", "tensorflow", "pytorch", "sql", "data analysis", 
        "machine learning", "statistical analysis", "big data technologies", "cloud computing"
    },
    "graphic designer": {
        "design", "illustrator", "photoshop", "indesign", "creative thinking", "visual communication"
    },
    "ui/ux designer": {
        "ui", "ux", "design", "prototyping", "user research", "sketch", "figma", "adobe xd"
    },
    "software developer": {
        "java", "python", "javascript", "node.js", "django", "flask", "c#", ".net", "react", "angular"
    },
    "product manager": {
        "market research", "product strategy", "user stories", 
        "roadmap planning", "agile methodologies", "competitive analysis"
    },
    "cloud engineer": {
        "aws", "azure", "gcp", "docker", "kubernetes", "cloud security", "devops practices"
    },
    "cybersecurity specialist": {
        "firewalls", "network security", "incident response", 
        "vulnerability assessment", "penetration testing", "cryptography"
    },
    "network administrator": {
        "lan/wan", "vpn", "network configuration", "cisco technologies", 
        "troubleshooting", "wireless networking"
    },
    "devops engineer": {
        "jenkins", "git", "ansible", "terraform", "docker", "kubernetes", "ci/cd pipelines", "scripting"
    },
    "data analyst": {
        "sql", "python", "data visualization", "excel", "power bi", "tableau", "statistics"
    },
    "project manager": {
        "project planning", "risk management", "scrum", "kanban", 
        "stakeholder management", "budget management"
    },
    "qa engineer": {
        "test automation", "selenium", "quality assurance", "bug tracking", 
        "performance testing", "cicd integration"
    },
    "business analyst": {
        "business process modeling", "requirements analysis", 
        "data analysis", "erp", "crm", "stakeholder communication"
    },
    "blockchain developer": {
        "smart contracts", "ethereum", "solidity", "blockchain architecture", "dapp development"
    },
    "ai/machine learning engineer": {
        "python", "tensorflow", "pytorch", "machine learning algorithms", 
        "neural networks", "natural language processing"
    },
    "iot developer": {
        "iot platforms", "mqtt", "coap", "iot security", "embedded systems", "sensor networks"
    },
    "big data engineer": {
        "hadoop", "spark", "kafka", "big data analytics", "nosql databases", "data warehousing"
    },
    "augmented reality (ar)/virtual reality (vr) developer": {
        "unity", "unreal engine", "c#", "3d modeling", "ar sdk", "vr sdk", "user experience design"
    },
    "ethical hacker": {
        "penetration testing", "security audits", "vulnerability scanning", "incident handling", "cybersecurity frameworks"
    },
    "android developer": {
        "java", "kotlin", "android sdk", "android studio", "material design", 
        "api integration", "performance optimization", "security best practices"
    }
}


extracted_skills = set()
extracted_experiences = []

def extract_info(text):
    global extracted_skills, extracted_experiences
    stop_words = set(stopwords.words('english'))
    words = set(word_tokenize(text.lower()))
    sentences = sent_tokenize(text.lower())

    extracted_skills = {word for word in words if word in SKILLS_KEYWORDS}
    extracted_experiences = [sentence for sentence in sentences if any(keyword in sentence for keyword in EXPERIENCE_KEYWORDS)]

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        file_content = f.read()
    base64_pdf = base64.b64encode(file_content).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def determine_job_fit():
    potential_fits = []
    for job, skills in JOB_SKILLS_MAP.items():
        if any(skill in extracted_skills for skill in skills):
            potential_fits.append(job)
    return potential_fits

def simple_qa(input_question):
    input_question_lower = input_question.lower()
    if "best fit for" in input_question_lower or "what is the resume best fit for" in input_question_lower:
        potential_fits = determine_job_fit()
        return f"The resume seems best fit for: {', '.join(potential_fits)}." if potential_fits else "No clear fit based on the extracted skills."
    elif "how can i improve my resume" in input_question_lower:
        # Provide a set of general tips for improving a resume
        resume_tips = [
            "Ensure your resume is clear and concise; avoid unnecessary jargon or complex language.",
            "Tailor your resume for each job application to match the job description and highlight relevant skills and experiences.",
            "Quantify your achievements where possible, using specific numbers or outcomes to demonstrate your impact.",
            "Include a mix of both hard skills (e.g., Python programming) and soft skills (e.g., teamwork, communication).",
            "Keep your design simple and professional; use bullet points for easier readability.",
            "Ensure there are no spelling or grammatical errors; consider having someone else review your resume.",
            "Include a summary statement at the top that clearly outlines your career objectives and key strengths.",
            "If applicable, add links to your professional online profiles (e.g., LinkedIn) or personal projects (e.g., GitHub, portfolio website)."
        ]
        tips_response = "Here are some tips to improve your resume:\n- " + "\n- ".join(resume_tips)
        return tips_response
    else:
        return "I'm not sure how to answer that. Could you try asking something else?"


def run_normal_user():
    st.title("Talent Scout: Your Resume Advisor")
    uploaded_file = st.file_uploader("Upload your resume in PDF format", type=["pdf"])
    if uploaded_file is not None:
        temp_file_path = "uploaded_resume.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        show_pdf(temp_file_path)
        extracted_text = extract_text(temp_file_path)
        extract_info(extracted_text)
        
        if extracted_skills:
            st.success("Extracted Skills:")
            st_tags(label='### Skills extracted from the resume:', value=list(extracted_skills), key='1')
        
        if extracted_experiences:
            st.success("Extracted Experiences:")
            for experience in extracted_experiences:
                st.write(experience)
        else:
            st.warning("No recognizable experiences found.")

        # Calculate and display the resume score
        score, feedback = calculate_resume_score(extracted_skills, extracted_experiences)
        st.metric(label="Resume Score", value=f"{score}/5")
        st.info(feedback)
    
        st.subheader("Ask a Question About Job Fit or How the Resume Suits a Specific Role:")
        user_question = st.text_input("Type your question here:")
        if user_question:
            response = simple_qa(user_question)
            st.text_area("Response", value=response, height=100, help=None)

def calculate_resume_score(skills, experiences):
    # Example scoring logic: this is very simplistic and should be replaced with your actual scoring logic
    score = 0
    feedback = []

    # Increase score for each skill and experience found, up to a maximum score of 5
    score += min(len(skills), 2)  # Let's assume a max of 2 points for skills
    score += min(len(experiences), 3)  # And a max of 3 points for experiences

    # Provide feedback based on the score
    if score <= 2:
        feedback.append("Consider enhancing your resume with more relevant skills and experiences.")
    elif score <= 4:
        feedback.append("Good job! Your resume is on the right track, but there's room for improvement.")
    else:
        feedback.append("Excellent! Your resume is well-tailored for the jobs you're targeting.")

    return score, " ".join(feedback)

def extract_keywords(text):
    # Simple keyword extraction through basic string manipulation
    # Here, you can add more sophisticated methods for keyword extraction
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    stop_words = set(stopwords.words('english'))
    keywords = [token for token in tokens if token not in stop_words]
    return keywords

def match_candidates(job_description, df):
    keywords = extract_keywords(job_description)
    matched_candidates = pd.DataFrame()
    
    for keyword in keywords:
        # Assuming 'Skills' column contains comma-separated skills
        matched = df[df['Skills'].str.contains(keyword, case=False, na=False)]
        matched_candidates = pd.concat([matched_candidates, matched]).drop_duplicates().reset_index(drop=True)
    
    return matched_candidates

def simple_qa_admin(input_question, df):
    # Extract keywords as before
    keywords = extract_keywords(input_question)
    matched_candidates = pd.DataFrame()

    # Attempt to parse years of experience
    years_experience_required = 0
    years_match = re.search(r'\b(\d+)\syears', input_question)
    if years_match:
        years_experience_required = int(years_match.group(1))

    for keyword in keywords:
        # Searching in both 'Skills' and 'Previous Role' for a broader match, including years of experience
        matched = df[((df['Skills'].str.contains(keyword, case=False, na=False)) | 
                      (df['Previous Role'].str.contains(keyword, case=False, na=False))) & 
                     (df['Experience (Years)'] >= years_experience_required)]
        matched_candidates = pd.concat([matched_candidates, matched]).drop_duplicates().reset_index(drop=True)

    if not matched_candidates.empty:
        return matched_candidates
    else:
        return "No candidates found matching the criteria."

# The rest of your run_admin function remains unchanged

def run_admin():
    st.title("Admin Dashboard")
    df_applicants = pd.read_csv("C:/Users/mailt/Documents/AI_RESUME_SUPPORT/candidate_db.csv")

    # Skills Distribution Pie Chart
    all_skills = df_applicants['Skills'].str.split(',').explode()
    fig_skills = px.pie(all_skills.value_counts(), names=all_skills.value_counts().index, title='Overall Skills Distribution')
    st.plotly_chart(fig_skills)

    # Experience Distribution Bar Chart
    fig_experience = px.bar(df_applicants['Experience (Years)'].value_counts().sort_index(), 
                            x=df_applicants['Experience (Years)'].value_counts().sort_index().index, 
                            y=df_applicants['Experience (Years)'].value_counts().sort_index(), 
                            labels={'x':'Years of Experience', 'y':'Number of Candidates'}, 
                            title='Experience Distribution')
    st.plotly_chart(fig_experience)

    # Previous Role Distribution Pie Chart
    fig_prev_role = px.pie(df_applicants, names='Previous Role', title='Distribution by Previous Role')
    st.plotly_chart(fig_prev_role)

    # Handling Job Description and Hiring Questions
    st.subheader("Input Job Description or Hiring Question:")
    job_description = st.text_area("Job Description")
    user_question = st.text_input("Enter your hiring question here:")
    if st.button("Find Matching Candidates"):
        if job_description:
            matches = match_candidates(job_description, df_applicants)
            st.write("Matching candidates based on the job description:")
            st.dataframe(matches)
        if user_question:
            response = simple_qa_admin(user_question, df_applicants)
            st.write("Matching candidates based on the hiring question:")
            if isinstance(response, pd.DataFrame):
                st.dataframe(response)
            else:
                st.write(response)

st.header("Welcome to the Resume and Recruitment Platform")
col1, col2 = st.columns(2)

with col1:
    if st.button("Resume Analyzer"):
        set_role("Normal User")

with col2:
    if st.button("Recruit"):
        set_role("Admin")

if st.session_state['role'] == "Normal User":
    run_normal_user()
elif st.session_state['role'] == "Admin":
    run_admin()
else:
    st.info("Please select a role to get started.")
