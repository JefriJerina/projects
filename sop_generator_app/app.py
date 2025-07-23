import streamlit as st
import pdfplumber
import fitz  # PyMuPDF
import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ============================
# Utils: Extract Questions
# ============================
@st.cache_data
def extract_questions_from_pdf(pdf_file):
    questions = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for line in text.split("\n"):
                    if "?" in line:
                        questions.append(line.strip())
    return questions


# ============================
# LangChain SOP Generator
# ============================
def get_sop_chain():
    template = """
    You are an SOP expert. Use the following inputs to write a personalized, original, and well-structured SOP.

    Questions to Answer:
    {questions}

    Academic Background:
    {academic_details}

    Resume Summary:
    {resume_summary}

    Country: {country}
    University: {university}
    Course: {course}

    Instructions:
    - Word Count: 500-800 words
    - Avoid plagiarism
    - Keep it formal and goal-oriented

    Now write the SOP.
    """

    prompt = PromptTemplate(
        input_variables=["questions", "academic_details", "resume_summary", "country", "university", "course"],
        template=template,
    )

    llm = ChatOpenAI(temperature=0.7, model="gpt-4", openai_api_key=openai_api_key)
    return LLMChain(llm=llm, prompt=prompt)


# ============================
# Streamlit UI
# ============================
st.set_page_config(page_title="GenAI SOP Generator", layout="centered")
st.title("🎓 GenAI SOP Generator")

# Upload SOP Questionnaire
st.subheader("📄 Step 1: Upload SOP Questionnaire (PDF)")
questionnaire_pdf = st.file_uploader("Upload SOP Questionnaire PDF", type="pdf")

# Upload Academic Docs
st.subheader("📚 Step 2: Upload Academic Documents (PDF only)")
uploaded_files = st.file_uploader(
    "Upload 10th, 12th, UG Marksheets and Resume", type="pdf", accept_multiple_files=True
)

# Preferences
st.subheader("🌍 Step 3: Enter Study Preferences")
country = st.text_input("Target Country")
university = st.text_input("Target University")
course = st.text_input("Target Course")

if st.button("🚀 Generate SOP"):
    if not questionnaire_pdf:
        st.warning("Please upload the SOP questionnaire PDF.")
    elif not uploaded_files or not country or not university or not course:
        st.warning("Please upload all academic documents and fill in study preferences.")
    else:
        with st.spinner("Extracting questions and processing documents..."):
            questions = extract_questions_from_pdf(questionnaire_pdf)
            questions_str = "\n".join(questions)

            academic_details = ""
            resume_summary = ""

            for file in uploaded_files:
                doc = fitz.open(stream=file.read(), filetype="pdf")
                text = "\n".join([page.get_text() for page in doc])
                if "resume" in file.name.lower():
                    resume_summary += text + "\n"
                else:
                    academic_details += text + "\n"

            academic_details = academic_details[:3000]
            resume_summary = resume_summary[:3000]

            sop_chain = get_sop_chain()
            response = sop_chain.run({
                "questions": questions_str,
                "academic_details": academic_details,
                "resume_summary": resume_summary,
                "country": country,
                "university": university,
                "course": course,
            })

        st.success("✅ SOP Generated Successfully!")
        st.text_area("📜 SOP Preview", response, height=400)
        st.download_button("📥 Download SOP", response, file_name="generated_sop.txt")

