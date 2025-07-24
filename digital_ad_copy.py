import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import os

from langchain_community.chat_models import ChatOpenAI  # ✅ Use this import
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ===================
# Load API Key
# ===================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment!")

llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-4", temperature=0.7)

# ===================
# Streamlit UI
# ===================
st.set_page_config(page_title="Digital Ad Strategist", layout="wide")
st.title("📊 AI-Powered Facebook Ad Generator")

# Upload Poster
poster = st.file_uploader("📤 Upload your campaign poster (optional)", type=["jpg", "jpeg", "png", "pdf"])

# Campaign Inputs
st.subheader("📝 Campaign Details")
ad_topic = st.text_input("1. What is the ad about?")
company_name = st.text_input("2. Company Name")
target_audience = st.text_area("3. Target Audience Description")
campaign_objective = st.selectbox(
    "4. Choose Campaign Objective",
    ["Awareness", "Traffic", "Engagement", "Leads", "App Promotion", "Sales"]
)

# Show uploaded poster
if poster:
    st.image(Image.open(poster), caption="Uploaded Poster", use_column_width=True)

# Prompt Template
prompt_template = PromptTemplate.from_template("""
You are a world-class digital marketer and ad strategist.

Input:
- Ad Topic: {ad_topic}
- Company: {company_name}
- Target Audience: {target_audience}
- Objective: {campaign_objective}

Tasks:
1. Define a detailed customer avatar.
2. Describe the full customer journey from awareness to conversion.
3. Generate 3 attention-grabbing Facebook Ad Headlines.
4. Write a high-converting Facebook Ad Copy using the AIDA framework (Attention, Interest, Desire, Action).

Output in markdown with headings for each section.
""")

# Process Input
if st.button("🚀 Generate Ad Strategy"):
    if not ad_topic or not company_name or not target_audience:
        st.warning("Please fill in all required fields.")
    else:
        chain = LLMChain(llm=llm, prompt=prompt_template)
        response = chain.run({
            "ad_topic": ad_topic,
            "company_name": company_name,
            "target_audience": target_audience,
            "campaign_objective": campaign_objective
        })

        st.markdown("---")
        st.subheader("🧠 AI-Generated Ad Strategy")
        st.markdown(response)
