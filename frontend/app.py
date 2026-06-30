import os
import streamlit as st
from groq import Groq

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials
import json
from datetime import datetime

# -----------------------
# Page Setup
# -----------------------

st.set_page_config(
    page_title="RishiKirti AI Assistant",
    page_icon="Kirti.png",
    layout="wide"
)

# -----------------------
# Load CSS
# -----------------------

current_dir = os.path.dirname(__file__)
css_file = os.path.join(current_dir, "style.css")

with open(css_file, encoding="utf-8") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# -----------------------
# Secrets Helper
# -----------------------
def get_secret(key):
    value = os.getenv(key)

    if value:
        return value

    return st.secrets[key]

# -----------------------
# Groq Setup
# -----------------------
groq_api_key = get_secret("GROQ_API_KEY")

client = Groq(
    api_key=groq_api_key
)

@st.cache_resource
def load_knowledge():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "knowledge.txt")

    if not os.path.exists(file_path):
        st.error(f"❌ knowledge.txt not found at: {file_path}")
        st.stop()

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

knowledge = load_knowledge()

def retrieve(query):
    return knowledge

# -----------------------
# Google Sheets Save
# -----------------------
def save_to_google_sheets(name, email, requirement):

    secret_value = get_secret("GOOGLE_CREDENTIALS")

    
    if secret_value.startswith('"'):
        secret_value = json.loads(secret_value)
    
    creds_dict = json.loads(secret_value)

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client_gs = gspread.authorize(creds)
    sheet = client_gs.open("Leads").sheet1

    sheet.append_row([
        name,
        email,
        requirement,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ])

# -----------------------
# Session State
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": """Hi, Welcome to RishiKirti Technologies.

I am Kirti, your Virtual Assistant.

How can I help you today?"""
        }
    ]

if "show_form" not in st.session_state:
    st.session_state.show_form = False

# -----------------------
# Display Chat History
# -----------------------

for msg in st.session_state.messages:
    avatar = (
        "https://rishikirti.com/image/Kirti.png"
        if msg["role"] == "assistant"
        else "👤"
    )
    with st.chat_message(
        msg["role"],
        avatar=avatar
    ):
        st.markdown(msg["content"])

# -----------------------
# User Input
# -----------------------
user_input = st.chat_input("Ask about our services...")

# -----------------------
# Chat Logic
# -----------------------
if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message(
        "user",
        avatar="👤"
    ):
        st.write(user_input)

    context = retrieve(user_input)

    if not context:
        context = "Rishikirti Technologies provides ERP customization, data analytics, and design services."

    # 🔥 Intent Detection
    intent_keywords = ["price", "cost", "demo", "interested", "contact", "connect", "talk","hire","talk","human","consultant","call", "meeting", "proposal"]

    if any(word in user_input.lower() for word in intent_keywords):
        st.session_state.show_form = True

    # SYSTEM PROMPT
    system_prompt = """
You are Kirti, the AI Virtual Assistant of Rishikirti Technologies.

Your personality:
- Professional
- Helpful
- Friendly
- Conversational
- Confident

Introduce yourself only once at the beginning of a new conversation.
After that:
- Do not repeat your name in every response.
- Respond naturally and conversationally.
- Focus on answering the user's question.

Keep responses concise and easy to read.

Company Services:
- Oracle ERP solutions
- JD Edwards ERP solutions
- ESG analytics and sustainability research & dashboards
- Data analytics and business intelligence
- Process improvement and workflow optimization
- Python and VBA automation solutions
- Machine learning and predictive analytics
- AI tool development

Your goals:
- Understand the user's requirement
- Suggest relevant services (ERP, analytics, design)
- Provide useful recommendations first
- Ask follow-up questions only when necessary

Important:
- Do not ask a question in every response.
- If the user's requirement is already clear, explain how RishiKirti can help.
- Then ask at most one follow-up question.
- Avoid long chains of questioning.

Rules:
- Keep answers under 4-5 lines
- Be conversational and helpful
- Our company ONLY provides Oracle ERP and JD Edwards ERP solutions.
- NEVER recommend SAP, Microsoft Dynamics, Odoo, NetSuite, or any competing ERP platforms.
- If users ask for ERP recommendations, ONLY recommend:
  - Oracle ERP
  - JD Edwards ERP
- If SAP or another competitor is mentioned:
  - Politely redirect the conversation toward Oracle and JD Edwards capabilities.
  - Do not compare competitors positively.
- For SMEs:
  - Suggest JD Edwards for flexible mid-sized operations.
  - Suggest Oracle ERP Cloud for scalable enterprise transformation

  Lead Conversion Rules:
  - When the user's requirement becomes clear, stop asking repetitive discovery questions.
  - Summarize the user's requirement in 1-2 sentences.
  - Explain briefly how RishiKirti can help.
  - If the user shows buying intent, hiring intent, implementation intent, asks for pricing, asks to connect, asks for a demo, or asks for a human: immediately guide them toward the contact form.
  - Do not continue asking unnecessary follow-up questions once the requirement is understood.
  - Focus on helping the user take the next step.

Always keep responses aligned with our company offerings.

"""

    messages = [{"role": "system", "content": system_prompt}]

    for msg in st.session_state.messages[-8:]:
        messages.append(msg)

    messages.append({
        "role": "user",
        "content": f"""
Context:
{context}

User Query:
{user_input}
"""
    })

    try:
        with st.spinner("Kirti is typing..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.4
            )

        bot_reply = response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Groq Error: {str(e)}")
        st.stop()

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    with st.chat_message(
        "assistant",
        avatar="https://rishikirti.com/image/Kirti.png"
    ):
        st.markdown(bot_reply)

    # Soft CTA
    if st.session_state.show_form:
        st.info("👉 Would you like to connect with our team? You can share your details below.")

# -----------------------
# Lead Capture Form (ONLY ON INTENT)
# -----------------------
if st.session_state.show_form:

    st.divider()
    st.subheader("📩 Get in Touch")

    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    Phone = st.text_input("Your Phone No.")
    requirement = st.text_area("Your Requirement")

    consent = st.checkbox("I agree to share my information for contact purposes")

    if st.button("Submit"):
        if name.strip() and email.strip() and requirement.strip() and consent:
            try:
                save_to_google_sheets(name, email, requirement)
                st.success("✅ Thank you! Our team will contact you soon.")
                st.session_state.show_form = False   # hide after submit
            except Exception as e:
                st.error(f"Error saving leads: {str(e)}")
        else:
            st.warning("Please fill all fields and accept consent")


st.markdown("""
<div style="text-align:center;
font-size:12px;
color:#7f8c8d;
padding:10px;">

Powered by <b>RishiKirti AI</b>

</div>
""", unsafe_allow_html=True)
