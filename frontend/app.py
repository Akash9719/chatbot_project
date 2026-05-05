import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials
import json
from datetime import datetime

# -----------------------
# Page Setup
# -----------------------
st.set_page_config(page_title="Rishikirti AI Assistant")
st.title("💬 Rishikirti AI Assistant")

# -----------------------
# Setup
# -----------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------
# Load Knowledge
# -----------------------
@st.cache_resource
def load_vector_store():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "knowledge.txt")

    if not os.path.exists(file_path):
        st.error(f"❌ knowledge.txt not found at: {file_path}")
        st.stop()

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    embeddings = embed_model.encode(chunks)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    return index, chunks

index, chunks = load_vector_store()

# -----------------------
# Retrieval
# -----------------------
def retrieve(query, k=3):
    q_vec = embed_model.encode([query])
    D, I = index.search(np.array(q_vec), k)

    results = []
    for i, score in zip(I[0], D[0]):
        if score < 1.5:
            results.append(chunks[i])

    return "\n".join(results)

# -----------------------
# Google Sheets Save
# -----------------------
def save_to_google_sheets(name, email, requirement):
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
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
# Chat Memory
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------
# Display Chat
# -----------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------
# User Input
# -----------------------
user_input = st.chat_input("Ask about our services...")

# -----------------------
# Chat Logic
# -----------------------
if user_input:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    context = retrieve(user_input)

    if not context:
        context = "Rishikirti Technologies provides ERP customization, data analytics, and design services."

    # SYSTEM PROMPT (Sales + Smart)
    system_prompt = """
You are a smart business consultant for Rishikirti Technologies.

Your goals:
- Understand the user's requirement
- Suggest relevant services (ERP, analytics, design)
- Ask follow-up questions
- Guide user toward sharing their requirement

Rules:
- Keep answers under 4-5 lines
- Be conversational and helpful
"""

    # Build messages
    messages = [{"role": "system", "content": system_prompt}]

    for msg in st.session_state.messages[-5:]:
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
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.4
            )

        bot_reply = response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Groq Error: {str(e)}")
        st.stop()

    # Save response
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    with st.chat_message("assistant"):
        st.write(bot_reply)

    # Smart lead trigger
    if any(word in user_input.lower() for word in ["price", "cost", "demo", "interested", "contact"]):
        st.info("👉 Looks like you're interested. Please fill the form below and our team will reach out!")

# -----------------------
# Lead Capture Form
# -----------------------
st.divider()
st.subheader("📩 Get in Touch")

name = st.text_input("Your Name")
email = st.text_input("Your Email")
requirement = st.text_area("Your Requirement")

consent = st.checkbox("I agree to share my information for contact purposes")

if st.button("Submit"):
    if name and email and requirement and consent:
        try:
            save_to_google_sheets(name, email, requirement)
            st.success("✅ Thank you! Our team will contact you soon.")
        Except Exception as e:
            st.error(f"Error saving leads: {str(e)}")
    else:
        st.warning("Please fill all fields and accept consent")
