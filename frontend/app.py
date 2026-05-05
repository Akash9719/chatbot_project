import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------
# Setup
# -----------------------
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------
# Load Knowledge (FIXED PATH)
# -----------------------
@st.cache_resource
def load_vector_store():
    # Get path of current file (app.py)
    current_dir = os.path.dirname(__file__)

    # Build correct path to knowledge.txt
    file_path = os.path.join(current_dir, "knowledge.txt")

    # Debug (optional)
    # st.write("Looking for file at:", file_path)

    if not os.path.exists(file_path):
        st.error(f"❌ knowledge.txt not found at: {file_path}")
        st.stop()

    # Read file
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into chunks
    chunks = text.split("\n\n")

    # Create embeddings
    embeddings = embed_model.encode(chunks)

    # Create FAISS index
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    return index, chunks

# Load once
index, chunks = load_vector_store()

# -----------------------
# Memory
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Rishikirti AI Assistant")
st.title("💬 Rishikirti AI Assistant")

user_input = st.chat_input("Ask about our services...")

# -----------------------
# Retrieval
# -----------------------
def retrieve(query, k=2):
    q_vec = embed_model.encode([query])
    D, I = index.search(np.array(q_vec), k)
    return "\n".join([chunks[i] for i in I[0]])

# -----------------------
# Chat Logic
# -----------------------
if user_input:

    if not user_input.strip():
        st.warning("Please enter a valid question")
        st.stop()

    context = retrieve(user_input)

    history_text = ""
    for h in st.session_state.history[-5:]:
        history_text += f"User: {h['user']}\nAssistant: {h['bot']}\n"

    prompt = f"""
You are a professional business assistant for Rishikirti Technologies.

Keep answers short, clear, and helpful.

Context:
{context}

Conversation:
{history_text}

User: {user_input}
Assistant:
"""

    try:
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="llama3-8b-8192",   # 🔥 safer model
                messages=[{"role": "user", "content": prompt[:4000]}],  # 🔥 prevent overflow
                temperature=0.7
            )

        bot_reply = response.choices[0].message.content

    except Exception as e:
        st.error(f"Groq Error: {str(e)}")
        st.stop()

    st.session_state.history.append({
        "user": user_input,
        "bot": bot_reply
    })
