import os
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------
# Setup
# -----------------------
st.set_page_config(page_title="Rishikirti AI Assistant")
st.title("💬 Rishikirti AI Assistant")

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
# Session Memory (CHAT FORMAT)
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------
# Retrieval
# -----------------------
def retrieve(query, k=4):
    q_vec = embed_model.encode([query])
    D, I = index.search(np.array(q_vec), k)
    return "\n".join([chunks[i] for i in I[0]])

# -----------------------
# Display existing chat
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

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Show user message immediately
    with st.chat_message("user"):
        st.write(user_input)

    # Retrieve context
    context = retrieve(user_input)

    # Build prompt
    prompt = f"""
You are a professional business assistant for Rishikirti Technologies.

STRICT RULES:
- ONLY answer using the provided context
- DO NOT make up information
- If answer not in context, say: "Please contact our team for more details"

Context:
{context}

User: {user_input}
Assistant:
"""

    try:
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt[:4000]}],
                temperature=0.7
            )

        bot_reply = response.choices[0].message.content

    except Exception as e:
        st.error(f"Groq Error: {str(e)}")
        st.stop()

    # Add assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    # Show assistant response
    with st.chat_message("assistant"):
        st.write(bot_reply)
