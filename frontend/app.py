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
# Load Knowledge
# -----------------------
@st.cache_resource
def load_vector_store():
    with open("knowledge.txt", "r") as f:
        text = f.read()

    chunks = text.split("\n\n")
    embeddings = embed_model.encode(chunks)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    return index, chunks

index, chunks = load_vector_store()

# -----------------------
# Memory
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# UI
# -----------------------
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
    context = retrieve(user_input)

    history_text = ""
    for h in st.session_state.history[-5:]:
        history_text += f"User: {h['user']}\nAssistant: {h['bot']}\n"

    prompt = f"""
You are a professional business assistant for Rishikirti Technologies.

Context:
{context}

Conversation:
{history_text}

User: {user_input}
Assistant:
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    bot_reply = response.choices[0].message.content

    st.session_state.history.append({
        "user": user_input,
        "bot": bot_reply
    })

# -----------------------
# Display Chat
# -----------------------
for h in st.session_state.history:
    st.chat_message("user").write(h["user"])
    st.chat_message("assistant").write(h["bot"])
