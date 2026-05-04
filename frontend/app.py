import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000/chat"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("💬 AI Assistant")

user_input = st.text_input("Ask something:")

if user_input:
    res = requests.post(API_URL, json={
        "message": user_input,
        "session_id": st.session_state.session_id
    })
    st.write(res.json()["reply"])
