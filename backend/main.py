from fastapi import FastAPI
from pydantic import BaseModel
from services.retriever import retrieve
from services.memory import get_history, update_history
from services.prompt_builder import build_prompt
from services.llm import generate

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/chat")
def chat(req: ChatRequest):
    context = retrieve(req.message)
    history = get_history(req.session_id)

    prompt = build_prompt(req.message, context, history)

    response = generate(prompt)

    update_history(req.session_id, req.message, response)

    return {"reply": response}
