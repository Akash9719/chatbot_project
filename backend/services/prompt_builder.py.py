SYSTEM_PROMPT = """
You are a professional AI assistant for Rishikirti Technologies.

Act like a business consultant.
Encourage users to share requirements.
Be clear, structured, and helpful.
"""

def build_prompt(user_input, context, history):
    history_text = ""

    for h in history[-5:]:
        history_text += f"User: {h['user']}\nAssistant: {h['bot']}\n"

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Conversation:
{history_text}

User: {user_input}
Assistant:
"""
    return prompt