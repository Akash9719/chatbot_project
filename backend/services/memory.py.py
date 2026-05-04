from collections import defaultdict

memory_store = defaultdict(list)

def get_history(session_id):
    return memory_store[session_id]

def update_history(session_id, user, bot):
    memory_store[session_id].append({
        "user": user,
        "bot": bot
    })