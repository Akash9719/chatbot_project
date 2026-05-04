import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("backend/vector_store/index.pkl", "rb") as f:
    index, chunks = pickle.load(f)

def retrieve(query, k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, k)

    results = [chunks[i] for i in indices[0]]
    return "\n".join(results)