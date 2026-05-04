import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer

DATA_PATH = "backend/data/knowledge.txt"
VECTOR_PATH = "backend/vector_store/index.pkl"

model = SentenceTransformer('all-MiniLM-L6-v2')

with open(DATA_PATH, "r") as f:
    text = f.read()

chunks = text.split("\n\n")
embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)

os.makedirs("backend/vector_store", exist_ok=True)

with open(VECTOR_PATH, "wb") as f:
    pickle.dump((index, chunks), f)

print("✅ Vector DB created")
