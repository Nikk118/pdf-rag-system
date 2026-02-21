# ================================
# query.py — Production RAG Query (Improved Complete Answers)
# ================================

import os

# Force Hugging Face to use D: drive cache
os.environ["HF_HOME"] = "D:/huggingface_cache"

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from config import *

print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL)


print("Loading LLM...")
generator = pipeline(
    "text-generation",
    model=LLM_MODEL,
    device_map="auto",
    torch_dtype="auto"
)


print("Loading vector database...")
index = faiss.read_index(VECTOR_DB_PATH)

with open("vector_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print("\nRAG system ready.")


# Limit context size to prevent overload
MAX_CONTEXT_LENGTH = 1500


while True:

    query = input("\nAsk question (type 'exit'): ")

    if query.lower() == "exit":
        break

    # Create embedding
    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # Search vector DB
    distances, indices = index.search(query_embedding, TOP_K)

    # Build controlled context
    context = ""

    for idx in indices[0]:
        chunk = chunks[idx]

        if len(context) + len(chunk) > MAX_CONTEXT_LENGTH:
            break

        context += chunk + "\n"


    # Improved prompt forcing complete answer
    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.

INSTRUCTIONS:
- Give a clear and complete explanation.
- Use simple language.
- Structure your answer properly.
- Finish your answer cleanly.
- Do NOT include unrelated sentences.
- Do NOT jump between topics.
- If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}

Answer in clear paragraph form:
"""

    # Generate response with proper length
    result = generator(
        prompt,
        max_new_tokens=500,   # increased for complete answers
        do_sample=False,
        eos_token_id=generator.tokenizer.eos_token_id
    )


    full_output = result[0]["generated_text"]

    answer = full_output.split("Answer:")[-1].strip()


    print("\n====================")
    print("Answer:")
    print("====================\n")

    print(answer)