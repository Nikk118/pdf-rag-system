# ==========================================
# RAG + Conversational Memory (Single File)
# ==========================================

import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from config import *

# ==========================
# CONFIGURATION
# ==========================



MAX_CONTEXT_LENGTH = 1500
MAX_HISTORY_TURNS = 5
MEMORY_PATH = "vector_store/conversation_memory.pkl"
CHUNKS_FILE_PATH = globals().get("CHUNKS_PATH", "vector_store/chunks.pkl")

# Force HuggingFace cache to D drive
os.environ["HF_HOME"] = "D:/huggingface_cache"



def load_memory(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "rb") as f:
            memory = pickle.load(f)
        if isinstance(memory, list):
            return memory
    except Exception:
        pass
    return []


def save_memory(path, memory):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(memory, f)


def build_context(chunks_data, retrieved_indices, max_context_length):
    selected = []
    total_len = 0
    for idx in retrieved_indices:
        if idx < 0 or idx >= len(chunks_data):
            continue
        chunk = chunks_data[idx].strip()
        if not chunk:
            continue
        next_len = total_len + len(chunk) + (1 if selected else 0)
        if next_len > max_context_length:
            break
        selected.append(chunk)
        total_len = next_len
    return "\n".join(selected)


def build_history(memory, max_turns):
    lines = []
    for turn in memory[-max_turns:]:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
        lines.append("")
    return "\n".join(lines).strip()


def extract_answer(generated_text):
    marker = "Answer:"
    if marker in generated_text:
        return generated_text.rsplit(marker, 1)[-1].strip()
    return generated_text.strip()


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

with open(CHUNKS_FILE_PATH, "rb") as f:
    chunks = pickle.load(f)

conversation_history = load_memory(MEMORY_PATH)
if len(conversation_history) > MAX_HISTORY_TURNS:
    conversation_history = conversation_history[-MAX_HISTORY_TURNS:]

print("\nRAG system ready.")



while True:

    query = input("\nAsk question (type 'exit'): ")

    if query.lower() == "exit":
        break



    query_embedding = np.asarray(
        embed_model.encode([query]),
        dtype=np.float32
    )

    _, indices = index.search(query_embedding, TOP_K)
    context = build_context(chunks, indices[0], MAX_CONTEXT_LENGTH)
    history_text = build_history(conversation_history, MAX_HISTORY_TURNS)



    prompt = f"""
You are a helpful AI assistant.

Use conversation history and context to answer.

RULES:
- Answer using ONLY the context.
- If answer not in context, say "I don't know".
- Keep answer clear and structured.
- Finish cleanly.

Conversation History:
{history_text}

Context:
{context}

User Question: {query}

Answer:
"""



    result = generator(
        prompt,
        max_new_tokens=500,
        do_sample=False,
        eos_token_id=generator.tokenizer.eos_token_id
    )

    full_output = result[0]["generated_text"]
    answer = extract_answer(full_output)

   

    conversation_history.append({
        "question": query,
        "answer": answer
    })

 
    if len(conversation_history) > MAX_HISTORY_TURNS:
        conversation_history = conversation_history[-MAX_HISTORY_TURNS:]

    save_memory(MEMORY_PATH, conversation_history)


    print("\n====================")
    print("Answer:")
    print("====================\n")
    print(answer)
