import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import pickle
import os
from config import *


# load embedding
model=SentenceTransformer(EMBED_MODEL)

# load pdf
pdf_path=input("Enter PDF file path:")

if not os.path.exists(pdf_path):
    print("file not found!")
    exit()

reader=PdfReader(pdf_path)
text=""
for page in reader.pages:
    page_text=page.extract_text()
    if page_text:
        text+=page_text

def chunk_text(text,chunk_size,overlap):
    chunks=[]
    start=0
    while start<len(text):
        end=start+chunk_size
        chunk=text[start:end]
        chunks.append(chunk)
        start=end-overlap
    return chunks

chunks=chunk_text(text,CHUNK_SIZE,CHUNK_OVERLAP)

print(f"craeted {len(chunks)} chunks")

# embedding
embeddings=model.encode(chunks)
embeddings=np.array(embeddings).astype("float32")

#faiss index
dimension=embeddings.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(embeddings)

# save index
os.makedirs("vector_store",exist_ok=True)
faiss.write_index(index,VECTOR_DB_PATH)
with open("vector_store/chunks.pkl","wb") as f:
    pickle.dump(chunks,f)

print("Vector DB saved successfully")
                                      