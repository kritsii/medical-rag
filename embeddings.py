import os
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_DIR = BASE_DIR / "chunks"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks():
    """Embed all chunks and save vectors"""
    chunk_files = list(CHUNKS_DIR.glob("*_chunks.json"))
    print(f"Reading from: {CHUNKS_DIR}")
    print(f"Found {len(chunk_files)} chunk files\n")
    
    total_chunks = 0
    
    for i, chunk_file in enumerate(chunk_files):
        pmid = chunk_file.stem.replace("_chunks", "")
        print(f"[{i+1}/{len(chunk_files)}] {pmid}", end=" ")
        
        with open(chunk_file, "r") as f:
            data = json.load(f)
        
        chunks = data["chunks"]
        embeddings = model.encode(chunks)
        
        # Prepare for Pinecone
        embeddings_data = []
        for chunk_idx, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            embeddings_data.append({
                "id": f"{pmid}_{chunk_idx}",
                "pmid": pmid,
                "chunk_idx": chunk_idx,
                "text": chunk_text,
                "embedding": embedding.tolist()
            })
        
        # Save
        with open(f"{EMBEDDINGS_DIR}/{pmid}_embeddings.json", "w") as f:
            json.dump(embeddings_data, f)
        
        total_chunks += len(chunks)
        print(f"→ {len(chunks)} chunks embedded")
    
    print(f"\n✓ Total chunks: {total_chunks}")

if __name__ == "__main__":
    embed_chunks()