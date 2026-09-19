import os
from pathlib import Path
import json
import tiktoken

BASE_DIR = Path(__file__).resolve().parent
EXTRACTED_DIR = BASE_DIR / "extracted_text"
CHUNKS_DIR = BASE_DIR / "chunks"
CHUNKS_DIR.mkdir(exist_ok=True)

enc = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with overlap"""
    tokens = enc.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks

def process_texts():
    """Chunk all extracted texts"""
    texts = list(EXTRACTED_DIR.glob("*_text.txt"))
    print(f"Reading from: {EXTRACTED_DIR}")
    print(f"Found {len(texts)} text files\n")
    
    for i, text_path in enumerate(texts):
        pmid = text_path.stem.replace("_text", "")
        print(f"[{i+1}/{len(texts)}] {pmid}", end=" ")
        
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        chunks = chunk_text(text)
        
        chunks_data = {
            "pmid": pmid,
            "num_chunks": len(chunks),
            "chunks": chunks
        }
        
        with open(f"{CHUNKS_DIR}/{pmid}_chunks.json", "w") as f:
            json.dump(chunks_data, f)
        
        print(f"→ {len(chunks)} chunks")

if __name__ == "__main__":
    process_texts()
    print("\n✓ Done")