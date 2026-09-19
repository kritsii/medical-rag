from pathlib import Path
import json
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

EMBEDDINGS_DIR = Path("embeddings")
METADATA_DIR = Path("metadata")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not found. Please set PINECONE_API_KEY in your environment variables.")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("diabetes-rag")

def upload_to_pinecone():
    """Upload embeddings to Pinecone in batches"""
    embed_files = list(EMBEDDINGS_DIR.glob("*_embeddings.json"))
    print(f"Processing {len(embed_files)} files\n")
    
    vectors = []
    batch_size = 100
    batch_count = 0
    
    for i, embed_file in enumerate(embed_files):
        pmid = embed_file.stem.replace("_embeddings", "")
        print(f"[{i+1}/{len(embed_files)}] {pmid}", end=" ")
        
        with open(embed_file, "r") as f:
            embeddings_data = json.load(f)
        
        meta_file = f"{METADATA_DIR}/{pmid}_metadata.json"
        metadata = {}
        if Path(meta_file).exists():
            with open(meta_file, "r") as f:
                metadata = json.load(f)
        
        for item in embeddings_data:
            vectors.append((
                item["id"],
                item["embedding"],
                {
                    "pmid": pmid,
                    "chunk_idx": item["chunk_idx"],
                    "text": item["text"][:500],  # Limit text size
                    "title": metadata.get("title", "")[:100]
                }
            ))
        
        # Upsert every 100 vectors
        if len(vectors) >= batch_size:
            print(f"→ Batch {batch_count+1}")
            index.upsert(vectors=vectors)
            batch_count += 1
            vectors = []
        else:
            print()
    
    # Upsert remaining vectors
    if vectors:
        print(f"→ Batch {batch_count+1} (final)")
        index.upsert(vectors=vectors)
    
    print("\n✓ Done")

if __name__ == "__main__":
    upload_to_pinecone()