from pathlib import Path
import argparse
import json
import os
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone

load_dotenv()

EMBEDDINGS_DIR = Path("embeddings")
METADATA_DIR = Path("metadata")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not found. Please set PINECONE_API_KEY in your environment variables.")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("diabetes-rag")

def get_groq_client():
    """Create a Groq client using the key from the environment."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Groq API key not found. Please set GROQ_API_KEY in your environment variables.")
    return Groq(api_key=groq_api_key)

def test_groq_completion():
    """Send a minimal request to verify Groq API access."""
    client = get_groq_client()
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Respond with exactly: Groq connection works."}],
        temperature=0,
        max_tokens=32,
    )
    response = completion.choices[0].message.content
    print(response or "Groq returned an empty response.")

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-groq",
        action="store_true",
        help="Test Groq API access with a basic completion instead of uploading vectors.",
    )
    args = parser.parse_args()

    if args.test_groq:
        test_groq_completion()
    else:
        upload_to_pinecone()
