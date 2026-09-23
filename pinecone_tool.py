from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is not set in .env.")
if not pinecone_index_name:
    raise ValueError("PINECONE_INDEX_NAME is not set in .env.")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

def load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers could not load. Use Python 3.11 or 3.12, "
            "then reinstall its dependencies in that environment."
        ) from exc
    return SentenceTransformer("all-MiniLM-L6-v2")

def query_pinecone(query_text: str, top_k: int = 5):
    """Query Pinecone, return chunks with metadata"""
    model = load_embedding_model()
    query_embedding = model.encode(query_text).tolist()
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    chunks = []
    for match in results['matches']:
        chunks.append({
            'text': match['metadata'].get('text', ''),
            'title': match['metadata'].get('title', 'N/A'),
            'score': match['score']
        })
    
    return chunks

# Test
if __name__ == "__main__":
    results = query_pinecone("Type 1 diabetes")
    for r in results:
        print(f"Title: {r['title']}, Score: {r['score']:.2f}")