from pinecone import Pinecone
import os
from functools import lru_cache
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is not set in the environment.")
if not pinecone_index_name:
    raise ValueError("PINECONE_INDEX_NAME is not set in the environment.")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

@lru_cache(maxsize=1)
def load_embedding_model() -> TextEmbedding:
    return TextEmbedding("BAAI/bge-small-en-v1.5")

def query_pinecone(query_text: str, top_k: int = 5):
    """Query Pinecone, return chunks with metadata"""
    model = load_embedding_model()
    query_embedding = next(model.embed([query_text])).tolist()
    
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
            'doi': match['metadata'].get('doi', 'N/A'),
            'pmid': match['metadata'].get('pmid', ''),
            'score': match['score']
        })
    
    return chunks

# Test
if __name__ == "__main__":
    results = query_pinecone("Type 1 diabetes")
    for r in results:
        print(f"Title: {r['title']}, Score: {r['score']:.2f}")