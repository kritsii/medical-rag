import os
from pinecone import Pinecone
from fastembed import TextEmbedding
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not found. Please set PINECONE_API_KEY in your environment variables.")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("diabetes-rag")
model = TextEmbedding("BAAI/bge-small-en-v1.5")

def test_retrieval():
    """Test queries"""
    queries = [
        "diabetes management treatment",
        "insulin therapy diabetes",
        "blood glucose control",
        "type 2 diabetes diagnosis",
        "diabetes complications"
    ]
    
    print("Testing retrieval...\n")
    
    for query in queries:
        query_embedding = next(model.embed([query])).tolist()
        results = index.query(vector=query_embedding, top_k=3, include_metadata=True)
        
        print(f"Query: '{query}'")
        for i, match in enumerate(results["matches"]):
            title = match['metadata']['title'][:60]
            print(f"  {i+1}. {title}... (score: {match['score']:.3f})")
        print()

if __name__ == "__main__":
    test_retrieval()