from pinecone_tool import query_pinecone

def retrieve(user_query: str) -> list:
    """Pure retrieval — returns list of chunks with metadata"""
    results = query_pinecone(user_query)
    return results

if __name__ == "__main__":
    results = retrieve("Type 1 vs Type 2 diabetes")
    for r in results:
        print(f"Title: {r['title']}")
        print(f"DOI: {r.get('doi', 'N/A')}")
        print(f"Score: {r['score']:.2f}")
        print(f"Excerpt: {r['text'][:200]}")
        print("---")