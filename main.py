from langchain_core.runnables import RunnableLambda
from retrieval_agent import retrieve
from synthesis_agent import synthesize
from citation import CitationTracker
from dotenv import load_dotenv

load_dotenv()

def retrieve_step(inputs: dict) -> dict:
    query = inputs["query"]
    return {"query": query, "chunks": retrieve(query)}

def synthesize_step(state: dict) -> dict:
    chunks = state["chunks"]
    result = synthesize(state["query"], chunks) if chunks else None
    return {**state, "result": result}

rag_chain = RunnableLambda(retrieve_step) | RunnableLambda(synthesize_step)

def run_pipeline(user_query: str) -> None:
    print(f"\n{'='*50}")
    print(f"Query: {user_query}")
    print(f"{'='*50}")
    
    print("\n[1] Retrieving papers...")
    state = rag_chain.invoke({"query": user_query})
    chunks = state["chunks"]
    
    if not chunks:
        print("No relevant papers found.")
        return
    
    print(f"Found {len(chunks)} chunks")
    
    tracker = CitationTracker()
    for chunk in chunks:
        tracker.add(
            chunk["title"],
            chunk.get("doi", "N/A"),
            chunk["text"]
        )
    
    print("\n[2] Synthesizing answer...")
    result = state["result"]
    
    print("\nANSWER:")
    print(result['answer'])
    
    print(tracker.format())

if __name__ == "__main__":
    run_pipeline("What are risk factors of diabetes?")