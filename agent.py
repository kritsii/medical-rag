from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from pinecone_tool import query_pinecone
from dotenv import load_dotenv
import os

load_dotenv()

@tool
def query_papers(query_text: str) -> str:
    """Query scientific papers from Pinecone. Input: research query. Returns top 5 papers with DOI and title."""
    results = query_pinecone(query_text)
    if not results:
        return "No results found."
    formatted = "\n".join([
        f"- {r['title']} (DOI: {r['doi']}, Score: {r['score']:.2f})"
        for r in results
    ])
    return formatted

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Bind tools directly to LLM — no AgentExecutor needed
llm_with_tools = llm.bind_tools([query_papers])

def run_agent(user_query: str):
    print(f"\nQuery: {user_query}")
    
    # Step 1: LLM decides whether to call tool
    messages = [{"role": "user", "content": user_query}]
    response = llm_with_tools.invoke(messages)
    
    # Step 2: If tool was called, execute it
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"\nCalling tool: {tool_call['name']} with input: {tool_call['args']}")
            tool_result = query_papers.invoke(tool_call['args'])
            print(f"\nRetrieved Papers:\n{tool_result}")
            
            # Step 3: Send result back to LLM for final answer
            messages.append({"role": "assistant", "content": response.content, "tool_calls": response.tool_calls})
            messages.append({"role": "tool", "content": tool_result, "tool_call_id": tool_call['id']})
            final_response = llm_with_tools.invoke(messages)
            return final_response.content
    else:
        return response.content

if __name__ == "__main__":
    answer = run_agent("What is the difference between Type 1 and Type 2 diabetes?")
    print(f"\nFinal Answer:\n{answer}")