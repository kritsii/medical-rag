from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a medical research assistant. Answer using only the provided research papers. If they do not contain enough information, say so."),
    ("human", """Question: {query}

Research papers:
{context}

Answer clearly and concisely. Cite sources inline using [1], [2], etc. matching the paper numbers. End with a CITATIONS section listing each cited paper's title and DOI.

Use this format:
ANSWER:
<answer with inline citations>

CITATIONS:
[1] Title | DOI: doi_here""")
])

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
    temperature=0.2,
    request_timeout=60,
    api_key=groq_api_key
)

chain = prompt | llm | StrOutputParser()

def format_chunks_for_prompt(chunks: list) -> str:
    """Format retrieved chunks into prompt context"""
    formatted = ""
    for i, chunk in enumerate(chunks):
        formatted += f"""
[{i+1}] Title: {chunk['title']}
DOI: {chunk.get('doi', 'N/A')}
Excerpt: {chunk['text'][:300]}
---"""
    return formatted

def synthesize(query: str, chunks: list) -> dict:
    """Generate answer from retrieved chunks with citations"""
    
    context = format_chunks_for_prompt(chunks)
    
    response = chain.invoke({"query": query, "context": context})
    return parse_response(response, chunks)

def parse_response(raw: str, chunks: list) -> dict:
    """Parse LLM response into structured output"""
    
    answer = ""
    citations = []
    
    if "ANSWER:" in raw and "CITATIONS:" in raw:
        parts = raw.split("CITATIONS:", maxsplit=1)
        answer = parts[0].replace("ANSWER:", "").strip()
        citation_block = parts[1].strip()
        
        for line in citation_block.split("\n"):
            line = line.strip()
            if line and line[0] == "[":
                citations.append(line)
    else:
        answer = raw
    
    return {
        "answer": answer,
        "citations": citations,
        "raw_chunks": chunks
    }

if __name__ == "__main__":
    # Test with dummy chunks
    dummy_chunks = [
        {
            "title": "Type 1 Diabetes Pathophysiology",
            "doi": "10.1234/test1",
            "text": "Type 1 diabetes is an autoimmune condition where the immune system destroys insulin-producing beta cells in the pancreas.",
            "score": 0.95
        },
        {
            "title": "Type 2 Diabetes Mechanisms",
            "doi": "10.1234/test2",
            "text": "Type 2 diabetes is characterized by insulin resistance and relative insulin deficiency, often linked to obesity and lifestyle factors.",
            "score": 0.91
        }
    ]
    
    result = synthesize("What is the difference between Type 1 and Type 2 diabetes?", dummy_chunks)
    
    print("ANSWER:")
    print(result['answer'])
    print("\nCITATIONS:")
    for c in result['citations']:
        print(c)