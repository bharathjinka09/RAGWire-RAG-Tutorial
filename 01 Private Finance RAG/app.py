from pathlib import Path
from ragwire import RAGWire

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
DATA_DIR = BASE_DIR / "data"

rag = RAGWire(str(CONFIG_PATH))

# Ingest
stats = rag.ingest_documents(["data/Apple_10k_2025.pdf"])
print(f"Chunks created: {stats['chunks_created']}")

# Retrieve
results = rag.retrieve("What is Apple's total revenue?", top_k=5)

print("Retrieved results:")
for doc in results:
    print(f"[{doc.metadata.get('file_name')}]\n{doc.page_content}\n\n---\n\n")

# RAG Agent end to end
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from ragwire import RAGWire

rag = RAGWire("config.yaml")
rag.ingest_directory("data/")

@tool
def search_documents(query: str) -> str:
    """Search the document knowledge base for relevant information."""
    results = rag.retrieve(query, top_k=5)
    if not results:
        return "No relevant documents found."
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('file_name')}]\n{doc.page_content}"
        for doc in results
    )

agent = create_agent(
    model=ChatOllama(model="qwen3.5:9b"),
    tools=[search_documents],
    system_prompt="You are a helpful document assistant. Use search_documents to answer questions.",
    checkpointer=InMemorySaver(),
)

from langchain.messages import HumanMessage
config = {"configurable": {"thread_id": "session-1"}}
response = agent.invoke(
    {"messages": [HumanMessage("What is the total revenue?")]},
    config=config,
)

print(response["messages"][-1].content)