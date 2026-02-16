import os
import dotenv

from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent

from langgraph.checkpoint.memory import MemorySaver

from transformers import logging

logging.set_verbosity_error()

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPEN_API_URL", "https://inference.do-ai.run/v1")

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-sael")

model = init_chat_model(
    model=os.getenv("OPEN_MODEL", "openai-gpt-oss-120b"),
    model_provider="openai",
    base_url=os.getenv("OPEN_API_URL", "https://inference.do-ai.run/v1")
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = InMemoryVectorStore(embeddings)

loader = DirectoryLoader(
    path="docs",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

all_splits = text_splitter.split_documents(docs)

document_ids = vector_store.add_documents(documents=all_splits)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]

prompt = (
    "You have access to a tool that retrieves context from a book for logic programming. "
    "Use the tool to help answer user queries."
)

checkpointer = MemorySaver()

agent = create_agent(
    model, 
    tools, 
    system_prompt=prompt,
    checkpointer=checkpointer
)

thread_id = "sael"

print("Agent RAG pronto! Digite 'parar' para sair.\n")

while True:
    user_input = input("Você: ")
    if user_input.lower() in ["parar", "sair", "exit", "quit"]:
        print("Aplicação encerrada.")
        break
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": thread_id}}
    )
    
    print(f"\nAgent: {result['messages'][-1].content}\n")
