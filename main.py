import os
import uuid
import dotenv
from typing import List, Dict, Any

from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent

from langgraph.checkpoint.memory import MemorySaver

from transformers import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI

logging.set_verbosity_error()

dotenv.load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPEN_API_URL", "https://inference.do-ai.run/v1")

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-sael")

# %%
model = init_chat_model(
    model=os.getenv("OPEN_MODEL", "openai-gpt-oss-120b"),
    model_provider="openai",
    base_url=os.getenv("OPEN_API_URL", "https://inference.do-ai.run/v1")
)


# %%
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
    "Responda sempre em português, mesmo que a pergunta seja feita em outro idioma."
)

checkpointer = MemorySaver()

agent = create_agent(
    model, 
    tools, 
    system_prompt=prompt,
    checkpointer=checkpointer
)

test_queries = [
    "O que é lógica proposicional segundo a apostila?",
    "Como a apostila define uma proposição?",
    "O que são conectivos lógicos e quais são apresentados no material?",
    "O que é uma tabela-verdade e para que ela é utilizada?",
    "Como a apostila define tautologia, contradição e contingência?"
]

ground_truths = [
    "Lógica proposicional é o ramo da lógica que estuda proposições e as relações entre elas por meio de conectivos lógicos.",
    "Proposição é toda sentença declarativa que pode ser classificada como verdadeira ou falsa, mas não ambas.",
    "Conectivos lógicos são operadores que conectam proposições, como negação (¬), conjunção (∧), disjunção (∨), condicional (→) e bicondicional (↔).",
    "Tabela-verdade é um método utilizado para determinar o valor lógico de proposições compostas a partir dos valores lógicos das proposições simples.",
    "Tautologia é uma proposição composta que é sempre verdadeira; contradição é sempre falsa; contingência é aquela que pode ser verdadeira ou falsa dependendo dos valores das proposições componentes."
]

def run_agent_and_collect_data(query: str, ground_truth: str) -> Dict[str, Any]:
    thread_id = str(uuid.uuid4())
    
    events = list(agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="values",
    ))
    
    final_event = events[-1]
    answer = final_event["messages"][-1].content

    retrieved_docs = vector_store.similarity_search(query, k=2)
    contexts = [doc.page_content for doc in retrieved_docs]
    
    return {
        "question": query,
        "contexts": contexts,
        "answer": answer,
        "ground_truth": ground_truth
    }

def evaluate_with_ragas():
    """Executa avaliação completa com RAGAS."""
    print("Executando agent para coletar dados de teste...")
    ragas_data = []
    
    for i, query in enumerate(test_queries):
        print(f"Testando: {query}")
        data_point = run_agent_and_collect_data(query, ground_truths[i])
        ragas_data.append(data_point)
    
    test_dataset = Dataset.from_list(ragas_data)
    
    print("\nExecutando avaliação RAGAS...")

    eval_llm = ChatOpenAI(
        model=os.getenv("OPEN_MODEL", "openai-gpt-oss-120b"),
        base_url=os.getenv("OPEN_API_URL", "https://inference.do-ai.run/v1"),
        temperature=0
    )
    
    result = evaluate(
        test_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=eval_llm,
        embeddings=embeddings 
    )
    
    print("\n=== RESULTADOS RAGAS ===")
    print(result)
    
    df = result.to_pandas()
    print("\nDetalhes por query:")
    print(df)
    
    return result

if __name__ == "__main__":
    thread_id = "sael"
    query = (
        "O que é array em programação lógica?\n\n"
        "Depois que obtiver a resposta, pesquise extensões comuns desse método."
    )
    
    print("=== EXECUÇÃO NORMAL DO AGENT ===")
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()
    
    
    print("\n\n=== AVALIAÇÃO RAGAS ===")
    try:
        evaluate_with_ragas()
    except Exception as e:
        print(f"Erro na avaliação RAGAS: {e}")
        print("Verifique se tem OPENAI_API_KEY válido e ragas instalado.")