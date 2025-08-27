import os
from dotenv import load_dotenv
load_dotenv()

from typing_extensions import TypedDict
from typing import List, Dict, Any, TypedDict
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, START, END

# Load all PDFs from a directory
def load_documents(data_path: str):
    loader = DirectoryLoader(
        data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    docs = loader.load()
    return docs


# Split documents into chunks
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    return text_splitter.split_documents(docs)


# Build FAISS vectorstore
def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# State for LangGraph
class RAGState(TypedDict, total=False):
    """
    State object passed between LangGraph nodes.
    - messages: conversation history (user + assistant turns)
    - context: retrieved knowledge base text for the current query
    """
    messages: List[Dict[str, Any]]
    context: str


# Nodes 
def retrieve_node(state: RAGState, vectorstore: FAISS) -> RAGState:
    """
    Retrieve top-k relevant documents for the latest user query
    and update the state with concatenated context.
    """
    query = state["messages"][-1]["content"]
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    return {**state, "context": context}


def generate_node(state: RAGState, llm: BaseChatModel) -> RAGState:
    """
    Generate a response using the LLM and retrieved context.
    """
    query = state["messages"][-1]["content"]
    context = state.get("context", "")

    prompt = f"""
    You are a helpful hotel assistant.
    Use the following context from hotel FAQs, policies, and amenities to answer the user query.
    If the answer is not available in the context, politely say you don’t know.

    Context:
    {context}

    Query: {query}
    """

    response = llm.invoke(prompt)

    updated_messages = state["messages"] + [
        {"role": "assistant", "content": response.content}
    ]

    return {**state, "messages": updated_messages}


# Build Graph


def build_graph(vectorstore):
    # Using OpenAI LLM here, can replace with HuggingFace pipeline if needed
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", lambda state: retrieve_node(state, vectorstore))
    graph.add_node("generate", lambda state: generate_node(state, llm))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# Prepare Pipeline
def init_rag_pipeline(data_path="./data"):
    docs = load_documents(data_path)
    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    app = build_graph(vectorstore)
    return app