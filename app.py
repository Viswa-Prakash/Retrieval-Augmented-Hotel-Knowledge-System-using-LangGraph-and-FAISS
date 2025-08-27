from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import init_rag_pipeline

# -------- Init RAG Pipeline --------
rag_app = init_rag_pipeline("./data")

# -------- FastAPI Setup --------
app = FastAPI(title="Hotel RAG Chatbot")

class Query(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "Hotel RAG API is running!"}


@app.post("/chat")
def chat(query: Query):
    state = {"messages": [{"role": "user", "content": query.question}]}
    result = rag_app.invoke(state)
    return {"answer": result["messages"][-1]["content"]}

@app.get("/chat")
def chat_get(question: str = "What is the check-in time?"):
    state = {"messages": [{"role": "user", "content": question}]}
    result = rag_app.invoke(state)
    return {"answer": result["messages"][-1]["content"]}