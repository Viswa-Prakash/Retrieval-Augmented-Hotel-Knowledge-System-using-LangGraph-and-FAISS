# 🏨 Retrieval-Augmented Hotel Knowledge System (RAG with LangGraph + FAISS)

This project implements a **hotel knowledge assistant** powered by **LangGraph** and **FAISS** for efficient retrieval from large hotel documentation (FAQs, amenities, policies).  
It exposes a **FastAPI backend** with both **POST** and **GET** endpoints for querying the assistant.

---

## 📌 Features
- ✅ Load hotel knowledge base from **PDF documents** (FAQs, amenities, policies).  
- ✅ Uses **LangChain DirectoryLoader** for fast PDF ingestion.  
- ✅ Splits documents using **RecursiveCharacterTextSplitter**.  
- ✅ Embeds documents with **HuggingFace sentence-transformers**.  
- ✅ Stores embeddings in **FAISS** vector store for fast retrieval.  
- ✅ Retrieval-Augmented Generation (RAG) workflow managed with **LangGraph**.  
- ✅ REST API via **FastAPI** (`/chat` endpoint for queries).  
- ✅ Supports both `POST` (JSON body) and `GET` (query parameter) requests.  

---

---
## ⚙️ Installation
```bash

### 1. Clone the repo
```bash
git clone https://github.com/Viswa-Prakash/Retrieval-Augmented-Hotel-Knowledge-System-using-LangGraph-and-FAISS.git
cd Retrieval-Augmented-Hotel-Knowledge-System-using-LangGraph-and-FAISS


### 2. Create and activate virtual environment
```bash
python -m venv hotelzify
hotelzify\Scripts\activate      # (Windows)



### 3. Install dependencies
```bash
pip install -r requirements.txt

---

---
### Running the App

### 1. Start FastAPI server
```bash
uvicorn app:app --reload

### 2. Access Endpoints

Swagger UI: http://127.0.0.1:8000/docs

---