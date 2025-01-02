# FastAPI-Powered LLM & RAG App

### 🚀 **Features**
- **LLM-Powered Q&A**: Direct integration with ChatGroq's Llama model for seamless question answering.
- **Document Ingestion**: Upload PDF files via FastAPI and index them into Elasticsearch for vector-based storage.
- **RAG Implementation**: Query your ingested documents using a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🛠️ **Skills and Technologies Used**
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-005571?style=flat-square&logo=elasticsearch&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-333333?style=flat-square&logo=data:image/png;base64,...)
![ChatGroq](https://img.shields.io/badge/ChatGroq-004080?style=flat-square&logo=data:image/png;base64,...)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD700?style=flat-square&logo=huggingface&logoColor=black)
![Llama](https://img.shields.io/badge/Llama-70b%20Versatile-blueviolet?style=flat-square&logo=data:image/png;base64,...)
![Gemini](https://img.shields.io/badge/Gemini-Pro-orange?style=flat-square&logo=data:image/png;base64,...)

---

## 📚 **Intuition Behind the Project**
This project is designed to:
- Provide a **plug-and-play interface** for using cutting-edge LLMs.
- Combine **structured document ingestion** with the power of semantic search.
- Leverage the strengths of **FastAPI for API development**, **Elasticsearch for vector storage**, and **LangChain for chaining LLMs** into a cohesive pipeline.

---

## 🧩 **Endpoints**

### 1. **Q&A Endpoint**  
`POST /QA`
- **Description**: Directly interacts with ChatGroq's Llama model for basic Q&A.
- **Payload**:
  ```json
  {
    "ques": "What is FastAPI?"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "FastAPI is a modern, fast web framework for building APIs with Python."
  }
  ```

### 2. **Document Ingestion Endpoint**
`POST /uploadfiles/`
- **Description**: Accepts PDF files, processes their content, and indexes them in Elasticsearch as vector embeddings.
- **Steps**:
  1. PDF content is read using `PyMuPDF`.
  2. Text is split into chunks using LangChain's `RecursiveCharacterTextSplitter`.
  3. Embeddings are generated via HuggingFace's BGE-small model.
  4. Indexed in Elasticsearch using `ElasticsearchStore`.
- **Response**: Confirmation of successful ingestion.

### 3. **Query Documents Endpoint**
`POST /query/`
- **Description**: Implements RAG to retrieve relevant document chunks and generate answers using ChatGroq's Llama model.
- **Payload**:
  ```json
  {
    "query": "Explain the main idea of the uploaded document."
  }
  ```
- **Response**:
  ```json
  {
    "answer": "The document discusses modern advancements in AI."
  }
  ```

---

## 🛠️ **Key Implementation Details**

### **Direct LLM Integration**
- ChatGroq's `llama-3.3-70b-versatile` is invoked for fast and versatile responses.
- Example Code:
  ```python
  llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
  response = llm.invoke(chat.ques)
  ```

### **Document Ingestion**
- PDF files are processed using `fitz` (PyMuPDF).
- Chunks are generated with overlap to preserve context.
- Embeddings are normalized for better indexing and retrieval.

### **Retrieval-Augmented Generation (RAG)**
- Combines `ElasticsearchStore` retriever with ChatGroq's LLM.
- Utilizes `LangChain`'s `create_stuff_documents_chain` for accurate and context-aware responses.

---

## 🏗️ **Setup and Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   export GROQ_API_KEY=<your_groq_key>
   export ELASTIC_API_KEY=<your_elastic_key>
   export ELASTIC_LINK=<your_elastic_link>
   export GOOGLE_API_KEY=<your_google_key>
   ```
4. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

---

## 🌟 **Future Enhancements**
- Add support for more file types (e.g., Images, Text).

---

