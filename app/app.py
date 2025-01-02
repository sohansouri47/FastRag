from fastapi import FastAPI, Path
import os
from typing import Optional
from pydantic import BaseModel
from io import BytesIO
from langchain.schema import Document 
import google.generativeai as genai
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from elasticsearch import Elasticsearch
from langchain_community.vectorstores import ElasticsearchStore
from elasticsearch import Elasticsearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import ElasticsearchStore
from elasticsearch import Elasticsearch, helpers
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community import chains
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from fastapi import File,UploadFile
from dotenv import load_dotenv
from dotenv import load_dotenv
import fitz
load_dotenv()

#Keys and Env Variables
groq_api_key=os.environ["GROQ_API_KEY"]
elastic_api_key=os.environ["ELASTIC_API_KEY"]
elastic_link=os.environ["ELASTIC_LINK"]
Link="https://langchain-ai.github.io/langgraph/tutorials/introduction/"
es_cloud_id=os.environ["es_cloud_id"]
GOOGLE_API_KEY=os.environ["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
# Initialize FastAPI app
app = FastAPI()
# Pydantic models for FastAPI
class interaction(BaseModel):
    ques:str


res=[]
@app.post("/QA")
def post_query(chat: interaction):
    try:
        llm=ChatGroq(groq_api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile")
        
        
        response=llm.invoke(chat.ques)
        print(llm.invoke("how can u help me?"))
        return response.content
    except Exception as e:
        return {"error": "Error", "details": str(e)}
    

@app.post("/uploadfiles/")
async def create_upload_files(file: list[UploadFile]):
    text = ""
    for files in file:
        res=await files.read()
        doc = fitz.open("pdf", res)  # Open the PDF from byte stream
        text=text+"### New File ###"
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()  
        
    
    document = Document(page_content=text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([document])
    embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}

    )
   
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
    client = Elasticsearch(
    elastic_link,
    api_key=elastic_api_key,
    )
    elastic_index_name="stock_docs2"

    if not client.indices.exists(index=elastic_index_name):
        client.indices.create(index=elastic_index_name)

    es = ElasticsearchStore.from_documents(
    chunks,
    es_cloud_id="<ID>",
    es_api_key=elastic_api_key,
    index_name=elastic_index_name,
    embedding=embeddings,
)
    return "The Upload to Elastic Works"
    

    
    
@app.post("/query/")
async def query_documents(query: str):
    elastic_index_name="stock_docs2"
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
    embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}

    )
    # Initialize retriever with ElasticsearchStore
    es = ElasticsearchStore(
        es_cloud_id=es_cloud_id,
        es_api_key=elastic_api_key,
        index_name=elastic_index_name,
        embedding=embeddings,
    )
    
    llm=ChatGroq(groq_api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile")
    # llm = ChatGoogleGenerativeAI(model="gemini-pro",
    #                          google_api_key=GOOGLE_API_KEY,)
    # Create a retriever to search the documents based on the query
    retriever = es.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant documents

    # Create a simple document chain with the retriever and LLM
    document_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_template("""
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        </context>
        Question: {input}
    """))

    # Create a retrieval chain combining the retriever and document chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Get the response from the retrieval chain
    response = retrieval_chain.invoke({"input": query})
    # print(response)
    return response['answer']

