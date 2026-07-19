import streamlit as st
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI



load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

print(api_key)



load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print(api_key)

loader = PyPDFLoader("data/AI ( krishna ).pdf")
documents = loader.load()

print("PDF Loaded Successfully!")
print("Total Pages:", len(documents))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = text_splitter.split_documents(documents)

print("Total Chunks:", len(documents))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = text_splitter.split_documents(documents)

print("Total Chunks:", len(documents))


from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.from_documents(documents, embeddings)

vector_db.save_local("db")

print("FAISS Database Created Successfully!")




llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

query = input("Enter your question: ")

docs = retriever.invoke(query)

print("\nRelevant Chunks:\n")

for doc in docs:
    print(doc.page_content)
    print("-" * 50)