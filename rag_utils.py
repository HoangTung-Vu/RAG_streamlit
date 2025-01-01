from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

def load_and_split_pdf(file_path, chunk_size=1000):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    return text_splitter.split_documents(data)

def initialize_embedding_model(model="models/text-embedding-004"):
    return GoogleGenerativeAIEmbeddings(model=model)

def get_vector_store(docs=None, embedding_model=None, persist_dir="chroma_db"):
    if os.path.exists(persist_dir):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        return vectorstore

def initialize_llm(model="gemini-1.5-pro", temperature=0.3, max_tokens=5000):
    return ChatGoogleGenerativeAI(
        model=model, 
        temperature=temperature, 
        max_tokens=max_tokens
    )

def create_rag_chain(retriever, llm, system_prompt):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)