import streamlit as st
from huggingface_hub import login
import pandas as pd
import re
from dotenv import load_dotenv
import os
from datasets import load_dataset
from langchain_community.vectorstores.pgvector import PGVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import sqlalchemy
import logging
import random

# Set seed for reproducibility
random.seed(42)

logging.getLogger("streamlit").setLevel(logging.ERROR)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")
login(hf_token)

def clean_text(text):
    if pd.isnull(text):
        return ""
    return re.sub(r"[\r\n\t\xa0]+", " ", str(text)).strip()

def check_collection_exists(connection_string, collection_name):
    try:
        engine = sqlalchemy.create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(
                sqlalchemy.text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :table_name)"
                ),
                {"table_name": f"langchain_pg_collection"}
            ).scalar()
            if result:
                result = conn.execute(
                    sqlalchemy.text(
                        "SELECT 1 FROM langchain_pg_collection WHERE name = :name LIMIT 1"
                    ),
                    {"name": collection_name}
                ).scalar()
                return result is not None
            return False
    except Exception as e:
        st.error(f"Error checking collection: {str(e)}")
        return False

@st.cache_resource
def load_vector_store():
    ds = load_dataset("MakTek/Customer_support_faqs_dataset")
    df = ds['train'].to_pandas()
    
    for col in ['question', 'answer']:
        df[col] = df[col].apply(clean_text)
    
    docs = [
        {"content": f"Q: {q}\nA: {a}"}
        for q, a in zip(df['question'], df['answer'])
    ]
    
    docs = [Document(page_content=d['content']) for d in docs]
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    connection_string = os.getenv(
        "POSTGRES_URL_NON_POOLING",
        "postgresql+psycopg2://postgres.kuzwbhqbxmibamfckabl:umgCXtTHVEXYDHUu@aws-0-us-east-1.pooler.supabase.com:5432/postgres?sslmode=require"
    )
    
    try:
        vectorstore = PGVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name="documents",
            distance_strategy="cosine"  # Menentukan strategi jarak untuk retriever
        )
        
        if not check_collection_exists(connection_string, "documents"):
            vectorstore.add_documents(docs)
        
        return vectorstore
    except AttributeError as e:
        st.error(f"Error initializing PGVector: {str(e)}")
        return None

@st.cache_resource
def setup_qa_chain(_vectorstore):
    if _vectorstore is None:
        st.error("Vector store initialization failed.")
        return None
    
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192",
        temperature=0,
        model_kwargs={"seed": 42}  # Menambahkan seed untuk konsistensi
    )
    
    prompt = PromptTemplate(
        template="""You are a helpful customer service assistant. You MUST use only the information provided in the following CONTEXT to answer the customer's question accurately. Do NOT add information that is not present in the CONTEXT.

If the context contains relevant information, provide a concise and accurate answer based on it.
If the context does not contain relevant information or you are unsure, respond with: 'I'm sorry, I don't have enough information to answer your question.'

CONTEXT:
{context}

CUSTOMER QUESTION: {question}

ASSISTANT RESPONSE:""",
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 3}),  # Mengambil 3 dokumen paling relevan
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def create_prompt_with_history(history, current_question):
    history_text = "\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in history])
    return f"""You are a helpful customer service assistant. You MUST use only the information provided in the following CONTEXT to answer the customer's question accurately. Do NOT add information that is not present in the CONTEXT.

If the context contains relevant information, provide a concise and accurate answer based on it.
If the context does not contain relevant information or you are unsure, respond with: 'I'm sorry, I don't have enough information to answer your question.'

CONTEXT:
{history_text}

CUSTOMER QUESTION: {current_question}

ASSISTANT RESPONSE:"""

st.title("Customer Support Chatbot")

vectorstore = load_vector_store()
qa_chain = setup_qa_chain(vectorstore)

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Simpan riwayat ke file untuk menjaga konsistensi antar sesi
if os.path.exists("session_history.json"):
    import json
    with open("session_history.json", "r") as f:
        st.session_state.conversation_history = json.load(f)
    st.session_state.messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": msg} 
                                for i, msg in enumerate(sum([[h['question'], h['answer']] for h in st.session_state.conversation_history], []))]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    prompt_with_history = create_prompt_with_history(st.session_state.conversation_history, prompt)
    
    with st.spinner("Thinking..."):
        if qa_chain is None:
            st.error("QA chain is not initialized. Please check the vector store setup.")
        else:
            result = qa_chain.invoke({"query": prompt_with_history})
            answer = result["result"]
        
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
            
            st.session_state.conversation_history.append({"question": prompt, "answer": answer})
            # Simpan riwayat ke file
            with open("session_history.json", "w") as f:
                json.dump(st.session_state.conversation_history, f)
