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

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# Login to Hugging Face
login(hf_token)

# Function to clean text
def clean_text(text):
    if pd.isnull(text):
        return ""
    return re.sub(r"[\r\n\t\xa0]+", " ", str(text)).strip()

# Load and process dataset
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
    
    # Gunakan connection string dengan skema postgresql+psycopg2
    connection_string = os.getenv(
        "POSTGRES_URL_NON_POOLING",
        "postgresql+psycopg2://postgres.kuzwbhqbxmibamfckabl:umgCXtTHVEXYDHUu@aws-0-us-east-1.pooler.supabase.com:5432/postgres?sslmode=require"
    )
    
    vectorstore = PGVector(
        connection_string=connection_string,
        embedding_function=embeddings,
        collection_name="documents"
    )
    
    # Cek apakah koleksi sudah ada untuk mencegah duplikasi
    if not vectorstore.collection_exists(collection_name="documents"):
        vectorstore.add_documents(docs)
    
    return vectorstore

# Setup LLM and QA chain
@st.cache_resource
def setup_qa_chain(_vectorstore):
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192"
    )
    
    prompt = PromptTemplate(
        template="""You are a helpful customer service assistant. Use the following CONTEXT to answer the customer's question accurately.

If the context contains relevant information, provide a comprehensive and helpful answer.

If you're not sure or the information isn't available, say so honestly.

CONTEXT:
{context}

CUSTOMER QUESTION: {question}

ASSISTANT RESPONSE:""",
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# Create prompt with history
def create_prompt_with_history(history, current_question):
    history_text = "\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in history])
    return f"""You are a helpful customer service assistant. Use the following CONTEXT to answer the customer's question accurately.

If the context contains relevant information, provide a comprehensive and helpful answer.

If you're not sure or the information isn't available, say so honestly.

CONTEXT:
{history_text}

CUSTOMER QUESTION: {current_question}

ASSISTANT RESPONSE:"""

# Streamlit app
st.title("Customer Support Chatbot")

# Load vector store and QA chain
vectorstore = load_vector_store()
qa_chain = setup_qa_chain(vectorstore)

# Initialize chat history in session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate prompt with history
    prompt_with_history = create_prompt_with_history(st.session_state.conversation_history, prompt)
    
    # Run QA chain
    with st.spinner("Thinking..."):
        result = qa_chain.invoke({"query": prompt_with_history})
        answer = result["result"]
    
    # Add assistant message to chat
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
    
    # Update conversation history

    st.session_state.conversation_history.append({"question": prompt, "answer": answer})

