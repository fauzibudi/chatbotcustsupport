Customer Support Chatbot <br>
Link deployment: https://chatbotcustsupport.streamlit.app/

Description
The Customer Support Chatbot is a Streamlit-based application that uses Retrieval-Augmented Generation (RAG) to answer customer queries based on a FAQ dataset. It leverages a PostgreSQL database with the pgvector extension for storing text embeddings, Hugging Face for embedding generation, and Groq as the language model for generating responses. The dataset used is MakTek/Customer_support_faqs_dataset from Hugging Face.
This project is designed to run both locally and on Streamlit Cloud, with an external Supabase PostgreSQL database for vector storage.

Features
```
1. Chatbot Interface: Users can ask questions through a Streamlit interface.
2. Semantic Search: Uses pgvector to retrieve relevant documents based on embeddings.
3. Conversation History: Maintains a history of questions and answers for better context.
4. Supabase Integration: Stores embeddings in a PostgreSQL database with the pgvector extension.
5. Language Model: Utilizes Groq to generate natural responses.
```
Prerequisites
```
Python 3.10 or higher
Hugging Face account and API token (HF_TOKEN)
Groq account and API key (GROQ_API_KEY)
Supabase account with a PostgreSQL database supporting the pgvector extension
Streamlit Cloud account for deployment
```
