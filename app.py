import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import textwrap
import os
import tempfile

# Helper function to render Markdown content
def render_markdown(text):
    st.markdown(text)

# Initialize Streamlit app
st.title("PDF Chatbot")

# Function to load and process PDF
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_filename = temp_file.name
    loader = PyPDFLoader(temp_filename)
    pages = loader.load_and_split()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(pages, embeddings)
    os.remove(temp_filename)  # Remove temporary file
    return db

# Function to interact with Google Generative AI
def chat_with_ai(query, docs):
    GOOGLE_API_KEY='AIzaSyC1tM6uFpBohcLW8zuJF2s-6cZSwftP7zY'
    genai.configure(api_key=GOOGLE_API_KEY)
    content = "\n".join([x.page_content for x in docs])
    qa_prompt = "Use the following pieces of context to answer the user's question.----------------"
    input_text = qa_prompt + "\nContext:" + content + "\nUser question:\n" + query
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    result = llm.invoke(input_text)
    return result.content

# Main Streamlit UI
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    st.write("PDF successfully uploaded.")

    # Process PDF
    db = process_pdf(uploaded_file)

    # Initialize conversation
    st.markdown("AI: Hi there! What can I help you with today?")
    st.text("")  # Empty line

    while True:
        question_count = 0
        # User input
        user_input = st.text_input("You:", key=question_count)

        if st.button("Ask") or user_input:
            # Search for similar documents
            docs = db.similarity_search(user_input)

            # Chat with AI
            response = chat_with_ai(user_input, docs)

            # Display conversation
            st.text("")  # Empty line
            st.markdown(f"You: {user_input}")
            st.markdown(f"AI: {response}")

            # Display relevant document paragraphs
            st.markdown("Relevant document paragraphs:")
            for doc in docs:
                st.text(doc.page_content)
        question_count += 1
else:
    st.write("Please upload a PDF file.")
