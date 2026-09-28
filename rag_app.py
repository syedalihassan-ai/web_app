"""
Super Simple RAG Chatbot
--------------------------
This app lets you:
 1. Paste a web page URL
 2. Click a button to "learn" that page
 3. Ask questions about it and get answers from an AI

Run with:
    pip install streamlit langchain langchain-core langchain-groq langchain-huggingface \
                langchain-text-splitters langchain-community langchain-classic \
                faiss-cpu sentence-transformers bs4

    streamlit run rag_app_super_simple.py
"""

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


st.title("🔎 Simple RAG Chatbot")

# Streamlit reruns this whole file every time you click something.
# So we save the chatbot and chat history in st.session_state to not lose them.
if "chain" not in st.session_state:
    st.session_state.chain = None
if "history" not in st.session_state:
    st.session_state.history = []


# ---------- Step 1: Get inputs from the user ----------
api_key = st.text_input("Enter your Groq API Key", type="password")
url = st.text_input("Enter a web page URL", value="https://en.wikipedia.org/wiki/Lahore")

# ---------- Step 2: Build the chatbot when the button is clicked ----------
if st.button("Load Page"):

    # Download the web page
    loader = WebBaseLoader(url)
    documents = loader.load()

    # Break the page into small chunks of text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Turn the chunks into embeddings (numbers that represent meaning)
    # and store them in a searchable database (FAISS)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Set up the AI model
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=api_key)

    # Tell the AI to answer using only the retrieved text
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question using this context:\n\n{context}"),
        ("human", "{input}"),
    ])
    combine_chain = create_stuff_documents_chain(llm, prompt)

    # Save the finished chatbot so we can use it below
    st.session_state.chain = create_retrieval_chain(retriever, combine_chain)
    st.session_state.history = []
    st.success("Page loaded! You can ask questions now.")


# ---------- Step 3: Show the chat ----------
for question, answer in st.session_state.history:
    st.write("🧑 " + question)
    st.write("🤖 " + answer)

question = st.text_input("Ask a question about the page")

if st.button("Ask"):
    if st.session_state.chain is None:
        st.warning("Please load a page first.")
    else:
        result = st.session_state.chain.invoke({"input": question})
        answer = result["answer"]

        st.session_state.history.append((question, answer))
        st.write("🧑 " + question)
        st.write("🤖 " + answer)