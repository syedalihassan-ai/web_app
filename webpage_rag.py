# Required packages — run this to install everything:
# pip install langchain langchain-core langchain-groq langchain-huggingface langchain-text-splitters langchain-community faiss-cpu sentence-transformers bs4

# ── Document loading ──────────────────────────────────────────────────────────
from langchain_community.document_loaders import WebBaseLoader

# ── Text splitting ────────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Embeddings ────────────────────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings

# ── Vector store ──────────────────────────────────────────────────────────────
from langchain_community.vectorstores import FAISS

# ── LLM ───────────────────────────────────────────────────────────────────────
from langchain_groq import ChatGroq

# ── Chain helpers ─────────────────────────────────────────────────────────────
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# ── Prompt ────────────────────────────────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate



# Step 1: Load documents from the web
url = "https://en.wikipedia.org/wiki/Lahore"   # ← replace with your actual URL
loader = WebBaseLoader(url)
docs = loader.load()

# Step 2: Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# Step 3: Generate embeddings with HuggingFace Sentence Transformers
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Create a FAISS vector store and retriever
vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Step 5: Initialize the LLM via Groq
llm = ChatGroq(model="openai/gpt-oss-120b", api_key="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")

# Step 6: Build the RAG chain
system_prompt = (
    "You are a helpful assistant. Provide answers based on the provided context. "
    "If the information is not in the context, use your intelligence to answer the questions."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Step 7: Ask a question
query = input("Enter your query: ")
response = rag_chain.invoke({"input": query})

print("\nQuery:", query)
print("Answer:", response["answer"])