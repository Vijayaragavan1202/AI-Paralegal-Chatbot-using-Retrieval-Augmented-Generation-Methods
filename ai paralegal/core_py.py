# ==============================================================================
# Fully Local AI Paralegal RAG Script
# ==============================================================================
# This script runs entirely on your local machine.
# - Embeddings: Handled by Hugging Face's 'all-MiniLM-L6-v2' model.
# - LLM Generation: Handled by a local model served via Ollama (e.g., 'llama3:8b').
# ==============================================================================

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Load the Document ---
# This loads the text from our sample case file.
print("Loading document...")
try:
    loader = TextLoader("case_document.txt")
    docs = loader.load()
except Exception as e:
    print(f"Error: Could not load 'case_document.txt'. Make sure the file exists in the same directory.")
    print(f"Details: {e}")
    exit()

# --- 2. Split the Document into Chunks ---
# The text is split into smaller, semantically meaningful pieces.
print("Splitting document into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# --- 3. Create Local Embeddings and Store in Chroma Vector DB ---
# We use a Hugging Face model to create embeddings locally.
# The model 'all-MiniLM-L6-v2' will be downloaded automatically the first time.
print("Creating local embeddings and vector store...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# --- 4. Define the Local LLM and the Prompt ---
# We connect to the local model served by Ollama.
# Ensure the Ollama application is running and you have pulled the model.
print("Initializing local LLM via Ollama...")
llm = ChatOllama(model="llama3:8b")

prompt = ChatPromptTemplate.from_template("""
You are an AI paralegal assistant. Answer the following question based ONLY on the provided context.
If the answer is not in the context, reply with "I cannot find the answer in the provided documents."

<context>
{context}
</context>

Question: {input}
""")

# --- 5. Create the Retrieval Chain ---
# This chain retrieves relevant document chunks and "stuffs" them into the prompt.
print("Creating retrieval chain...")
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- 6. Ask a Question ---
# We invoke the chain with a question to get a response from the local LLM.
print("Asking a question...")
question = "Who was driving the Ford F-150 and what did they claim?"
response = retrieval_chain.invoke({"input": question})

# --- Print the Result ---
print("\n" + "="*50)
print(f"Question: {question}\n")
print(f"Answer: {response['answer']}")
print("="*50 + "\n")