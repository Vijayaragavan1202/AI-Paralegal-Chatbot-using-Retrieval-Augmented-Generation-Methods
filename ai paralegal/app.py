import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile

# --- App Title and Configuration ---
st.set_page_config(page_title="AI Paralegal", page_icon="⚖️")
st.title("AI Paralegal Assistant")
st.write("Upload a document and ask questions about its content. The AI runs completely on your local machine.")

# --- Function to Create the RAG Chain (Cached for performance) ---
# This function creates the vector store and retrieval chain.
# Caching ensures we don't re-process the document on every interaction.
@st.cache_resource(show_spinner="Processing document...")
def create_rag_chain(uploaded_file):
    # Use a temporary file to handle the upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load the document based on its file type
    if tmp_file_path.endswith('.pdf'):
        loader = PyPDFLoader(tmp_file_path)
    else:
        loader = TextLoader(tmp_file_path)
    
    docs = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Create local embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Define the local LLM and prompt
    llm = ChatOllama(model="llama3:8b")
    prompt = ChatPromptTemplate.from_template("""
    You are an AI paralegal assistant. Answer the following question based ONLY on the provided context.
    If the answer is not in the context, reply with "I cannot find the answer in the provided documents."

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Create the retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Clean up the temporary file
    os.remove(tmp_file_path)
    
    return retrieval_chain

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Your Document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    st.info("Your document is processed locally and is never sent to an external server.")

# --- Main Chat Interface ---
if uploaded_file is not None:
    # Create the RAG chain once the file is uploaded
    try:
        rag_chain = create_rag_chain(uploaded_file)
    except Exception as e:
        st.error(f"Failed to process the document. Please try again.")
        st.error(f"Error details: {e}")
        st.stop() # Stop the app if processing fails

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": prompt})
                st.markdown(response["answer"])
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

else:
    st.info("Please upload a document in the sidebar to begin.")
