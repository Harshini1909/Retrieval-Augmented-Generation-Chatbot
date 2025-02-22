import os  # For file system operations
import tempfile  # For creating temporary files
import pinecone  # For interacting with Pinecone vector database
from pathlib import Path  # For managing file paths

from langchain.chains import ConversationalRetrievalChain  # For retrieval-augmented conversations
from langchain.embeddings import OpenAIEmbeddings  # For embedding text using OpenAI models
from langchain.vectorstores import Chroma, Pinecone  # For vector storage options
from langchain.document_loaders import (  # For loading various document formats
    DirectoryLoader, PyPDFLoader, DocxLoader, TextLoader
)
from langchain.text_splitter import CharacterTextSplitter  # For splitting text into smaller chunks
from langchain.llms.openai import OpenAIChat  # For OpenAI-based chat models

import streamlit as st  # For building the web interface

# Directory for temporary storage of uploaded files
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')

# Directory for storing the local vector database
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

# Configure the Streamlit app
st.set_page_config(page_title="RAG Application", layout="wide")
st.title("Retrieval Augmented Generation Chatbot")

# Ensure the temporary and local vector store directories exist
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

def load_documents():
    """Loads documents from the temporary directory."""
    documents = []  # Initialize an empty list for storing documents
    for file in TMP_DIR.iterdir():  # Iterate through files in the temporary directory
        try:
            # Determine the appropriate loader based on file type
            if file.suffix == '.pdf':
                loader = PyPDFLoader(file.as_posix())
            elif file.suffix == '.docx':
                loader = DocxLoader(file.as_posix())
            elif file.suffix == '.txt':
                loader = TextLoader(file.as_posix())
            else:
                continue  # Skip unsupported file types
            documents.extend(loader.load())  # Load and append document content
        except Exception as e:
            # Show error if document loading fails
            st.error(f"Error loading file {file.name}: {e}")
    return documents  # Return the loaded documents

def split_documents(documents):
    """Splits documents into smaller chunks."""
    try:
        # Define a text splitter with specified chunk size and overlap
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(documents)  # Split and return chunks
    except Exception as e:
        # Show error if splitting fails
        st.error(f"Error splitting documents: {e}")
        return []

def embeddings_on_local_vectordb(texts):
    """Embeds and stores documents locally using Chroma."""
    try:
        # Create a Chroma vector database from documents
        vectordb = Chroma.from_documents(
            texts, embedding=OpenAIEmbeddings(),
            persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
        )
        vectordb.persist()  # Persist the database to disk
        return vectordb.as_retriever(search_kwargs={'k': 5})  # Return retriever
    except Exception as e:
        # Show error if local embedding fails
        st.error(f"Error initializing local vector database: {e}")
        return None

def embeddings_on_pinecone(texts):
    """Embeds and stores documents in Pinecone."""
    try:
        # Initialize Pinecone with API key and environment
        pinecone.init(api_key=st.session_state.pinecone_api_key, environment=st.session_state.pinecone_env)
        # Create embeddings using OpenAI
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
        # Store documents in Pinecone
        vectordb = Pinecone.from_documents(texts, embeddings, index_name=st.session_state.pinecone_index)
        return vectordb.as_retriever()  # Return retriever
    except Exception as e:
        # Show error if Pinecone embedding fails
        st.error(f"Error initializing Pinecone vector database: {e}")
        return None

def query_llm(retriever, query):
    """Queries the LLM using a conversational retrieval chain."""
    try:
        # Create a retrieval-augmented chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=OpenAIChat(openai_api_key=st.session_state.openai_api_key),
            retriever=retriever,
            return_source_documents=True
        )
        # Get response from the chain
        result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
        st.session_state.messages.append((query, result['answer']))  # Update chat history

        # Display source documents in an expandable section
        with st.expander("Source Documents"):
            for source_doc in result.get('source_documents', []):
                st.write(source_doc.metadata.get('source', 'Unknown Source'))
                st.write(source_doc.page_content)

        return result['answer']  # Return the answer
    except Exception as e:
        # Show error if querying fails
        st.error(f"Error querying LLM: {e}")
        return ""

def input_fields():
    """Renders input fields in the sidebar."""
    with st.sidebar:
        # OpenAI API Key input
        if "openai_api_key" in st.secrets:
            st.session_state.openai_api_key = st.secrets.openai_api_key
        else:
            st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password")

        # Pinecone API Key input
        if "pinecone_api_key" in st.secrets:
            st.session_state.pinecone_api_key = st.secrets.pinecone_api_key
        else:
            st.session_state.pinecone_api_key = st.text_input("Pinecone API Key", type="password")

        # Pinecone Environment input
        if "pinecone_env" in st.secrets:
            st.session_state.pinecone_env = st.secrets.pinecone_env
        else:
            st.session_state.pinecone_env = st.text_input("Pinecone Environment")

        # Pinecone Index Name input
        if "pinecone_index" in st.secrets:
            st.session_state.pinecone_index = st.secrets.pinecone_index
        else:
            st.session_state.pinecone_index = st.text_input("Pinecone Index Name")

        # Checkbox to toggle Pinecone usage
        st.session_state.pinecone_db = st.checkbox('Use Pinecone Vector DB', value=False)

        # File uploader for document uploads
        st.session_state.source_docs = st.file_uploader("Upload Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

def process_documents():
    """Processes uploaded documents and prepares retriever."""
    if not st.session_state.openai_api_key:
        st.warning("Please provide the OpenAI API Key.")  # Warn if OpenAI key is missing
        return

    if not st.session_state.source_docs:
        st.warning("Please upload documents to process.")  # Warn if no documents uploaded
        return

    with st.spinner("Processing documents..."):
        # Save uploaded documents temporarily
        for source_doc in st.session_state.source_docs:
            with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix=Path(source_doc.name).suffix) as tmp_file:
                tmp_file.write(source_doc.read())  # Write uploaded content to temp file

        try:
            # Load, split, and embed documents
            documents = load_documents()
            texts = split_documents(documents)

            # Initialize the retriever based on selected vector DB
            if st.session_state.pinecone_db:
                st.session_state.retriever = embeddings_on_pinecone(texts)
            else:
                st.session_state.retriever = embeddings_on_local_vectordb(texts)

            if st.session_state.retriever:
                st.success("Documents processed successfully!")  # Show success message
        except Exception as e:
            st.error(f"An error occurred during document processing: {e}")  # Show error

def boot():
    """Bootstraps the Streamlit app."""
    input_fields()  # Render sidebar input fields
    if st.button("Process Documents"):
        process_documents()  # Process documents on button click

    if "messages" not in st.session_state:
        st.session_state.messages = []  # Initialize chat history if not present

    for message in st.session_state.messages:
        st.chat_message("user").write(message[0])  # Display user messages
        st.chat_message("assistant").write(message[1])  # Display assistant responses

    # Handle user query input
    if query := st.chat_input("Ask a question about the uploaded documents"):
        st.chat_message("user").write(query)  # Display user query
        response = query_llm(st.session_state.retriever, query)  # Get LLM response
        st.chat_message("assistant").write(response)  # Display LLM response

    # Clear chat history on button click
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared.")

if __name__ == '__main__':
    boot()  # Run the app
