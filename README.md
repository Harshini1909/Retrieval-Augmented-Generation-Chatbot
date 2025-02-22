# Retrieval Augmented Generation (RAG) Chatbot Application

---

## Overview
This application is a Retrieval Augmented Generation (RAG) chatbot built with Streamlit. It allows users to upload multiple documents (PDF, DOCX, TXT), parse and preprocess them, and interact with the documents through a chat interface. The application leverages large language models and retrieval mechanisms to fetch relevant passages and generate responses.

---

## Features
- **File Uploading**: Users can upload multiple document formats (PDF, DOCX, TXT).
- **Document Preprocessing**: Automatically parses and preprocesses uploaded documents.
- **Interactive Chat Interface**: Users can ask questions and receive contextually relevant answers.
- **Local Vector Database**: Stores and retrieves document embeddings using Chroma.
- **Cloud Vector Database**: Optionally integrates with Pinecone for embedding storage and retrieval.
- **Source Document Retrieval**: Displays source passages that contribute to generated answers.

---

## Design and Approach
### Application Workflow
1. **File Upload**: Users upload documents using the Streamlit file uploader.
2. **Document Parsing**: Files are parsed based on their type (PDF, DOCX, TXT).
3. **Text Splitting**: Documents are split into manageable chunks for embedding.
4. **Embedding and Storage**: Document chunks are embedded and stored in a vector database (local or Pinecone).
5. **Chat Interaction**: Users ask questions, and the chatbot retrieves relevant passages to augment responses.
6. **Response Display**: Answers are generated and displayed along with source documents.

### Context Management
- Chat history is maintained during each session to enable context-aware responses.
- Users can clear chat history via a sidebar button.

---

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- API keys for:
  - OpenAI
  - Pinecone (optional for cloud storage)

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Harshini1909/Retrieval-Augmented-Generation-Chatbot.git
   cd RAG-Chatbot
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Secrets:**
   Create a `secrets.toml` file in the `.streamlit` directory with the following structure:
   ```toml
   [secrets]
   openai_api_key = "your_openai_api_key"
   pinecone_api_key = "your_pinecone_api_key"
   pinecone_env = "your_pinecone_environment"
   pinecone_index = "your_pinecone_index_name"
   ```

5. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

---

## Usage Instructions

1. Launch the application by running:
   ```bash
   streamlit run app.py
   ```

2. Upload documents using the file uploader in the sidebar. Supported formats include:
   - PDF
   - DOCX
   - TXT

3. Enter your OpenAI API key and (optionally) Pinecone credentials in the sidebar.

4. Click "Process Documents" to preprocess and embed the uploaded files.

5. Use the chat input field to ask questions about the uploaded documents.

6. View answers and their source documents in the main interface.

7. Clear chat history using the sidebar button if needed.

---

## Enhancements and Future Improvements

- **Additional Document Formats**: Extend support for more file types such as CSV and Markdown.
- **Advanced Retrieval**: Enhance retrieval logic with semantic search and ranking techniques.
- **Improved Context Management**: Incorporate advanced memory modules for long-term context retention.
- **UI Improvements**: Add features like query autocomplete and file management.

---

## Acknowledgments
This project utilizes the following tools and libraries:

- [LangChain](https://www.langchain.com/): For conversational AI workflows.
- [Streamlit](https://streamlit.io/): For building an interactive web application.
- [OpenAI API](https://openai.com/api/): For language model integration.
- [Pinecone](https://www.pinecone.io/): For scalable vector storage and retrieval.
- [PyPDF2](https://pypdf2.readthedocs.io/): For parsing PDF documents.
- [python-docx](https://python-docx.readthedocs.io/): For parsing DOCX documents.
- [Chroma](https://docs.trychroma.com/): For local vector database management.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

