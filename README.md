# Learn RAG

`learn-rag` is a simple Retrieval-Augmented Generation (RAG) pipeline built using Python, LangChain, Google GenAI, and Streamlit. It allows users to upload documents (PDF, TXT, DOCX) and ask questions based on the uploaded content.

## Features
- **Document Upload & Processing:** Upload your PDF, TXT, or DOCX files directly via a user-friendly Streamlit web interface.
- **Automated Document Chunking & Embedding:** The backend automatically loads, chunks, and generates embeddings for your documents.
- **Retrieval-Augmented Generation:** Uses a locally managed vector store to fetch relevant document context and answers your queries utilizing Google GenAI through LangChain.

## Project Structure
```text
learn-rag/
├── data/                  # Directory where uploaded files are stored for processing
│   └── pdf/               # Automatically created to store uploaded files
├── notebook/              # Jupyter notebooks for experimentation and testing
├── src/                   # Main source code
│   ├── chuncking.py       # Handles document slicing/chunking
│   ├── embedding.py       # Manages embedding generation (e.g., HuggingFace/SentenceTransformers)
│   ├── llm.py             # Configures the LLM (Google GenAI)
│   ├── loader.py          # Document loaders
│   ├── main.py            # Main RAG orchestrator logic
│   ├── ragretriever.py    # Logic for retrieving relevant chunks from the vector store
│   ├── vectorstore.py     # Manages the vector database (Chroma/FAISS)
│   └── web/               # Web application code
│       └── main.py        # Streamlit frontend application
├── pyproject.toml         # Requirements and application metadata
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation (this file)
```

## Prerequisites
- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) (optional, strictly for dependency management if you prefer)
- A Google Gemini API key (set in your `.env` file)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Soundar-rajan916/learn-rag.git
   cd learn-rag
   ```

2. Set up your virtual environment and install dependencies:
   You can install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
   Or, if you are using `pyproject.toml` directly:
   ```bash
   pip install .
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add your Google API Key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

### Running the Web Application
The easiest way to interact with the pipeline is via the Streamlit interface:

```bash
streamlit run src/web/main.py
```

- **Upload a file:** Use the file uploader to upload your document. Click **Process File** to chunk and embed the text into the vector store.
- **Ask questions:** Once processed, enter your query into the text input box to get answers based on the context from your uploaded document.

## Technologies Used
- **[LangChain](https://python.langchain.com/)**: Orchestration of the RAG pipeline.
- **[Streamlit](https://streamlit.io/)**: Interactive web UI.
- **[ChromaDB](https://www.trychroma.com/) / [FAISS](https://github.com/facebookresearch/faiss)**: Vector storage and retrieval.
- **[Google GenAI](https://ai.google.dev/)**: Large Language Model for generation.
- **[Sentence-Transformers](https://sbert.net/)**: Embedding generation.
- **[PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) / [PyPDF](https://pypdf.readthedocs.io/en/latest/)**: PDF document loading.
