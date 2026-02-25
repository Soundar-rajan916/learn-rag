# embedding_manager = EmbeddingManager()
# embeddingmanager = embedding_manager.generate_embeddings(texts)
# vector_store.add_documents(chunks, embeddingmanager)

from .loader import loader
from .chuncking import chunk_documents
from .embedding import EmbeddingManager, embedding_manager
from .vectorstore import vector_store
from .llm import rag_simple, llm
from .ragretriever import RAGRetriever


def question(query):
    """Answer a query using the RAG retriever."""
    retriever = RAGRetriever(vector_store, embedding_manager)
    response = rag_simple(query, retriever)
    return response


def main(file_path):
        import os
        print(f"Loading and processing documents from {file_path}...")
        print(f"File size in main(): {os.path.getsize(file_path)} bytes")
        documents = loader(file_path)
        print(f"Loaded documents count: {len(documents)}")

        chunks = chunk_documents(documents)
        print(f"Generated chunks count: {len(chunks)}")

        texts = [chunk.page_content for chunk in chunks]
        
        if not texts:
            print("Warning: No texts extracted from the documents. Skipping embedding generation.")
            return False

        emb_manager = EmbeddingManager()
        embeddingmanager = emb_manager.generate_embeddings(texts)
        if len(chunks) > 0:
             vector_store.add_documents(chunks, embeddingmanager)
             
        return True
        
if __name__ == "__main__":
    main()