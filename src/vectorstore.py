import os
import chromadb
from chromadb.config import Settings
import uuid
from typing import List,Dict,Any,Tuple
import numpy as np

class VectorStore:
    """Manage document embeddings in a chromadb vector store"""
    def __init__(self, collection_name: str = 'pdf_documents', persist_directory: str = './data/chroma_db'):
        """Initialize the vector store
        
        Args:
            collection_name: Name of the chromadb collection 
            persist_directory: Directory to persist the vector data
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
        
        
    def _initialize_store(self):
        """Initialize the chromadb client and collection"""
        try:
            #Create the persistent chromadb client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            #Get or create the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF Document embeddings for Rag"} 
            )
            print(f"Vector store initialized successfully. Collection: {self.collection_name}")
            print(f"Persist directory: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
        
    def add_documents(self, documents: List[Any], embedding: np.ndarray):
        """Add documents and their embeddings to the vector store
        
        Args:
            documents: List of Document objects to add
            embedding_manager: Instance of EmbeddingManager to generate embeddings
        """
        if len(documents)!=len(embedding):
            raise ValueError("Number of documents and embeddings must match.")
        print(f"Adding {len(documents)} documents to vector store...")
        
        
        ids = []
        metadatas = []
        documents_texts = []
        embeddings_list = []
        for i ,(doc, embedding) in enumerate(zip(documents, embedding)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadatas.append(metadata)
            documents_texts.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
            
            
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_texts
            )
            print(f"Successfully added {len(documents)} documents to vector store.")
            print(f"Documents added successfully. Total documents in store: {self.collection.count()}")
        
        
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
        
vector_store = VectorStore()


if __name__ == "__main__":
    vector_store = VectorStore()