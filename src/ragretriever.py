from .embedding import EmbeddingManager,embedding_manager
from .vectorstore import vector_store,VectorStore
from langchain_core.documents import Document
from typing import List


class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, vector_store:VectorStore, embedding_manager:EmbeddingManager):
        """
        Initialize the retriever
        
        Args:
        vector_store:Vector store containing document embeddingd
        embedding_manager: Embedding manager to generate query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        
        
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Document]:
        """Retrive relevant documents for a given query
        Args:
            query: User query string
            top_k: Number of top relevant documents to retrieve
            score_threshold: Minimum similarity score for retrieved documents
            
        Returns:
            List of dictionaries containing retrieved documents and their metadata       
        """
        print(f"Retrieving documents for query: '{query}' with top_k={top_k} and score_threshold={score_threshold}")
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                
                for i,(doc_id,document,metadata,distance) in enumerate(zip(ids,documents,metadatas,distances)):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'document': document,
                            'metadata': metadata,
                            'similarity': similarity_score,
                            'distance': distance,
                            'rank': i+1
                        })
                         
                print(f"Retrieved {len(retrieved_docs)} documents for query: '{query}'")
            else:
                print("No documents found")
    
    
            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval:{e}")
            return []


ragretriever =  RAGRetriever(vector_store,embedding_manager)



if __name__ == "__main__":
    retriever = RAGRetriever(vector_store, embedding_manager)
