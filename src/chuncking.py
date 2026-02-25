from langchain_core.documents import Document

def chunk_documents(documents, size=500, overlap=50):
    chunks = []

    for doc in documents:
        text = doc.page_content
        metadata = doc.metadata

        start = 0
        while start < len(text):
            end = start + size
            chunk_text = text[start:end]

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata
                )
            )

            start = end - overlap

    return chunks

if __name__ == "__main__":
    chunk_documents()