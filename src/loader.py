from langchain_core import documents
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader , DirectoryLoader
def loader(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents

if __name__ == "__main__":
    loader()