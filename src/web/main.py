import sys
import os
# Add the project root to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
from src.main import question as q
from src.main import main as m

st.title("This is the sample Rag-pipeline")
question = st.text_input("Enter your question")

if question:
	st.write(q(question))

st.write("Upload a file for RAG pipeline")
file_upload = st.file_uploader("Upload a file", type=["pdf", "txt", "docx"])


if file_upload is not None:
    st.write(file_upload.name)
    # Create the data/pdf directory at project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    pdf_dir = os.path.join(project_root, "data", "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    
    path = os.path.join(pdf_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getbuffer())
    if st.button("Process File"):
        if file_upload:
           st.write("Processing file...")
           success = m(path)
           if success:
               st.success("File processed successfully!")
           else:
               st.warning("Could not extract any text from the file. Is it a scanned image or empty PDF?")
        else:
           st.write("Please upload a file first.")

# if file_upload:
# 	st.write(file_upload)
# 	m(file_upload)




# print(type(file_upload))