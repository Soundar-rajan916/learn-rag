import streamlit as st
import os
import tempfile
from src.main import main as process_document
from src.main import question

st.set_page_config(page_title="Learn RAG", page_icon="📚", layout="centered")

st.title("📚 Learn RAG Assistant")
st.write("Upload a PDF document to train the AI, then ask questions about it!")

# Sidebar for document upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF document", type=['pdf'])
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document... This may take a moment."):
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    # Process the document using the backend
                    success = process_document(tmp_file_path)
                    if success:
                        st.success("Document successfully processed and added to the knowledge base!")
                        st.session_state['document_processed'] = True
                    else:
                        st.error("Failed to process the document. Please try a different file.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    # Clean up the temporary file
                    if os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)
    else:
        st.info("Please upload a PDF to get started.")

# Main chat interface
st.header("Chat with your Document")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the uploaded document..."):
    # Check if a document has been processed
    if not st.session_state.get('document_processed', False):
        st.warning("Please upload and process a document first before asking questions.")
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            try:
                # Query the RAG backend
                response = question(prompt)
                
                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Failed to get a response: {e}")
