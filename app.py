import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from rag_utils import (
    load_and_split_pdf,
    initialize_embedding_model,
    get_vector_store,
    initialize_llm,
    create_rag_chain
)

# Load environment variables
load_dotenv()

def main():
    st.title("RAG with Streamlit and Gemini")

    # Initialize session state
    if 'vectorstore_created' not in st.session_state:
        st.session_state.vectorstore_created = False
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])

    if uploaded_file is not None:
        # Check if a new file has been uploaded
        if uploaded_file.name != st.session_state.uploaded_file_name:
            # Clear the previous vectorstore if a new file is uploaded
            st.session_state.vectorstore = None
            st.session_state.vectorstore_created = False
            st.session_state.uploaded_file_name = uploaded_file.name

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                # Initialize embedding model
                embedding_model = initialize_embedding_model()

                # Load and process document
                docs = load_and_split_pdf(temp_file_path)

                # Delete the old vector store directory if it exists
                if st.session_state.vectorstore_created:
                    try:
                        st.session_state.vectorstore.delete_collection()
                        st.session_state.vectorstore = None
                        st.session_state.vectorstore_created = False
                        st.write("Old vector store deleted.")
                    except Exception as e:
                        st.error(f"Error deleting old vector store: {e}")

                # Create a new vector store
                st.session_state.vectorstore = get_vector_store(docs, embedding_model)
                st.session_state.vectorstore_created = True
                st.success("Document processed successfully!")

            # Clean up temporary file
            os.unlink(temp_file_path)

    # Question input and answer generation
    if st.session_state.vectorstore_created:
        question = st.text_input("Ask a question about the document:")

        if question:
            with st.spinner("Generating answer..."):
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}
                )

                system_prompt = """
                You are a knowledgeable expert. Your task is to answer questions based on the provided documents.
                If the answer is not clearly stated in the documents, use your knowledge in combination with the documents.
                Clearly indicate which parts come from the documents and which parts are from your knowledge (if any).
                Your answers should be clear and accurate.

                {context}
                """

                llm = initialize_llm()
                rag_chain = create_rag_chain(retriever, llm, system_prompt)

                response = rag_chain.invoke({"input": question})
                st.write("Answer:", response['answer'])

if __name__ == "__main__":
    main()