import streamlit as st
import os
from rag_utils import (
    load_and_split_pdf,
    initialize_embedding_model,
    get_vector_store,
    initialize_llm,
    create_rag_chain
)
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # # Load HTML template
    # with open('templates/index.html', 'r') as file:
    #     html_template = file.read()
    # st.markdown(html_template, unsafe_allow_html=True)

    # Initialize session state
    if 'vectorstore_created' not in st.session_state:
        st.session_state.vectorstore_created = False

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])

    if uploaded_file is not None:
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
                vectorstore = get_vector_store(docs, embedding_model)
                
                st.session_state.vectorstore_created = True
                st.success("Document processed successfully!")

            # Clean up temporary file
            os.unlink(temp_file_path)

    # Question input and answer generation
    if st.session_state.vectorstore_created:
        question = st.text_input("Ask a question about the document:")
        
        if question:
            with st.spinner("Generating answer..."):
                embedding_model = initialize_embedding_model()
                vectorstore = get_vector_store(embedding_model=embedding_model)
                retriever = vectorstore.as_retriever(
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