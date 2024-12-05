import os
import streamlit as st
import requests
from io import BytesIO
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader


# Function to initialize LLM
def get_llm_with_free_802key_1():
    api_key = os.environ['OPENAI_API_KEY']
    return ChatOpenAI(api_key=api_key, model_name="gpt-4o-mini", temperature=0)

# Streamlit UI
st.title('Auto Order System')

# Input: URL
url = st.text_input("Enter the URL of the menu:")


# Input: Number of diners and budget
question_1 = st.text_input("How many people are dining? Reply in numbers: ")
question_2 = st.text_input("Additional requests (Optional): ")
past_review = st.text_input("Input customer reviews: (Optional. "
                            "For current version, we are not able to read views from websites. Please copy and paste the reviews directly below)")

# Process documents and answer questions
if question_1 and url:
    try:

        if url:
            try:
                if url.lower().endswith(".pdf"):
                    
                    response = requests.get(url)
                    pdf_filename = "new_menu.pdf"
                    if response.status_code == 200:
                        with open(pdf_filename, 'wb') as f:
                            f.write(response.content)
                        st.info("PDF processed successfully")
                    else:
                        st.info("Failed to download PDF. Status code: {response.status_code}")
                        loader = None
                    loader = PyPDFLoader(pdf_filename)
                else:
                    st.info("Scraping webpage...")
                    loader = WebBaseLoader(url)
                
                if loader:  # Ensure loader is defined
                    documents = loader.load()
                    if not documents:
                        st.error("No documents were loaded. Please check the URL or PDF content.")
                    else:
                        st.success(f"Loaded {len(documents)} documents successfully!")
                        # Debugging: Print the content of the first document
                        #st.write("First Document Content:", documents[0].page_content if documents else "No content found.")
                        #for i, doc in enumerate(documents):
                        #    st.write(f"Document {i + 1} Content:", doc.page_content)
                else:
                    st.error("Loader could not be initialized.")
            except Exception as e:
                st.error(f"Failed to process the URL: {e}")
                
        # Convert documents to LangChain format
        # documents = [Document(page_content=doc["content"], metadata={}) for doc in documents]

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        #if chunks:
        #    if len(chunks) > 0:
        #        st.write(chunks[0])
        #    if len(chunks) > 1:
        #        st.write(chunks[1])
        #else:
        #    st.error("No content could be split into chunks. Please check the input document.")
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever()

        # Initialize LLM and chain
        llm = get_llm_with_free_802key_1()
        chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

        # Construct query
        if question_2:
            hint  = f"Help order food for {question_1} people. Additional requests:{question_2}" 
            query = f"""Help order food for {question_1} people. Additional requests:{question_2}.
                    Provide a list food based on the number of people and give a calculation of total cost before tax and tip."""
        else:
            hint  = f"Help order food for {question_1} people."
            query = f"""Help order food for {question_1} people. 
            Provide a list food based on the number of people and give a calculation of total cost before tax and tip. 
            Assume no dietary restriction and normal budget requirement."""
        # Get response
        if past_review:
            query += f""" When providing dish recommendations, please refer to the following reviews: {past_review}. """
            hint  += f""" We are referring to the dining reviews you provided."""
        st.write(hint)
        result = chain.invoke(query)
        response = result['result']
        st.write(response)
    except Exception as e:
        st.error(f"Error generating response: {e}")
else:
    if url and (not question_1):
        st.info("Please provide both the number of people.")
