import os
import streamlit as st                        # used to create our UI frontend

#Libraries for Document Loaders
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader # import loaders
from langchain_openai import ChatOpenAI

#Libraries for Document Splitting, embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

#Libraries for VectorStores
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS

#Chain for Q&A after retrieving external documents
from langchain.chains import RetrievalQA

def get_llm(temperature, model):
  my_openaikey=os.environ['OPENAI_API_KEY']
  return ChatOpenAI(
    api_key=my_openaikey,
    model_name=model,
    temperature=temperature
  )

#A new llm function that will work with the free key provided to you
from langchain_openai import ChatOpenAI
def get_llm_with_free_802key_1():
  api_key_from_kyle_and_sudhir = os.environ['OPENAI_API_KEY']
  return ChatOpenAI(
    api_key=api_key_from_kyle_and_sudhir,
    #base_url="https://api.802.mba/api/providers/openai/v1/",
    model_name="gpt-4o-mini",
    temperature=0
  )
#Title for the StreamLit Page
st.title('Auto Order')
# Loading documents (Make sure constitution.pdf is available in the  directory)
loader = PyPDFLoader("The-Grill-Dinner.pdf")
documents = loader.load()
# print(documents) # print to ensure document loaded correctly.

#Splitting Documents into Chunks for embeddings and the store them in vector stores
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
# to see the chunks
# st.write(chunks[0])
# st.write(chunks[1])

embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

vector_store = FAISS.from_documents(chunks, embeddings)
# initialize OpenAI instance and set up a chain for Q&A from an LLM
#llm=get_llm(temperature=0.7, model="gpt-3.5-turbo")
#llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
llm = get_llm_with_free_802key_1()

retriever=vector_store.as_retriever()
chain = RetrievalQA.from_chain_type(llm, retriever=retriever)



# Input fields for user to enter details
question_1 = st.text_input('How many people are dining? Reply in numbers')
question_2 = st.text_input('What is the meal budget? Reply in High/Medium/Low')

question = None
# Generating the question string
if question_1 and question_2:  # Ensure inputs are provided
    question = f'Help order food for {question_1} people under a {question_2} budget.'

if question:
  # run chain
  result = chain.invoke(question)
  response = result['result']
  st.write(response)
