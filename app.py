import streamlit as st
import json
import requests
from dotenv import load_dotenv
from langchain import hub
from PyPDF2 import PdfReader
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor



import re

def isValidURL(url):
    regex = r"((http|https)://)(www\.)?[a-zA-Z0-9@:%._\+~#?&//=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%._\+~#?&//=]*)"
    return True if re.match(regex, url) else False

def wiki_tool():
    api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
    return wiki


def website_tool(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap = 200).split_documents(docs)
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
    retriever = vectordb.as_retriever()
    retriever_tool = create_retriever_tool(retriever, "world-best-cities","Guide on best cities in the world")
    return retriever_tool

# get all the pdf in a single document
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf,strict=False)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# get text chunks
def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap = 200)
    chunks = text_splitter.split_text(docs)
    return chunks

#get text embedding
def get_vector_store(chunks):
    embedding = OpenAIEmbeddings()
    vectordb = FAISS.from_texts(chunks, embedding)
    return vectordb

def pdf_retriever_tool(pdf_path):
    pdf_text = get_pdf_text(pdf_path)
    chunks = get_text_chunks(pdf_text)
    vector_db = get_vector_store(chunks)
    pdf_retriever =  vector_db.as_retriever()
    return create_retriever_tool(pdf_retriever, "pdf_tool","PDF Content")

# llm response
def llm_response(user_question,selected_sources):

    tools = []
    with st.sidebar:
        if "Wikipedia" in selected_sources:
            wiki = wiki_tool()
            tools.append(wiki)
        
        uploaded_file = None
        if "PDF" in selected_sources :
            # Upload document option
            uploaded_files = st.file_uploader("Upload a document (PDF):", type=['pdf'], accept_multiple_files=True)
            if uploaded_file is not None:
                with st.spinner("Processing..."):
                    pdf = pdf_retriever_tool(uploaded_files)
                    tools.append(pdf)
            else:
                st.write("Please make sure to select a pdf.")
        
        if "Website" in selected_sources :
            url = st.text_input("Enter website URL:")
            if isValidURL(url) == False :
                st.warning("Please enter a valid URL to proceed.")
            else:
                website = website_tool(url)
                tools.append(website)

    #select llm model
    gpt = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    ## get the prompt from hub
    prompt = hub.pull("hwchase17/openai-functions-agent")

    my_agent = create_openai_tools_agent(gpt, tools, prompt)
    agent_executer  = AgentExecutor(agent=my_agent, tools= tools, verbose=False)
    
    answer = agent_executer.invoke({
            "input": {user_question}
            })
    return answer['output']
  

def main():
    try:
        st.title("Question-Answering System")
        # Input box for user question
        user_question = st.text_input("Ask your question:")
        
        
        with st.sidebar:
            st.subheader("More Options")
            # Multi-select for data sources
            sources = ["Wikipedia", "PDF", "Website"]
            selected_sources = st.multiselect("Select data sources:", options=sources,default="Wikipedia")

        # Process the question and display the response
        if user_question and selected_sources:
            answer = llm_response(user_question, selected_sources)
            st.write(answer)

    except:
        pass


if __name__ == '__main__':
    main()
