import os
import re
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

import streamlit as st


# Function to validate a URL using regex
def is_valid_url(url: str) -> bool:
    regex = r"((http|https)://)(www\.)?[a-zA-Z0-9@:%._\+~#?&//=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%._\+~#?&//=]*)"
    return bool(re.match(regex, url))


# Function to split text into chunks for processing
def split_text_into_chunks(text: str, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)


# Function to create an FAISS vector store from chunks of text
def create_vector_store(chunks: list) -> FAISS:
    embedding = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embedding)


# Function to create a Wikipedia tool
def wiki_tool() -> WikipediaQueryRun:
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    return WikipediaQueryRun(api_wrapper=api_wrapper)


# Function to create a retriever tool from a website URL
def website_tool(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    documents = split_text_into_chunks(docs)
    vectordb = create_vector_store(documents)
    retriever = vectordb.as_retriever()
    return create_retriever_tool(retriever, "world-best-cities", "Guide on best cities in the world")


# Function to extract text from a list of PDF files
def get_pdf_text(pdf_docs) -> str:
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf, strict=False)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to create a retriever tool for PDF content
def pdf_retriever_tool(pdf_docs):
    pdf_text = get_pdf_text(pdf_docs)
    chunks = split_text_into_chunks(pdf_text)
    vector_db = create_vector_store(chunks)
    return create_retriever_tool(vector_db.as_retriever(), "pdf_tool", "PDF Content")


# Function to generate an LLM response based on the selected sources
def llm_response(user_question: str, tools: list, openai_api_key: str) -> str:
    # Select the LLM model with the provided OpenAI API key
    gpt = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    
    # Create agent
    my_agent = create_openai_tools_agent(gpt, tools, hub.pull("hwchase17/openai-functions-agent"))
    agent_executor = AgentExecutor(agent=my_agent, tools=tools, verbose=False)

    # Get the response from the LLM agent
    answer = agent_executor.invoke({"input": user_question})
    return answer['output']


# Main function to set up the Streamlit app interface
def main():
    st.title("Question-Answering System")

    # Sidebar for OpenAI API key
    with st.sidebar:
        api_key_from_env = os.getenv('OPENAI_API_KEY')
        openai_api_key = api_key_from_env if api_key_from_env else st.text_input("Enter your OpenAI API Key:", type="password")

        st.subheader("More Options")
        sources = ["Wikipedia", "PDF", "Website"]
        selected_sources = st.multiselect("Select data sources:", options=sources, default=["Wikipedia"])

    # Input box for user question
    user_question = st.text_input("Ask your question:")

    # Tools selected
    tools = []

    if "Wikipedia" in selected_sources:
        tools.append(wiki_tool())

    if "PDF" in selected_sources:
        uploaded_files = st.file_uploader("Upload a document (PDF):", type=['pdf'], accept_multiple_files=True)
        if uploaded_files:
            with st.spinner("Processing PDF..."):
                tools.append(pdf_retriever_tool(uploaded_files))
        else:
            st.warning("Please upload at least one PDF file.")

    if "Website" in selected_sources:
        url = st.text_input("Enter website URL:")
        if is_valid_url(url):
            tools.append(website_tool(url))
        else:
            st.warning("Please enter a valid URL.")

    # Process the question and display the response
    if user_question and tools and openai_api_key:
        with st.spinner("Generating response..."):
            answer = llm_response(user_question, tools, openai_api_key)
            st.write(answer)
    else:
        st.warning("Please provide all inputs (question, data sources, and OpenAI API key).")


# Entry point of the Streamlit app
if __name__ == '__main__':
    main()
