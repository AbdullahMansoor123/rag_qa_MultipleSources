---
title: Q/A Rag System
emoji: 🏆
colorFrom: gray
colorTo: green
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
short_description: RAG Project for Customized Question-Answering System
---

# RAG Project for Customized Question-Answering System

This project implements a **Retrieval-Augmented Generation (RAG) System** for a customized question-answering platform. Users can ask questions, and the system retrieves relevant information from multiple sources (Wikipedia, PDFs, and websites), using GPT-3.5-turbo to generate precise and contextual answers.

## Table of Contents
- [RAG Project for Customized Question-Answering System](#rag-project-for-customized-question-answering-system)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [System Architecture](#system-architecture)
  - [Prerequisites](#prerequisites)
  - [How to Run Locally](#how-to-run-locally)
  - [Uploading Documents](#uploading-documents)
  - [Querying the Model](#querying-the-model)
  - [Performance Trade-offs](#performance-trade-offs)
  - [Future Enhancements](#future-enhancements)

---

## Project Overview

This project leverages a RAG approach by combining retrieved documents from Wikipedia, user-uploaded PDFs, and online websites with generative models to provide contextually accurate answers to user queries. The application uses **Streamlit** for the user interface and integrates **OpenAI's GPT-3.5-turbo** for text generation.

Key features:
- **Multi-source information retrieval**: Pulls data from Wikipedia, uploaded PDFs, and user-specified websites.
- **Custom question-answering**: Users can interact with the model by asking questions and selecting the data sources.
- **Streamlit UI**: Provides a user-friendly interface for inputting questions, selecting sources, and viewing answers.
- **Langchain & FAISS**: For handling document retrieval and efficient vector search.

---

## System Architecture

The architecture of the RAG system consists of the following components:

1. **Document Retrieval**:
    - **Wikipedia API**: Extracts relevant content from Wikipedia based on user queries.
    - **PDF Uploader**: Allows users to upload documents, which are processed and stored for retrieval.
    - **Website Scraping**: Scrapes web pages to retrieve textual data for analysis.

2. **Document Embeddings**:
    - Each document is converted into embeddings using a language model and stored in a FAISS index for fast retrieval.

3. **Retriever**:
    - Queries are first passed to the retriever, which searches for the most relevant document embeddings from the stored sources (Wikipedia, PDFs, websites).

4. **Generative Model**:
    - The retrieved documents are combined with the user query and passed to the GPT-3.5-turbo model for generating a contextually relevant answer.

---

## Prerequisites

Before running the project, ensure you have the following:

- **Python 3.10+**: Make sure Python is installed on your machine.
- **OpenAI API Key**: Sign up for an OpenAI account and get your API key [here](https://platform.openai.com/signup).
- **LangSmith API Key**: To track each LLM response with Langchain [here](https://smith.langchain.com).
---

## How to Run Locally

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AbdullahMansoor123/rag_qa_MultipleSources
   cd rag-question-answering
   ```

2. **Install Dependencies**:
   Ensure all the required Python libraries are installed by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up OpenAI API Key**:
   - Add your OpenAI API key to the `.env` file or input it directly in the Streamlit sidebar when running the app.
   - Example `.env` file:
     ```bash
     OPENAI_API_KEY="your-openai-api-key-here"
     ```
    Note: You can either give openai API key locally in the `.env` file or by streamlit user interface

4. **Tracking LLM Responses with Langchain**:

    To track and analyze each LLM response, Langchain provides a tracing and logging service via **LangSmith**. Follow these steps:

   - Create an account at [LangSmith](https://smith.langchain.com/) and generate an API key.
   - Add the following to your `.env` file to enable tracing:
     ```bash
     LANGCHAIN_API_KEY="your-langchain-api-key-here"
     LANGCHAIN_TRACING_V2="true"
     LANGCHAIN_PROJECT="Your_Project_Name"
     ```

   - **Monitor LLM Interactions**:
    Once the app is running, you can view the logs of every interaction and response on the LangSmith dashboard under your project. This helps you track model behavior and optimize performance.

    Note: For now you can only use this feature when running streamlit locally
---

1. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

2. **Access the App**:
   - Open a browser and go to `http://localhost:8501` to interact with the Q/A system.
   - Use the interface to upload PDFs, enter website URLs, and select sources for question answering.

---

## Uploading Documents

The system supports uploading and processing PDF files:

1. Click the **"Upload PDF"** button in the sidebar.
2. Select a PDF document from your local machine.
3. The system will extract and store the content for retrieval.
4. The PDF will be processed, and its text will be indexed in the FAISS vector store.

---

## Querying the Model

To query the system:

1. **Input a Question**: Enter your question in the text input box.
2. **Select Data Sources**: Choose from the available data sources (Wikipedia, PDF, Websites) by toggling the respective options.
3. **Submit**: Press the **"Enter"** button.
4. **View Results**: The system will retrieve relevant documents and generate a response using GPT-3.5-turbo.

Example queries:
- "What are the latest AI trends in 2024?"
- "Summarize the content of the uploaded PDF."

---

## Performance Trade-offs

Several trade-offs were made during the development process:

1. **Retrieval Speed vs. Completeness**: The system uses FAISS for fast vector-based retrieval, which prioritizes speed over exhaustive document search. This is beneficial for quick responses but might miss some nuanced information in large documents.
  
2. **Memory Usage**: Storing embeddings in memory allows fast retrieval but comes at the cost of higher memory consumption, especially when large PDFs or web content are involved. A disk-based store could reduce memory usage but slow down retrieval times.

3. **Generative Model Latency**: The use of GPT-3.5-turbo introduces latency, especially for complex questions that require processing multiple sources. This trade-off was necessary to maintain high-quality answers.

---

## Future Enhancements

Potential improvements to the system include:

1. **Add More Data Sources**: Extend the system to support additional data sources like databases or cloud documents.
2. **Advanced Caching**: Implement caching mechanisms to speed up repeated queries and avoid redundant data processing.
3. **Fine-tune the Model**: Adapt the generative model with specific domain knowledge for improved accuracy.
4. **UI Improvements**: Add better visualization for source selection and document preview before querying.

