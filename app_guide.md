### README Guide: Question-Answering System with Streamlit and Langchain

This guide explains the steps and code implementation for a question-answering system that retrieves information from multiple sources such as Wikipedia, PDF documents, and websites. It uses Langchain, Streamlit, OpenAI, and FAISS for information retrieval and processing.

---

## Table of Contents
- [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Environment Setup](#environment-setup)
  - [Code Breakdown](#code-breakdown)
    - [1. URL Validation](#1-url-validation)
    - [2. Splitting Text into Chunks](#2-splitting-text-into-chunks)
    - [3. Creating FAISS Vector Store](#3-creating-faiss-vector-store)
    - [4. Wikipedia Tool](#4-wikipedia-tool)
    - [5. Website Retriever Tool](#5-website-retriever-tool)
    - [6. PDF Retriever Tool](#6-pdf-retriever-tool)
    - [7. Generating LLM Responses](#7-generating-llm-responses)
    - [8. Streamlit UI](#8-streamlit-ui)
  - [System Workflow](#system-workflow)
  - [How to Run the App](#how-to-run-the-app)

---

### Overview
This application allows users to ask questions and get answers from different data sources (Wikipedia, PDFs, and websites). It uses OpenAI's GPT-3.5 model to generate the answers and is built using Streamlit for the user interface.

The architecture includes:
- **Langchain** for data retrieval and handling various tools.
- **OpenAI's GPT-3.5** for generating responses.
- **FAISS** for efficient search over large textual datasets.
- **Streamlit** for creating a user-friendly web interface.

---

### Environment Setup
Before running the app, follow these steps to set up your environment:

1. Install the necessary dependencies:
   ```bash
   pip install langchain PyPDF2 streamlit faiss-cpu openai python-dotenv
   ```

2. Set up the environment variable for your OpenAI API key by creating a `.env` file:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

### Code Breakdown

#### 1. URL Validation
```python
def is_valid_url(url: str) -> bool:
    regex = r"((http|https)://)(www\.)?[a-zA-Z0-9@:%._\+~#?&//=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%._\+~#?&//=]*)"
    return bool(re.match(regex, url))
```
This function ensures the URL provided for website scraping is valid using a regular expression (regex).

#### 2. Splitting Text into Chunks
```python
def split_text_into_chunks(text: str, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)
```
This function splits long text into smaller chunks, improving processing efficiency. The `chunk_size` and `chunk_overlap` parameters help define the chunk dimensions.

#### 3. Creating FAISS Vector Store
```python
def create_vector_store(chunks: list) -> FAISS:
    embedding = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embedding)
```
Here, the FAISS vector store is created using OpenAI's embeddings. This allows the system to perform similarity searches efficiently based on chunks of text.

#### 4. Wikipedia Tool
```python
def wiki_tool() -> WikipediaQueryRun:
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
    return WikipediaQueryRun(api_wrapper=api_wrapper)
```
This tool uses the Wikipedia API wrapper to query Wikipedia and retrieve relevant content based on the user's question.

#### 5. Website Retriever Tool
```python
def website_tool(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    documents = split_text_into_chunks(docs)
    vectordb = create_vector_store(documents)
    retriever = vectordb.as_retriever()
    return create_retriever_tool(retriever, "world-best-cities", "Guide on best cities in the world")
```
This function scrapes content from a website, splits it into chunks, and creates a retriever to search through the website’s content.

#### 6. PDF Retriever Tool
```python
def get_pdf_text(pdf_docs) -> str:
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf, strict=False)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
```
This function extracts text from uploaded PDF documents.

```python
def pdf_retriever_tool(pdf_docs):
    pdf_text = get_pdf_text(pdf_docs)
    chunks = split_text_into_chunks(pdf_text)
    vector_db = create_vector_store(chunks)
    return create_retriever_tool(vector_db.as_retriever(), "pdf_tool", "PDF Content")
```
The PDF retriever tool uses the extracted text from PDF files, splits it into chunks, and creates a retriever to search the PDF content for relevant information.

#### 7. Generating LLM Responses
```python
def llm_response(user_question: str, tools: list, openai_api_key: str) -> str:
    gpt = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    my_agent = create_openai_tools_agent(gpt, tools, hub.pull("hwchase17/openai-functions-agent"))
    agent_executor = AgentExecutor(agent=my_agent, tools=tools, verbose=False)
    answer = agent_executor.invoke({"input": user_question})
    return answer['output']
```
This function generates responses using OpenAI’s GPT-3.5 model. It takes the user's question and applies the selected tools (Wikipedia, PDF, website) to provide a robust, data-driven answer.

#### 8. Streamlit UI
```python
def main():
    st.title("Question-Answering System")
    
    with st.sidebar:
        api_key_from_env = os.getenv('OPENAI_API_KEY')
        openai_api_key = api_key_from_env if api_key_from_env else st.text_input("Enter your OpenAI API Key:", type="password")
        st.subheader("More Options")
        sources = ["Wikipedia", "PDF", "Website"]
        selected_sources = st.multiselect("Select data sources:", options=sources, default=["Wikipedia"])
    
    user_question = st.text_input("Ask your question:")
    tools = []

    if "Wikipedia" in selected_sources:
        tools.append(wiki_tool())

    if "PDF" in selected_sources:
        uploaded_files = st.file_uploader("Upload a document (PDF):", type=['pdf'], accept_multiple_files=True)
        if uploaded_files:
            with st.spinner("Processing PDF..."):
                tools.append(pdf_retriever_tool(uploaded_files))

    if "Website" in selected_sources:
        url = st.text_input("Enter website URL:")
        if is_valid_url(url):
            tools.append(website_tool(url))

    if user_question and tools and openai_api_key:
        with st.spinner("Generating response..."):
            answer = llm_response(user_question, tools, openai_api_key)
            st.write(answer)
    else:
        st.warning("Please provide all inputs (question, data sources, and OpenAI API key).")
```
This function sets up the user interface using Streamlit. Users can select data sources, input their OpenAI API key, and submit a question. The app then returns an answer based on the chosen data sources.

---

### System Workflow

1. **User Inputs**: 
   - The user inputs their question and selects which sources they want to retrieve information from (Wikipedia, PDF, Website).
   - If a PDF is selected, the user can upload one or more PDF files.
   - If a website is selected, the user must enter a valid URL.
   
2. **Processing the Input**: 
   - The app validates the provided inputs (e.g., checking if the URL is valid).
   - Depending on the selected sources:
     - **Wikipedia**: The Wikipedia API is queried.
     - **PDFs**: Text from the uploaded PDFs is extracted, split into chunks, and stored in a FAISS vector store.
     - **Website**: The content of the website is scraped, split, and stored similarly.
   
3. **Retrieving Relevant Data**:
   - For each selected source, the relevant data is retrieved based on the user's query. The data is chunked and stored in a vector store using FAISS for efficient similarity searching.
   
4. **LLM Response Generation**:
   - Once the data from all sources is prepared, the LLM (GPT-3.5) generates a response to the user's query.
   - The agent created by Langchain invokes the tools corresponding to each source and combines the results.
   
5. **Displaying the Result**:
   - The result is displayed in the Streamlit interface for the user to review.

---

### How to Run the App
1. Clone the repository.
2. Install the required dependencies using the command: 
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your OpenAI API key.
4. Run the Streamlit app using the command:
   ```

bash
   streamlit run app.py
   ```