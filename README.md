# Document-Based Question Answering (Document-Based QA)

## Overview

Document-Based QA is a Python project that allows you to perform question answering on documents using various methods
and vector stores, including FAISS and Chroma. This project uses local Hugging Face embeddings, llm, and LangChain for
document-based question answering.

## Installation

1. Clone the project repository:

   ```bash
   git clone https://github.com/mariaafara/document-based-qa.git
   cd document-based-qa
   ```

   ```bash
   pip install -r requirements.txt
   ```

2. Install models locally
 
   ```bash
   mkdir models
   cd models/
   ```

   ```bash
   git clone https://huggingface.co/google/mt5-small
   ```

   ```bash
   git clone https://huggingface.co/pszemraj/flan-t5-large-instruct-dolly_hhrlhf
   ```

3. Lunch the API

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

# API Endpoints

- /upload_document/: Upload a document from a URL and initialize the Document-Based QA instance with the chosen vector store.

      Query Parameter:
         document_url: URL of the document to load.
         vector_store_type: Choose between "FAISS" or "Chroma" for the vector store.

- /ask_question/: Ask a question on the loaded document and get the answer.

      Form Data:
         query: The question you want to ask.
         method: Choose between "load_qa" or "RetrievalQA" for the question-answering method.