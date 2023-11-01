from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
import time


class DocumentBasedQA:
    """
    Document-Based Question Answering (Document-Based QA) is a Python class that enables question answering on documents
    using various methods and vector stores, including FAISS and Chroma. This class leverages Hugging Face embeddings, llm,
    and LangChain for document-based question answering.

    Attributes:
        documents (list): List of loaded documents.
        texts (list): List of split text documents.
        instructor_embeddings (HuggingFaceInstructEmbeddings): Hugging Face Instructor embeddings instance.
        vectorstore (FAISS or Chroma): Vector store instance for storing document vectors.
        retriever: RetrievalQA instance for answering questions.
        llm (HuggingFacePipeline): Hugging Face Language Model instance for text generation.
    """
    def __init__(self, document_url, embedder, llm, vector_store_type: str):
        """
        Initialize the DocumentBasedQA instance.

        Args:
            document_url (str): URL of the document to load.
            embedder (HuggingFaceInstructEmbeddings): Hugging Face Instructor embeddings instance.
            llm (HuggingFacePipeline): Hugging Face LLM instance for text generation.
            vector_store_type (str): Type of vector store, either "FAISS" or "Chroma".
        """
        # Load web document
        self.documents = self.load_document(document_url)  # TODO handle potential errors

        # Split the document
        self.texts = self.split_document()

        # Initialize HuggingFaceInstructEmbeddings
        self.instructor_embeddings = embedder

        # Initialize FAISS and Chroma vector stores
        self.vectorstore = None
        self.retriever = None

        # Set the Access Token for Hugging Face
        self.llm = llm

        if vector_store_type not in ["FAISS", "Chroma"]:
            raise ValueError("Invalid vector store. Use 'FAISS' or 'Chroma'.")

        self.initialize_vectorstore(vector_store_type)

    def load_document(self, document_url):
        """
        Load documents from a web source.

        Args:
            document_url (str): URL of the document to load.
        """
        loader = WebBaseLoader(document_url)
        return loader.load()

    def split_document(self):
        """
        Split the loaded documents into chunks.
        """
        start_time = time.time()  # Record the start time
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True
        )

        split_text = text_splitter.split_documents(self.documents)

        end_time = time.time()  # Record the end time
        document_split_time = end_time - start_time  # Calculate the time taken

        print(f"Time taken for document splitting: {document_split_time} seconds")
        return split_text

    def initialize_vectorstore(self, vector_store_type):
        """
        Initialize a vector store (FAISS or Chroma).

        Args:
            vector_store_type (str): The type of vector store to initialize.
        """
        print(f"Initializing {vector_store_type} vectorstore ...")

        if self.texts is not None and self.instructor_embeddings is not None:
            start_time = time.time()  # Record the start time
            if vector_store_type == "FAISS":
                self.vectorstore = FAISS.from_documents(self.texts, self.instructor_embeddings)
            elif vector_store_type == "Chroma":
                self.vectorstore = Chroma.from_documents(self.texts, self.instructor_embeddings)
            else:
                raise ValueError("Invalid vector store type. Use 'FAISS' or 'Chroma'.")

            end_time = time.time()  # Record the end time

            vectorstore_initialization_time = end_time - start_time  # Calculate the time taken

            print(f"Initialized {vector_store_type} vectorstore")
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            print(f"Time taken for vectorstore initialization: {vectorstore_initialization_time} seconds")

    def perform_qa(self, query: str, method: str) -> str:
        """
        Perform QA using the specified method and vector store.

        Args:
            query (str): Question to answer.
            method (str): "load_qa" or "RetrievalQA".
            vector_store_type (str): "FAISS" or "Chroma".

        Returns:
            str: Answer to the question.
        """

        print(f"Initializing {method} qa method...")
        if method == "load_qa":
            qa_chain = load_qa_chain(self.llm, chain_type="stuff")
            docs = self.retriever.get_relevant_documents(query)
            print(f"Retrieving answer for the question: {query}")
            answer = qa_chain.run(input_documents=docs, question=query)
        elif method == "RetrievalQA":
            qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                                   chain_type="stuff",
                                                   retriever=self.retriever,
                                                   return_source_documents=True)
            print(f"Retrieving answer for the question: {query}")
            answer = qa_chain(query)['result']
        else:
            raise ValueError("Invalid method. Use 'load_qa' or 'RetrievalQA'.")
        print(f"Answer: {answer}")
        return answer
