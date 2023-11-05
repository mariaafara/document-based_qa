from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
import time


class DocumentIndexer:
    def __init__(self, document_url, embedder, vector_store_type: str):
        self.documents = self.load_document(document_url)
        self.texts = self.split_document()
        self.instructor_embeddings = embedder
        self.vectorstore = None

        self.vector_store_type = vector_store_type
        if self.vector_store_type not in ["FAISS", "Chroma"]:
            raise ValueError("Invalid vector store. Use 'FAISS' or 'Chroma'.")

        self.vector_store_path = self.initialize_vectorstore(self.vector_store_type)

    def load_document(self, document_url):
        loader = WebBaseLoader(document_url)
        return loader.load()

    def split_document(self):
        start_time = time.time()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True
        )

        split_text = text_splitter.split_documents(self.documents)

        end_time = time.time()
        document_split_time = end_time - start_time
        print(f"Time taken for document splitting: {document_split_time} seconds")
        return split_text

    def initialize_vectorstore(self, vector_store_type):
        print(f"Initializing {vector_store_type} vectorstore ...")

        if self.texts is not None and self.instructor_embeddings is not None:
            start_time = time.time()
            if vector_store_type == "FAISS":
                self.vectorstore = FAISS.from_documents(self.texts, self.instructor_embeddings)
            elif vector_store_type == "Chroma":
                self.vectorstore = Chroma.from_documents(self.texts, self.instructor_embeddings)
            else:
                raise ValueError("Invalid vector store type. Use 'FAISS' or 'Chroma'.")

            end_time = time.time()
            vectorstore_initialization_time = end_time - start_time

            print(f"Initialized {vector_store_type} vectorstore")
            print(f"Time taken for vectorstore initialization: {vectorstore_initialization_time} seconds")
            vector_store_path = "vector_store"
            self.vectorstore.save_local(vector_store_path)
            return vector_store_path

    def get_vectorstore(self):
        return self.vectorstore
