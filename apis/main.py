# main.py
import logging
from fastapi import FastAPI, Form, Query
from pydantic import BaseModel
from src.document_indexing import DocumentIndexer
from enum import Enum
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import httpx

app = FastAPI()

logger = logging.getLogger("DocQA")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S')
embedding = "instructor"


@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Loading embedder...")
        # Initialize the HuggingFaceEmbeddings
        if embedding == "instructor":
            app.state.embedder = HuggingFaceInstructEmbeddings(model_name="../models/instructor-base",
                                                               model_kwargs={"device": "cpu"})  # cuda
        else:
            app.state.embedder = HuggingFaceEmbeddings(
                model_name="../models/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
        logger.info("Embedder loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedder: {str(e)}")
        raise e  # Reraise the exception for FastAPI to handle


class MethodEnum(str, Enum):
    load_qa = "load_qa"
    RetrievalQA = "RetrievalQA"


class VectorStoreEnum(str, Enum):
    FAISS = "FAISS"
    Chroma = "Chroma"


class DocumentUploadResponse(BaseModel):
    message: str
    nb_chunks: int


async def initialize_document_qa(document_url, vector_store_type: str):
    global document_indexer
    try:
        document_indexer = DocumentIndexer(document_url, app.state.embedder, vector_store_type)
        if document_indexer.texts:
            return {"message": "Initialization successful.",
                    "NB chunks": f"The doc is split into {len(document_indexer.texts)} chunks."}
        else:
            return {"message": "Initialization successful, but no chunks found in the document."}
    except Exception as e:
        return {"message": f"Initialization failed: {str(e)}"}


@app.post("/upload_document/", response_model=DocumentUploadResponse)
async def upload_document(document_url: str,
                          vector_store_type: VectorStoreEnum = Query(...,
                                                                     description="VectorStore for question answering")):
    response = await initialize_document_qa(document_url, vector_store_type)
    return DocumentUploadResponse(message=response.get("message", ""),
                                  nb_chunks=len(document_indexer.texts) if document_indexer.texts else 0)


async def ask_question_to_qa_api(query: str, method: MethodEnum, vector_store_path: str,
                                 vector_store_type: VectorStoreEnum):
    async with httpx.AsyncClient() as client:
        response = await client.post("https://0.0.0.0/8001", json={"query": query, "method": method,
                                                                  "vector_store_path": vector_store_path,
                                                                  "vector_store_type": vector_store_type,
                                                                  })
        return response.json()


@app.post("/ask_question/")
async def ask_question(query: str = Form(...), method: MethodEnum = Query(..., description="Method for question answering"),
                 ):
    response = await ask_question_to_qa_api(query, method, document_indexer.vector_store_path,
                                      document_indexer.vector_store_type)
    return response
