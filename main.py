# main.py
import logging
from fastapi import FastAPI, Form, Query
from pydantic import BaseModel
from document_based_qa import DocumentBasedQA
from langchain.llms import HuggingFacePipeline

from enum import Enum
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
import time


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
            app.state.embedder = HuggingFaceInstructEmbeddings(model_name="models/instructor-base",
                                                               model_kwargs={"device": "cpu"})  # cuda
        else:
            app.state.embedder = HuggingFaceEmbeddings(
                model_name="models/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
        logger.info("Embedder loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedder: {str(e)}")
        raise e  # Reraise the exception for FastAPI to handle

    try:
        logger.info("Loading llm... ")
        # Initialize the HuggingFacePipeline
        app.state.llm = HuggingFacePipeline.from_model_id(
            model_id="models/flan-t5-large-instruct-dolly_hhrlhf",
            task="text2text-generation",
            pipeline_kwargs={"max_new_tokens": 512},
        )
        logger.info("LLM loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load llm: {str(e)}")
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


class QuestionResponse(BaseModel):
    answer: str


document_qa: DocumentBasedQA = None


# async def initialize_document_qa(document_url, vector_store_type: str):
#     global document_qa
#     document_qa = DocumentBasedQA(document_url, app.state.embedder, app.state.llm, vector_store_type)

async def initialize_document_qa(document_url, vector_store_type: str):
    global document_qa
    try:
        document_qa = DocumentBasedQA(document_url, app.state.embedder, app.state.llm, vector_store_type)
        if document_qa.texts:
            return {"message": "Initialization successful.",
                    "NB chunks": f"The doc is split into {len(document_qa.texts)} chunks."}
        else:
            return {"message": "Initialization successful, but no chunks found in the document."}
    except Exception as e:
        return {"message": f"Initialization failed: {str(e)}"}


# @app.post("/upload_document/", response_model=DocumentUploadResponse)
# async def upload_document(document_url: str,
#                           vector_store_type: VectorStoreEnum = Query(...,
#                                                                      description="VectorStore for question answering")):
#     await initialize_document_qa(document_url, vector_store_type)
#     return {"NB chunks": f"The doc is split into {len(document_qa.texts)} chunks."}

@app.post("/upload_document/", response_model=DocumentUploadResponse)
async def upload_document(document_url: str,
                          vector_store_type: VectorStoreEnum = Query(...,
                                                                     description="VectorStore for question answering")):
    response = await initialize_document_qa(document_url, vector_store_type)
    return DocumentUploadResponse(message=response.get("message", ""),
                                  nb_chunks=len(document_qa.texts) if document_qa.texts else 0)


def perform_question(query, method):
    if document_qa is not None:
        start_time = time.time()  # Record the start time

        answer = document_qa.perform_qa(query, method)

        end_time = time.time()  # Record the end time

        question_answering_time = end_time - start_time  # Calculate the time taken
        print(f"Time taken for answering : {question_answering_time} seconds")
        return {"answer": answer}
    else:
        return {"answer": "No document has been uploaded yet."}


@app.post("/ask_question/", response_model=QuestionResponse)
def ask_question(query: str = Form(...),
                 method: MethodEnum = Query(..., description="Method for question answering")
                 ):
    response = perform_question(query, method)
    return response
