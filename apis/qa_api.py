# qa_api.py
from fastapi import FastAPI, Form, Query
from pydantic import BaseModel
from enum import Enum
from src.question_answering import QuestionAnswering
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
# Load environment variables from the .env file
load_dotenv()

qa_app = FastAPI()

question_answering = None


class MethodEnum(str, Enum):
    load_qa = "load_qa"
    RetrievalQA = "RetrievalQA"


class QuestionResponse(BaseModel):
    answer: str


embedding = "instructor"


@qa_app.on_event("startup")
async def startup_event():
    try:
        # Initialize the HuggingFaceEmbeddings
        if embedding == "instructor":
            qa_app.state.embedder = HuggingFaceInstructEmbeddings(model_name="../models/instructor-base",
                                                                  model_kwargs={"device": "cpu"})  # cuda
        else:
            qa_app.state.embedder = HuggingFaceEmbeddings(
                model_name="../models/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
    except Exception as e:
        raise e  # Reraise the exception for FastAPI to handle

    qa_app.state.llm = HuggingFacePipeline.from_model_id(
        model_id="../models/flan-t5-large-instruct-dolly_hhrlhf",
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 512},
    )


class VectorStoreEnum(str, Enum):
    FAISS = "FAISS"
    Chroma = "Chroma"


@qa_app.post("/ask_question/", response_model=QuestionResponse)
def ask_question(query: str = Form(...), method: MethodEnum = Query(..., description="Method for question answering"),
                 vector_store_path: str = Query(None, description="Path to the vector store"),
                 vector_store_type: VectorStoreEnum = Query(..., description="VectorStore for question answering")
                 ):
    # load vector store
    question_answering = QuestionAnswering(qa_app.state.llm,
                                           vector_store_path=vector_store_path,
                                           embedder=qa_app.state.embedder,
                                           vector_store_type=vector_store_type)
    answer = question_answering.perform_qa(query, method)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(qa_app, host="0.0.0.0", port=8001)
