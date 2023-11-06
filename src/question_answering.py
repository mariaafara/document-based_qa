from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

from langchain.vectorstores import FAISS, Chroma


class QuestionAnswering:
    def __init__(self, llm, vector_store_type, vector_store_path, embedder):
        self.llm = llm
        if vector_store_type == "FAISS":
            vector_store = FAISS.load_local(vector_store_path, embedder)
        else:
            vector_store = Chroma(embedding_function=embedder, persist_directory=vector_store_path)

        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def perform_qa(self, query: str, method: str) -> str:
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
