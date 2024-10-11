from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.globals import set_verbose, set_debug
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

set_debug(True)
set_verbose(True)


class ChatRAG:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self, llm_model: str = "llama3.2"):
        self.model = ChatOllama(model=llm_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100
        )
        self.prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    "You are a helpful assistant that can answer questions about the document that the user uploaded.",
                ),
                (
                    "human",
                    'Here are the document pieces: "{context}"\nQuestion: "{question}"',
                ),
            ]
        )

        self.vector_store = None
        self.retriever = None
        self.chain = None

    def ingest_pdf(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=FastEmbedEmbeddings(),
            persist_directory="chroma_db",
        )

    def ingest_csv(self, csv_file_path: str):
        loader = CSVLoader(file_path=csv_file_path, autodetect_encoding=True)
        data = loader.load()

        self.vector_store = Chroma.from_documents(
            documents=data,
            embedding=FastEmbedEmbeddings(),
            persist_directory="chroma_db",
        )

    def ask(self, query: str):
        if not self.vector_store:
            self.vector_store = Chroma(
                persist_directory="chroma_db", embedding=FastEmbedEmbeddings()
            )

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 10, "score_threshold": 0.0},
        )

        self.retriever.invoke(query)

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

        if not self.chain:
            return "Please, add a document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
