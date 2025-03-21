import os
from dotenv import load_dotenv
import bs4
import argparse
import os
import sys
from operator import itemgetter
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

from dotenv import load_dotenv
from langchain.load import dumps, loads
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter
from langchain import hub
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from uuid import uuid4
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from typing import List

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import logging




load_dotenv()
LANGSMITH_TRACING=os.environ['LANGSMITH_TRACING']
LANGSMITH_ENDPOINT=os.environ['LANGSMITH_ENDPOINT']
LANGSMITH_API_KEY=os.environ['LANGSMITH_API_KEY']
LANGSMITH_PROJECT=os.environ['LANGSMITH_PROJECT']
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
URI=os.environ['URI']




from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def loading_documents():
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()

    loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
    docs.extend(loader.load())

    return docs


def extract_page_content(doc):
    return {"doc": doc.page_content}



def create_retriever():
    summaries,docs = generating_summaries()
    embeddings = OpenAIEmbeddings()

    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI,"db_name": "milvus_demo"},
        collection_name="summaries",
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )

    store = InMemoryByteStore()
    id_key = "doc_id"

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )


    doc_ids = [str(uuid4()) for _ in range(len(docs))]

    # Docs linked to summaries
    summary_docs = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(summaries)
    ]

    retriever.vectorstore.add_documents(documents=summary_docs, ids=doc_ids)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    return retriever,vectorstore

def generating_summaries():
    docs = loading_documents()

    # Convert Documents into a list of dictionaries
    mapped_docs = [extract_page_content(doc) for doc in docs]

    llm = ChatOpenAI(model="gpt-3.5-turbo", max_retries=0)
    prompt = ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")

    chain = (
                RunnableLambda( lambda x: {"doc": x})
            | prompt
        | llm
        | StrOutputParser()
    )

    summaries = chain.batch(mapped_docs, {"max_concurrency": 5})

    print("summaries", summaries)
    return summaries,docs

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    create_retriever()




