import os
from dotenv import load_dotenv
import bs4
from langchain import hub
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

from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field



load_dotenv()


LANGSMITH_TRACING=os.environ['LANGSMITH_TRACING']
LANGSMITH_ENDPOINT=os.environ['LANGSMITH_ENDPOINT']
LANGSMITH_API_KEY=os.environ['LANGSMITH_API_KEY']
LANGSMITH_PROJECT=os.environ['LANGSMITH_PROJECT']
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
URI=os.environ['URI']



# # Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

#print(json.dumps(docs[0].dict(), indent=4, ensure_ascii=False))

#%%
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


embeddings = OpenAIEmbeddings()

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI,"db_name": "milvus_demo"},
collection_name="splits",
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)

uuids = [str(uuid4()) for _ in range(len(splits))]
vector_store.add_documents(documents=splits, ids=uuids)


# Output parser will split the LLM result into a list of queries
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


output_parser = LineListOutputParser()


QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
llm = ChatOpenAI(temperature=0)



# Chain
llm_chain = QUERY_PROMPT | llm | output_parser

# Other inputs
question = "What are the approaches to Task Decomposition?"

# Run
retriever = MultiQueryRetriever(
    retriever=vector_store.as_retriever(), llm_chain=llm_chain, parser_key="lines"
)  # "lines" is the key (attribute name) of the parsed output



# Set logging for the queries
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# Results
unique_docs = retriever.invoke("What does the course say about regression?")
