#Reference : https://github.com/kzhisa/rag-fusion/blob/main/rag_fusion.py

import os
from dotenv import load_dotenv
import bs4
import argparse
import os
import sys
from operator import itemgetter

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




def load_and_split_document(DOCUMENT_URL):
    # # Load Documents
    loader = WebBaseLoader(
        web_paths=(DOCUMENT_URL,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # for TEST
    print("Original document: ", len(splits), " docs")

    return splits


def create_retriever(DOCUMENT_URL="https://lilianweng.github.io/posts/2023-06-23-agent/"):
    splits = load_and_split_document(DOCUMENT_URL)
    embeddings = OpenAIEmbeddings()

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI,"db_name": "milvus_demo"},
        collection_name="decomposition",
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )

    uuids = [str(uuid4()) for _ in range(len(splits))]
    vector_store.add_documents(documents=splits, ids=uuids)

    # retriever
    retriever = vector_store.as_retriever()

    return retriever


def query_generator(original_query):
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    query = original_query.get("query")
    print("Original Query:", query)

    class LineListOutputParser(BaseOutputParser[List[str]]):
        """Output parser for a list of lines."""

        def parse(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            return list(filter(None, lines))  # Remove empty lines

    output_parser = LineListOutputParser()


    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are a helpful assistant that generates multiple sub-questions related to an input question. \n
        The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
        Generate multiple search queries related to: {question} \n
        Output (3 queries):""",
    )

    llm = ChatOpenAI(temperature=0)

    # Chain
    llm_chain = QUERY_PROMPT | llm | output_parser

    queries = llm_chain.invoke({"question": query})

    print("queries",queries)

    # Add original query
    #queries.insert(0, "0. " + query)


    # Print for debugging
    print('Generated queries:\n', '\n'.join(queries))

    return queries



def retrieve_and_rag(question,prompt_rag,sub_question_generator_chain,llm):
    sub_questions = sub_question_generator_chain.invoke({"query": question})

    rag_result = []

    for sub_question in sub_questions:
        retrieved_docs = retriever.get_relevant_documents(sub_question)

        chain = prompt_rag | llm | StrOutputParser()

        answer = chain.invoke({"context": retrieved_docs,
                "question": sub_question})

        rag_result.append(answer)
    return rag_result,sub_questions

def retriever(query):
    """RRF retriever"""
    retriever = create_retriever()  # Vector DB
    generated_queries = query_generator(query)


    retrieved_docs = [retriever.invoke(q) for q in generated_queries]

    prompt_rag = """
    Here is a question: 
    {question}

    and here is context:

    {context}

    Please answer, and if you don't know, say you have no idea.
    """
    decomposition_prompt = PromptTemplate(
        template=prompt_rag,
        input_variables=["context", "question"],
    )

    llm = ChatOpenAI(temperature=0)

    # Chain
    generate_queries_decomposition = ({"context": itemgetter("question") | retriever,
                                          "question": itemgetter("question")},
            decomposition_prompt | llm | StrOutputParser() | (lambda x: x.split("\n")))


    # Run the retrieval and RAG pipeline
    answers, questions = retrieve_and_rag(generated_queries, prompt_rag, generate_queries_decomposition, llm)

    print("Result of questions and answers:", answers, questions)

    return answers


def final_answer_generate(questions, retriever: BaseRetriever):
    """
    Retrieves relevant documents using `retriever` and generates an AI response.
    """
    # Define the prompt object

    # Prompt
    template = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question:

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """

    decomposition_prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )

    def format_qa_pair(question, answer):
        """Format Q and A pair"""
        formatted_string = ""
        formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
        return formatted_string.strip()

    # Initialize OpenAI model
    llm = ChatOpenAI(temperature=0)
    q_a_pairs = ""
    for q in questions:
        rag_chain = (
                {"context": itemgetter("question") | retriever,
                 "question": itemgetter("question"),
                 "q_a_pairs": itemgetter("q_a_pairs")}
                | decomposition_prompt
                | llm
                | StrOutputParser())

        answer = rag_chain.invoke({"question": q, "q_a_pairs": q_a_pairs})
        q_a_pair = format_qa_pair(q, answer)
        q_a_pairs = q_a_pairs + "\n---\n" + q_a_pair

    return answer




if __name__ == '__main__':
    # Step 1: Define the original question
    original_query = {"query": "What are the main components of an LLM-powered autonomous agent system?"}

    # Step 2: Generate sub-questions
    questions = query_generator(original_query)  # Generates list of sub-queries

    #print(retriever(original_query))


    # Step 3: Create a retriever
    retriever_instance = create_retriever()  # Initializes a vector database retriever

    # Step 4: Generate final answers using decomposition RAG
    print(final_answer_generate(questions, retriever_instance))

