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
from database import return_connection

#con = return_connection(db_name="milvus_demo")


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



retriever = vector_store.as_retriever()
# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
print(rag_chain.invoke("What is Task Decomposition?"))


