from langchain_community.llms.openai import OpenAIChat
from semantic_router import Route
#from semantic_router import RouteLayer
from semantic_router.routers import SemanticRouter
from langchain.utils.math import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from semantic_router.encoders import OpenAIEncoder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import os
from dotenv import load_dotenv
from torch.cuda import temperature
from langchain_openai import OpenAIEmbeddings
load_dotenv()

OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
embeddings = OpenAIEmbeddings()
from langchain_core.prompts import PromptTemplate


physics_prompt = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{question}"""

maths_prompt = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{question}"""


prompt_templates = [maths_prompt, physics_prompt]
prompt_embeddings = embeddings.embed_documents(prompt_templates)


def prompt_router(question):
    # Embed prompts
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(question["question"])
    similarity = cosine_similarity([query_embedding],  prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]

    # Chosen prompt
    print("Using MATH" if most_similar == maths_prompt else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)


llm = ChatOpenAI(temperature=0)
chain = prompt_router | llm

user_input = {"question" :"What's a black hole"}
chain = (
    RunnableLambda(prompt_router)
    | llm
    | StrOutputParser()
)
result = chain.invoke(user_input)
print(result)

