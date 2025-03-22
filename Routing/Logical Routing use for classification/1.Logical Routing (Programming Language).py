
# Few Shot Examples
#from tkinter.scrolledtext import example
import os

from accelerate.commands.config.update import description
from dotenv import load_dotenv
from typing import Literal
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from torch.cuda import temperature
from langchain_core.runnables import RunnableLambda
load_dotenv()


LANGSMITH_TRACING=os.environ['LANGSMITH_TRACING']
LANGSMITH_ENDPOINT=os.environ['LANGSMITH_ENDPOINT']
LANGSMITH_API_KEY=os.environ['LANGSMITH_API_KEY']
LANGSMITH_PROJECT=os.environ['LANGSMITH_PROJECT']
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
URI=os.environ['URI']



question = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

# question = """
#
# import { config } from 'dotenv';
# import OpenAI from 'openai';
#
# config(); // Load env variables
#
# const openai = new OpenAI({
#   apiKey: process.env.OPENAI_API_KEY
# });
# """

#question = "how to prepare biryani"

class RouterQuery(BaseModel):
    datasource: Literal["python", "js", "golang","marium","neb++","Non-relevant"] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question"
    )



Prompt = ChatPromptTemplate.from_messages(
    [
        ("system","""You are an expert at routing a user question to the appropriate data source.Based on the programming language the question is referring to, route it to the relevant data source."""),
        ("human","{question}")
    ]
)

llm = ChatOpenAI(temperature = 0)
structured_output = llm.with_structured_output(RouterQuery)
router =  Prompt | structured_output

result = router.invoke({"question": question})

print("result",result)


def choose_route(result):
    if "python" in result.datasource.lower():
        ### Logic here
        return "chain for python_docs"
    elif "js" in result.datasource.lower():
        ### Logic here
        return "chain for js_docs"
    elif "golang" in result.datasource.lower():
        ### Logic here
        return "golang_docs"
    elif "Non-relevant" in result.datasource.lower():
        return "Sorry i cannot answer this question"


print(choose_route(result))
