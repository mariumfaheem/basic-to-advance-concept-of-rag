from accelerate.commands.config.update import description
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field
from torch.cuda import temperature
from langchain_openai import OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY=os.environ['OPENAI_API_KEY']


class RouteQuery(BaseModel):
    datasource: Literal["anthropic","langchain","other"] = Field(
        ...,
        description="Given a user question choose which llm model would be most relevant for answering their question"
    )

def route_to_chain(route_name):
    print(route_name.datasource)
    if "anthropic" == route_name.datasource.lower():
        return "anthropic_chain"
    elif "langchain" == route_name.datasource.lower():
        print("in llm chain")
        return "langchain_chain"
    elif "other" == route_name.datasource.lower():
        return "general_chain"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are an expert at routing a user question to the appropriate llm model.Based on the LLM model the question is referring to, route it to the relevant data source"""),
        ("user","{question}")
        ]
)


question = "how can i use function LangChain text_splitter?"
llm = ChatOpenAI(temperature=0)
structured_llm = llm.with_structured_output(RouteQuery)
router = prompt | structured_llm

result = router.invoke({"question":question})
print(route_to_chain(result))








