from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY=os.environ['OPENAI_API_KEY']

def route_to_chain(route_name):
    if "anthropic" == route_name.lower():
        return "anthropic_chain"
    elif "langchain" == route_name.lower():
        return "langchain_chain"
    else:
        return "general_chain"


prompt = PromptTemplate.from_template("""
Given the user question below, classify it as either
being about `LangChain`, `Anthropic`, or `Other`.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:""")

user_query = "how can i use function LangChain text_splitter?"

llm = ChatOpenAI(temperature=0)

llm_completion_select_route_chain =prompt |  llm | StrOutputParser()
route_name = llm_completion_select_route_chain.invoke(user_query)


chain = route_to_chain(route_name)

print(chain)





