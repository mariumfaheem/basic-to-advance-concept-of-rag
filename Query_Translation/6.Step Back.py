# Few Shot Examples
#from tkinter.scrolledtext import example
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
load_dotenv()


LANGSMITH_TRACING=os.environ['LANGSMITH_TRACING']
LANGSMITH_ENDPOINT=os.environ['LANGSMITH_ENDPOINT']
LANGSMITH_API_KEY=os.environ['LANGSMITH_API_KEY']
LANGSMITH_PROJECT=os.environ['LANGSMITH_PROJECT']
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
URI=os.environ['URI']

step_back_vague_questions_examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
    {
        "input":"can AI help a medicical industry by analysing medical reports instead of doctors",
        "output":"can AI help medical industry"
    }
]


example_how_AI_generate_prompt =ChatPromptTemplate.from_messages(
    [("human", "{input}"),
     ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt = example_how_AI_generate_prompt,
    examples = step_back_vague_questions_examples,
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{question}"),
    ]
)

llm =ChatOpenAI(temperature=0)
generate_queries_step_back = prompt | llm | StrOutputParser()

#generate_queries_step_back.invoke({"question": question})
#print("generated_better_answer_by_llm",generated_better_answer_by_llm)

question = "i heard if i go to karachi, i will die"
answer = generate_queries_step_back.invoke({"question": question})
print("answer",answer)










