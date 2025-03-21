from semantic_router import Route
#from semantic_router import RouteLayer
from semantic_router.routers import SemanticRouter

from semantic_router.encoders import OpenAIEncoder
import os
from dotenv import load_dotenv





load_dotenv()



TOP_K = 2
MAX_DOCS_FOR_CONTEXT = 3

LANGSMITH_TRACING=os.environ['LANGSMITH_TRACING']
LANGSMITH_ENDPOINT=os.environ['LANGSMITH_ENDPOINT']
LANGSMITH_API_KEY=os.environ['LANGSMITH_API_KEY']
LANGSMITH_PROJECT=os.environ['LANGSMITH_PROJECT']
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
URI=os.environ['URI']




if __name__ == '__main__':

    politics = Route(
        name="politics",
    utterances=[
    "isn't politics the best thing ever",
            "why don't you tell me about your political opinions",
            "don't you just love the president",
            "they're going to destroy this country!",
            "they will save the country!",
    ]
    )


    # this could be used as an indicator to our chatbot to switch to a more
    # conversational prompt
    chitchat = Route(
        name="chitchat",
        utterances=[
            "how's the weather today?",
            "how are things going?",
            "lovely weather today",
            "the weather is horrendous",
            "let's go to the chippy",
        ],
    )

    routes = [politics, chitchat]


    encoder = OpenAIEncoder()


    route_layer = SemanticRouter(encoder=encoder, routes=routes,auto_sync="local")


    print(route_layer("don't you love politics?"))

    print(route_layer("how's the weather today?"))