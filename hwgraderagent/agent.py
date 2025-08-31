from typing import Annotated
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
import base64

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]

#os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
load_dotenv()
#os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
graph_builder = StateGraph(State)


llm = init_chat_model("google_genai:gemini-2.0-flash")


def identify_query_content(user_input: str):

    json_data = {"messages": [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": """
                You are an AI agent with very good understanding of the mathematical concepts like addition, subtraction, and multiplication. You are provided with a assignment answer sheet from a second grade student. Your tasks are the following
                1. Solve all the problems and compare with the student provided answers.
                2. If students answer doesn't match with your answers, you need to provide feedback to the student so he can improve.

               Your response should be in the following format:  
                1. For each of the question, give the question, followed by your answer and student provided answer. If your answer and student's answer are the same, say it is CORRECT else say WRONG
                2. Percentage of CORRECT answers  to the total questions e.g. 80% (16/20)
                3. Summary of the grading in two sentences highlighting what kind of questions the student is good at and what kind of questions the student needs to further improve to get all correct. 
                """,
            }
        ],
    }]}
    for event in graph.stream(json_data):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def stream_graph_updates(user_input: str):
    with open(user_input, "rb") as image_file:
        image_data = base64.b64encode(image_file.read())
        json_data = {"messages":[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
                    You are an AI agent with very good understanding of the mathematical concepts like addition, subtraction, and multiplication. You are provided with a assignment answer sheet from a second grade student. Your tasks are the following
                    1. Solve all the problems and compare with the student provided answers.
                    2. If students answer doesn't match with your answers, you need to provide feedback to the student so he can improve.
                   
                   Your response should be in the following format:  
                    1. For each of the question, give the question, followed by your answer and student provided answer. If your answer and student's answer are the same, say it is CORRECT else say WRONG
                    2. Percentage of CORRECT answers  to the total questions e.g. 80% (16/20)
                    3. Summary of the grading in two sentences highlighting what kind of questions the student is good at and what kind of questions the student needs to further improve to get all correct. 
                    """,
                },
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": image_data,
                    "mime_type": "application/pdf",
                }
            ],
        }]}
        for event in graph.stream(json_data):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)



def query_parser(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def image_grader(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", query_parser)
graph_builder.add_node("image_grader", image_grader )
graph_builder.add_edge(START, "query_parser")
graph_builder.add_edge( "query_parser","image_grader")
graph_builder.add_edge("image_grader", END)
graph = graph_builder.compile()
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break