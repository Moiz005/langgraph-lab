# A simple LangGraph chatbot using OpenAI's GPT-4o that processes user input in a loop and returns AI-generated responses.

from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import os

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

class AgentState(TypedDict):
  messages:List[HumanMessage]

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
  response = llm.invoke(state['messages'])
  print(f"\nAI: {response.content}")
  return state

graph = StateGraph(AgentState)

graph.add_node('process', process)
graph.add_edge(START, 'process')
graph.add_edge('process', END)
agent = graph.compile()

user_input = input("Enter: ")
while user_input != 'exit':
  agent.invoke({'messages': [HumanMessage(content=user_input)]})
  user_input = input("Enter: ")