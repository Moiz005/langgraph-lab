# A conversational LangGraph chatbot with memory that maintains full dialogue history, logs the interaction to a file, and uses GPT-4o for responses.

import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

llm = ChatOpenAI(model='gpt-4o')

class AgentState(TypedDict):
  messages: List[Union[HumanMessage, AIMessage]]

def process(state:AgentState) -> AgentState:
  response = llm.invoke(state['messages'])
  state['messages'].append(AIMessage(content=response.content))
  print(f"\nAI: {response.content}")
  return state

graph = StateGraph(AgentState)

graph.add_node('process', process)
graph.add_edge(START, 'process')
graph.add_edge('process', END)
agent = graph.compile()

conversation_history = []

user_input = input("Enter: ")
while user_input != 'exit':
  conversation_history.append(HumanMessage(content=user_input))
  result = agent.invoke({'messages': conversation_history})

  print(result['messages'])
  conversation_history = result['messages']

  user_input = input("Enter: ")

  with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    for message in conversation_history:
      if isinstance(message, HumanMessage):
        file.write(f"You: {message.content}\n")
      elif isinstance(message, AIMessage):
        file.write(f"AI: {message.content}\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")
