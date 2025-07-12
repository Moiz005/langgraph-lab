# A multi-node LangGraph workflow that builds a personalized message by progressively adding user info like name, age, and skills.

from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph

class AgentState(TypedDict):
  name: str
  age: int
  skills: List[str]
  result: str

def first_node(state: AgentState) -> AgentState:
  """This is my first node"""
  state["result"] = f"{state['name']}, welcome to the system!"
  return state

def second_node(state: AgentState) -> AgentState:
  """This is my second node"""
  state["result"] = f"{state['result']} You are {str(state['age'])} years old!"
  return state

def third_node(state: AgentState) -> AgentState:
  """This is my third node"""
  state["result"] = f"{state['result']} You have skills in: {', '.join(state['skills'])}"
  return state

graph = StateGraph(AgentState)

graph.add_node("first", first_node)
graph.add_node("second", second_node)
graph.add_node("third", third_node)

graph.set_entry_point("first")
graph.add_edge("first", "second")
graph.add_edge("second", "third")
graph.set_finish_point("third")

app = graph.compile()

answer = app.invoke({'name': 'Moiz', 'age': 22, 'skills': ['Python', 'C++', "JavaScript"]})
print(answer["result"])