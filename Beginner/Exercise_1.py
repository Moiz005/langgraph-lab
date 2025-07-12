#A minimal LangGraph example that defines a single node to generate a personalized greeting using agent state.

from typing import Dict, TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
  message: str

def greeting_node(state: AgentState) -> AgentState :
  """A simple node that adds a greeting message to the state"""
  state["message"] = "Hey " + state["message"] + ", how is your day going?"
  return state

graph = StateGraph(AgentState)

graph.add_node("greeter", greeting_node)

graph.set_entry_point("greeter")
graph.set_finish_point("greeter")

app = graph.compile()

result = app.invoke({"message": "Bob"})

print(result["message"])
