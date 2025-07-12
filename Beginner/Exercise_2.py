# A LangGraph program that performs basic arithmetic (sum or product) on a list of numbers and returns a personalized result message.

from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph
import math

class AgentState(TypedDict):
  values: List[int]
  name: str
  operation: str
  result: str

def process_elements(state: AgentState) -> AgentState :
  if state["operation"]:
    if state["operation"] == "+":
      state["result"] = f"Hi {state['name']}, your answer is {str(sum(state['values']))}"
    elif state["operation"] == "*":
      state["result"] = f"Hi {state['name']}, your answer is {str(math.prod(state['values']))}"

  return state

graph = StateGraph(AgentState)

graph.add_node("process", process_elements)

graph.set_entry_point("process")
graph.set_finish_point("process")

app = graph.compile()

answer = app.invoke({"values": [1,2,3,4], "name": "Moiz", "operation": "+"})

print(answer["result"])