# A conditional LangGraph workflow that routes through different arithmetic operations (+ or -) based on user input, handling two sequential expressions.

from typing import TypedDict, Dict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
  num1: int
  op1: str
  num2: int
  num3: int
  op2: str
  num4: int
  result1: int
  result2: int

def add_node(state: AgentState) -> AgentState:
  state['result1'] = state['num1'] + state['num2']
  return state

def subtract_node(state: AgentState) -> AgentState:
  state['result1'] = state['num1'] - state['num2']
  return state

def router(state: AgentState) -> AgentState:
  if state['op1'] == '+':
    return "addition_operation"
  elif state['op1'] == '-':
    return "subtraction_operation"

def add_node2(state: AgentState) -> AgentState:
  state['result2'] = state['num3'] + state['num4']
  return state

def subtract_node2(state: AgentState) -> AgentState:
  state['result2'] = state['num3'] - state['num4']

def router2(state: AgentState) -> AgentState:
  if state['op2'] == '+':
    return "addition_operation2"
  elif state['op2'] == '-':
    return "subtraction_operation2"

graph = StateGraph(AgentState)

graph.add_node("add_node", add_node)
graph.add_node("subtract_node", subtract_node)

graph.add_node("add_node2", add_node2)
graph.add_node("subtract_node2", subtract_node2)

graph.add_node("router", lambda state: state)

graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    router,
    {
        "addition_operation": "add_node",
        "subtraction_operation": "subtract_node"
    }
)

graph.add_node("router2", lambda state: state)
graph.add_edge("add_node", "router2")
graph.add_edge("subtract_node", "router2")

graph.add_conditional_edges(
    "router2",
    router2,
    {
        "addition_operation2": "add_node2",
        "subtraction_operation2": "subtract_node2"
    }
)
graph.add_edge("add_node2", END)
graph.add_edge("subtract_node2", END)

app = graph.compile()

answer = app.invoke({'num1': 10, 'op1': '-', 'num2': 5, 'num3': 7, 'op2': '+', 'num4': 2})

print(answer['result1'])
print(answer['result2'])