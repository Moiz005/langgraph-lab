# A LangGraph agent with tool use that detects when a tool (like addition) is needed, invokes it via ToolNode, and loops until the task is complete using GPT-4o.

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages  # Reduction Function
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import os

class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int):
  """This is an addition function that adds 2 numbers together"""
  return a+b

tools = [add]
os.environ["OPENAI_API_KEY"] = 'YOUR_API_KEY'
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
  system_prompt = SystemMessage(content="You are my AI assistant, please answer my queries to the best of your ability.")
  response = model.invoke([system_prompt] + state["messages"])
  return {"messages": state["messages"] + [response]}

def should_continue(state:AgentState) -> AgentState:
  messages = state["messages"]
  last_message = messages[-1]
  if not last_message.tool_calls:
    return "end"
  else:
    return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
  for s in stream:
    message = s["messages"][-1]
    if isinstance(message, tuple):
      print(message)
    else:
      message.pretty_print()

inputs = {"messages": ["Add 40 + 12."]}

print_stream(app.stream(inputs, stream_mode="values"))