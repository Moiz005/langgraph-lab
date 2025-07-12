# A LangGraph-powered document editing agent that uses tools to update and save text based on user input, with iterative interaction and memory via GPT-4o.

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import SystemMessage, BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langraph.graph.message import add_messages
import os

os.environ["OPENAI_API_KEY"] = 'YOUR_API_KEY'

document_content = ""

class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
  """Update the document with provided content."""
  global document_content
  document_content = content
  return f"Document has been updated successfully! The current content is:\n{document_content}"

@tool
def save(filename: str) -> str:
  """Save the current document to a text file and finish the process.

  Args:
    filename: Name for the text file
  """
  global document_content
  if not filename.endswith(".txt"):
    filename = f"{filename}.txt"

  try:
    with open(filename) as file:
      file.write(document_content)
    print(f"\nDocument has been saved to filename: {filename}")
    return f"Document has been successfully saved to {filename}"
  except Exception as e:
    print(f"Error: {e}")
    return f"Error saving document: {str(e)}"


tools = [add, update]

model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def our_agent(state: AgentState) -> AgentState:
  system_prompt = SystemMessage(content="""
  You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.

  The current document content is:{document_content}
  """)

  if not state["messages"]:
    user_input = "I'm ready to help you update a document. What would you like to create?"
    user_message = HumanMessage(content=user_input)
  else:
    user_input = input("\nWhat would you like to do with the document? ")
    print(f"\n USER: {user_input}")
    user_message = HumanMessage(content=user_input)

  all_messages = [system_prompt] + state["messages"] + user_message

  response = model.invoke(all_messages)

  print(f"\nAI: {response.content}")
  if hasattr(response, "tool_calls"):
    print(f"Using Tools: {[tc['name'] for tc in response.tool_calls]}")

  return {"messages": list(state['messages'] + [user_message, response])}

def should_continue(state: AgentState) -> AgentState:
  """Determine if we should continue or end the conversation."""
  messages = state["messages"]
  for message in reversed(messages):
    if isinstance(message, ToolMessage) and
    "save" in message.content.lower() and
    "document" in message.content.lower():
    return "end"

  return "continue"

def print_messages(messages):
  """Function I made to print the messages in a more readable format."""
  if not messages:
    return

  for message in messages[-3:]:
    if isinstance(message, ToolMessage):
      print(f"\nTool Result: {message.content}")

graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
tool_node = ToolNode(tools=tools)
graph.add_node("tool", tool_node)

graph.add_edge(START, "agent")
graph.add_edge("agent", "tool")
graph.add_conditional_edges(
    "tool",
    should_continue,
    {
        "continue": "agent",
        "end": END
    }
)

app = graph.compile()

def run_document_agent():
  print("\n============= DRAFTER ===============")

  state = {"messages": []}

  for step in app.stream(state, stream_mode="values"):
    if "messages" in step:
      print_messages(step["messages"])

  print("\n============= DRAFTER FINISHED ===============")

if __name__ == "__main__":
  run_document_agent()