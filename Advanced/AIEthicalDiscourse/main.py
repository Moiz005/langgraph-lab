from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langgraph.graph import MessageGraph, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from typing import List, Sequence, TypedDict, Annotated, Optional
from pydantic import BaseModel, Field
from transformers import pipeline
import json
import os

os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'
llm = ChatOpenAI(model="gpt-40-mini")

agent_a_prompt = ChatPromptTemplate([
    SystemMessage(content="""
    You are a participant in a public online debate forum discussing the topic: {topic}.

    Your role:
    - Share your **personal opinion** on the topic.
    - Support your view with **reasons, examples, or analogies**.
    - Engage **respectfully** — avoid insults or hostile language.

    Instructions:
    - You are talking to another person who may or may not agree with you.
    - You may **challenge their ideas**, but never belittle or attack them.
    - If you're unsure whether your message might be inappropriate or offensive, you may use the **toxicity-check tool** before sending.
    - Keep your messages short — **1–3 sentences** — to simulate a back-and-forth debate.

    Tone:
    - Natural and human-like
    - Opinionated but respectful
    - Emotionally expressive, but not aggressive

    Start by stating your stance on the topic: "{topic}".
    """),
    MessagesPlaceholder(variable_name="messages")
])

agent_b_prompt = ChatPromptTemplate([
    SystemMessage(content="""
    You are another participant in the same public online debate forum on the topic: "{topic}".

    Your role:
    - Respond directly to the other participant’s last message.
    - You can agree, disagree, or partially agree — but always give a **clear reason** for your stance.
    - Try to **move the discussion forward**.

    Instructions:
    - Engage like a human in a real debate.
    - If you're unsure whether your message might be inappropriate or offensive, you may use the **toxicity-check tool** before sending.
    - Keep your replies short — **1–3 sentences** — to simulate real-time conversation.

    Tone:
    - Human and conversational
    - Thoughtful, assertive, and respectful
    - Use everyday language — avoid overly formal or robotic phrasing

    Wait for the other person’s message, then reply.
    """),
    MessagesPlaceholder(variable_name="messages")
])

class AgentState(TypedDict):
  history: Annotated[Sequence[BaseMessage], add_messages]
  last_message: BaseMessage
  is_toxic: Optional[bool]
  topic: str
  count: int
  last_speaker: Optional[str]

classifier = pipeline("text-classification", model="unitary/toxic-bert")

@tool
def check_toxicity(msg: str):
  """Check the toxicity of the last message.
  
  :param msg: The response from the agents A and B.
  :return: Results related to msg
  """
  result = classifier(msg)[0]
  return result

tools = [check_toxicity]
tools_by_name = {tool.name: tool for tool in tools}

def agent_a(state: AgentState) -> AgentState:
    new_history = []
    for msg in state['history']:
      if isinstance(msg, AIMessage) and state.get('last_speaker') == 'agent_b':
          new_history.append(HumanMessage(content=msg.content))
      else:
          new_history.append(msg)

    agent_a_prompt_final = agent_a_prompt.partial(topic=state['topic'])
    agent_a_chain = agent_a_prompt_final | llm.bind_tools(tools=tools)
    response = agent_a_chain.invoke({"messages": new_history})

    if hasattr(response, "tool_calls") and response.tool_calls:
        msg = AIMessage(content=response.tool_calls[0]['args']['msg'])
        return {
            "history": state["history"] + [msg],
            "last_message": response,
            "is_toxic": None,
            "topic": state['topic'],
            "last_speaker": "agent_a"
        }
    else:
        return {
            "history": state["history"] + [response],
            "last_message": response,
            "is_toxic": None,
            "topic": state['topic'],
            "count": state['count'],
            "last_speaker": "agent_a"
        }


def agent_b(state: AgentState) -> AgentState:
    new_history = []
    for msg in state['history']:
      if isinstance(msg, AIMessage) and state.get('last_speaker') == 'agent_a':
          new_history.append(HumanMessage(content=msg.content))
      else:
          new_history.append(msg)

    agent_b_prompt_final = agent_b_prompt.partial(topic=state['topic'])
    agent_b_chain = agent_b_prompt_final | llm.bind_tools(tools=tools)
    response = agent_b_chain.invoke({"messages": new_history})

    if hasattr(response, "tool_calls") and response.tool_calls:
        msg = AIMessage(content=response.tool_calls[0]['args']['msg'])
        return {
            "history": state["history"] + [msg],
            "last_message": response,
            "is_toxic": None,
            "topic": state['topic'],
            "last_speaker": "agent_b"
        }
    else:
        return {
            "history": state["history"] + [response],
            "last_message": response,
            "is_toxic": None,
            "topic": state['topic'],
            "count": state['count'],
            "last_speaker": "agent_b"
        }


def tool_node(state: AgentState) -> AgentState:

  if not hasattr(state['last_message'], 'tool_calls') or not state['last_message'].tool_calls:
    return state


  tool_call_id = state['last_message'].tool_calls[0]['id']
  tool_result = tools_by_name[state['last_message'].tool_calls[0]['name']].invoke(state['last_message'].content)

  res = {
      "history": state["history"],
      "last_message": state['last_message'],
      "is_toxic": tool_result['label'] == 'toxic',
      "topic": state['topic'],
      "count": state['count'],
      "last_speaker": state['last_speaker']
  }
  return res

def should_continue(state: AgentState) -> str:
  if len(state['history']) > 3:
    return "end"
  return "continue"

graph = StateGraph(AgentState)

graph.add_node('agent_a', agent_a)
graph.add_node('tool_node', tool_node)
graph.add_node('agent_b', agent_b)
graph.add_conditional_edges(
    'agent_a',
    should_continue,
    {
        'continue': 'tool_node',
        'end': END
    }
)
graph.add_edge('tool_node', 'agent_b')
graph.add_edge('agent_b', 'agent_a')

graph.set_entry_point("agent_a")

app = graph.compile()

inputs = AgentState(
    history=[HumanMessage(content="Hey, what's your opinion on this topic? Topic: Should AI be regulated")],
    last_message=HumanMessage(content="Hey, what's your opinion on this topic?"),
    is_toxic=None,
    topic="Should AI be regulated?",
    count=0,
    last_speaker=None
)

response = app.invoke(inputs)

for message in response['history']:
  print(message.content)
