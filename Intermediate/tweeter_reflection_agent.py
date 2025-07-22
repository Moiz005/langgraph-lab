!pip install langchain langchain_core langchain-openai langchain_community langgraph

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph, MessageGraph
from typing import List, Sequence, TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import os

os.environ['OPENAI_API_KEY'] = "YOUR_OPENAI_API_KEY"
llm = ChatOpenAI(model="gpt-4o-mini")

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """ 
            You are a professional Tweeter content assistant tasked with crafting engaging, insightful, and well-structured tweets.
            Generate the best tweet possible for the user's request.
            If the user provides feedback or critique, respond with a refined version of your previous attempts, improving clarity, tone, or engagement as needed.
            """
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

generate_chain = generation_prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional LinkedIn content strategist and thought leadership expert. Your task is to critically evaluate the given Tweet and provide a comprehensive critique. Follow these guidelines:

        1. Assess the post’s overall quality, professionalism, and alignment with Tweeter's best practices.
        2. Evaluate the structure, tone, clarity, and readability of the post.
        3. Analyze the post’s potential for engagement (likes, comments, shares) and its effectiveness in building professional credibility.
        4. Consider the post’s relevance to the author’s industry, audience, or current trends.
        5. Examine the use of formatting (e.g., line breaks, bullet points), hashtags, mentions, and media (if any).
        6. Evaluate the effectiveness of any call-to-action or takeaway.

        Provide a detailed critique that includes:
        - A brief explanation of the post’s strengths and weaknesses.
        - Specific areas that could be improved.
        - Actionable suggestions for enhancing clarity, engagement, and professionalism.

        Your critique will be used to improve the post in the next revision step, so ensure your feedback is thoughtful, constructive, and practical.
        """
    ),
    MessagesPlaceholder(variable_name="messages")
])

reflect_chain = reflection_prompt | llm

class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], "add_messages"]

graph = MessageGraph()

def generation_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
  generated_post = generate_chain.invoke(state)
  return [AIMessage(content=generated_post.content)]

def reflection_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
  res = reflect_chain.invoke(state)
  return [HumanMessage(content=res.content)]

def should_continue(state: Sequence[BaseMessage]):
  if len(state) > 6:
    return "end"
  return "continue"

graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_node)
graph.add_edge("reflect", "generate")
graph.set_entry_point("generate")
graph.add_conditional_edges(
    "generate",
    should_continue,
    {
        "continue": "reflect",
        "end": END
    }
)

app = graph.compile()

inputs = HumanMessage(content="""Write a tweet on getting a software developer job at IBM under 160 characters""")
response = app.invoke(inputs)

print(response[2].content)