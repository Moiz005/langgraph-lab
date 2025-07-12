# A LangGraph-based number guessing game where the agent makes random guesses with bounded updates and loops until the correct number is found or attempts run out.

from typing import TypedDict, Dict, List
import random
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
  player_name: str
  guesses: List[int]
  attempts: int
  lower_bound: int
  upper_bound: int
  num: int

def setup(state: AgentState) -> AgentState:
  state['attempts'] = 0
  state['guesses'] = []
  return state

def guess(state: AgentState) -> AgentState:
  state['guesses'].append(random.randint(state['lower_bound'], state['upper_bound']))
  state['attempts'] += 1
  return state


def hint(state: AgentState) -> AgentState:
  if state['num'] > state['guesses'][-1]:
    state['lower_bound'] = state['guesses'][-1] + 1
  else:
    state['upper_bound'] = state['guesses'][-1] + 1
  return state

def should_continue(state: AgentState) -> str:
  if state['attempts'] < 7 and state['num'] not in state['guesses']:
    return "loop"
  else:
    return "exit"

graph = StateGraph(AgentState)

graph.add_node('setup', setup)
graph.add_node('guess', guess)
graph.add_node('hint', hint)

graph.add_edge(START, 'setup')
graph.add_edge('setup', 'guess')
graph.add_conditional_edges(
    "guess",
    should_continue,
    {
        "loop": "guess",
        "exit": END
    }
)

app = graph.compile()

anwser = app.invoke({'player_name': 'Student', 'guesses': [], 'attempts': 0, 'lower_bound': 1, 'upper_bound': 20, 'num': 17})

print("Guesses: ")
print(anwser['guesses'])