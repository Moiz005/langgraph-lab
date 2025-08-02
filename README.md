# LangGraph Projects

**A curated collection of LangGraph-based projects showcasing intelligent agent workflows — from beginner-friendly examples to advanced multi-tool agent architectures.**

This repository is part of my journey to master [LangGraph](https://www.langchain.com/langgraph), a powerful framework for building multi-node, agentic applications on top of LangChain.

---

## 🚀 What’s Inside

| Level        | Project Name                 | Description                                                                |
| ------------ | ---------------------------- | -------------------------------------------------------------------------- |
| Beginner     | `greeting_node`           | A simple node that adds a personalized greeting message.                   |
| Beginner     | `arithmetic_agent`        | Performs sum or product on input list and returns a result message.        |
| Beginner     | `sequential_user_info`    | Builds a user profile message in 3 connected steps: name, age, skills.     |
| Intermediate | `conditional_arithmetic`  | Routes through operations based on user input using conditional edges.     |
| Intermediate | `guessing_game`           | A looped guessing game where the agent narrows its guesses over attempts.  |
| Intermediate | `chat_loop_gpt4o`         | Stateless chatbot using OpenAI GPT-4o in a simple LangGraph.               |
| Intermediate | `chat_with_memory_logger` | Chatbot with conversation memory that logs messages to a text file.        |
| Intermediate | `tool_use_agent`          | Agent that uses tools (e.g., add) via `ToolNode` and loops intelligently.  |
| Intermediate | `document_drafter_agent`  | Full-fledged tool-using agent for document editing and saving with memory. |
| Advanced     | `AIEthicalDiscourse`      | It simulates a respectful debate between two AI agents on topics like AI regulation, using LangChain and LangGraph with toxicity checks to ensure ethical, human-like dialogue.|
| Advanced     | `research-team-sim`       | A multi-agent system built in LangGraph that simulates a research team to read, analyze, and summarize scientific papers using AI agents with distinct roles and memory types.|
---

## 📦 Requirements

* Python 3.9+
* `langgraph`
* `langchain`
* `langchain_openai`
* `openai` API Key (add to environment as `OPENAI_API_KEY`)

Install dependencies:

```bash
pip install langchain langchain_core langchain_community langchain_openai langgraph transformers
```

---

## 🧠 About LangGraph

[LangGraph](https://docs.langchain.com/langgraph/) is a library for building stateful, multi-step agents and workflows using a graph-based execution model. It builds on top of LangChain and is ideal for applications involving reasoning, memory, and tool use.

---

## 📂 Structure

```
langgraph-projects/
│
├── beginner/
│   ├── greeting_node.py
│   ├── arithmetic_agent.py
│   └── sequential_user_info.py
│
├── intermediate/
│   ├── conditional_arithmetic.py
│   ├── guessing_game.py
│   ├── chat_loop_gpt4o.py
│   ├── chat_with_memory_logger.py
|   └── document_drafter_agent.py
│
├── advanced/
│   └── AIEthicalDiscourse
│
└── README.md
```

---

## 🧑‍💻 Author

**Muhammad Moiz**
Exploring agentic AI, LangChain, and graph-based workflows.
Reach out on [LinkedIn](https://www.linkedin.com/in/muhammad-moiz-49aa4a239/) or follow my GitHub for more AI projects.

---

