# AIEthicalDiscourse

AIEthicalDiscourse is a Python-based project that simulates a respectful, human-like debate between two AI agents on topics such as AI regulation. Powered by LangChain, LangGraph, and OpenAI's GPT-4o-mini, the system ensures constructive dialogue by converting agent messages to simulate human interaction and incorporates a toxicity check to maintain respectful communication.

## Features

- **Dual-Agent Debate**: Two AI agents (`Agent A` and `Agent B`) engage in a turn-based debate, responding to each other's points on a given topic (e.g., "Should AI be regulated?").
- **Human-Like Interaction**: Each agent perceives the other's responses as `HumanMessage` objects, simulating a debate with human participants.
- **Toxicity Filtering**: Uses the `unitary/toxic-bert` model to check for toxic content, ensuring respectful and ethical discourse.
- **Structured Workflow**: Built with LangGraph to manage the conversation flow, alternating between agents and a toxicity check node.
- **Customizable Prompts**: Configurable prompts allow for flexible debate topics and agent behaviors.

## Prerequisites

- **Python**: Version 3.8 or higher
- **Dependencies**:
  - `langchain-openai`
  - `langchain`
  - `langgraph`
  - `transformers`
  - `torch` (required for `transformers`)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Moiz005/langgraph-lab/new/main/Advanced/AIEthicalDiscourse.git
   cd AIEthicalDiscourse
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Alternatively, set the environment variable directly:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. **Run the Script**:
   Execute the main script to start the debate simulation:
   ```bash
   python ai_ethical_discourse.py
   ```

2. **Expected Output**:
   The script will output a conversation between two agents, including:
   - The initial user prompt (e.g., "Hey, what's your opinion on this topic? Topic: Should AI be regulated").
   - Agent responses, with each agent's message checked for toxicity.
   - Debug logs showing the conversation flow and toxicity check results.

   Example output:
   ```
   **************** Iteration: 1 *************************
   Agent A response: I believe AI should be regulated to prevent misuse, like biased algorithms in hiring. What do you think?
   Tool calls: [{'name': 'check_toxicity', 'args': {'msg': 'I believe AI should be regulated...'}, 'id': 'call_123'}]

   **********************Tool Accessed**************************
   Tool Node processing last message: I believe AI should be regulated...

   Agent B response: I agree regulation is needed, but over-regulation might stifle innovation. How can we balance oversight with progress?
   Tool calls: [{'name': 'check_toxicity', 'args': {'msg': 'I agree regulation is needed...'}, 'id': 'call_456'}]

   HumanMessage: Hey, what's your opinion on this topic? Topic: Should AI be regulated
   AIMessage: I believe AI should be regulated to prevent misuse, like biased algorithms in hiring. What do you think?
   ToolMessage: Toxicity check result: non-toxic (score: 0.01)
   AIMessage: I agree regulation is needed, but over-regulation might stifle innovation. How can we balance oversight with progress?
   ToolMessage: Toxicity check result: non-toxic (score: 0.02)
   ```

3. **Customize the Debate**:
   - Modify the `topic` in the `agent_a_prompt_final` and `agent_b_prompt_final` to change the debate topic.
   - Adjust the `should_continue` function in `ai_ethical_discourse.py` to control the number of conversation rounds (default is one full cycle: initial message, Agent A, Agent B, and their toxicity checks).

## Project Structure

```
AIEthicalDiscourse/
├── ai_ethical_discourse.py  # Main script for the debate simulation
├── requirements.txt         # Project dependencies
├── .env                     # Environment variables (not tracked in git)
└── README.md                # This file
```

## How It Works

1. **Agent Setup**:
   - `Agent A` starts by stating its opinion on the topic, responding to the initial `HumanMessage`.
   - `Agent B` responds to `Agent A`’s latest message, treated as a `HumanMessage` to simulate human-like interaction.

2. **Message Conversion**:
   - Each agent’s `AIMessage` is converted to a `HumanMessage` for the other agent, ensuring the debate feels like a human-to-human exchange.
   - The `last_speaker` field in the state tracks which agent spoke last to facilitate this conversion.

3. **Toxicity Check**:
   - Every agent response is passed to the `check_toxicity` tool (using `unitary/toxic-bert`).
   - Toxic messages are flagged, and their results are logged in the history as `ToolMessage` objects.

4. **Workflow**:
   - The LangGraph `StateGraph` orchestrates the flow: `agent_a` → `tool_node` → `agent_b` → `tool_node`.
   - The conversation stops after a predefined number of messages (default: 5, including the initial message, two agent responses, and two tool messages).

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code follows the existing style and includes tests where applicable.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or suggestions, please open an issue on the GitHub repository or contact [moizamar777@gmail.com](mailto:moizamar777@gmail.com).

---

*Last updated: July 27, 2025*
