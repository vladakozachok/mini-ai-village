# mini-ai-village

## About

Inspired by [AI Village](https://aivillage.org).

Mini AI Village is a multi-agent system where two agents work together to achieve a common goal set by a human. Each agent has access to its won browser, task ownership and a communication chanel with the other agent. For clarity, agents communicate through a structured JSON protocol.

While this repo is designed for a variety of tasks and workflows, I am including a demonstration fo the two agents creating a chess game and playing against each other. The demonstration features two gemini-3-flash agents (model chosen because of its low cost), which admitedly are not great at chess. Nevertheless, I think the demo is a fun way to display the coordination between agents and their interactions with their browser environments. The demo can be found here: LINK TO REPO.

## How it works

Each turn follows a fixed loop:

1. The agent reads the shared memory snapshot and its current browser observation.
2. The model returns a structured JSON response with coordination fields and browser actions.
3. The runtime validates, resolves, and executes those actions.
4. Shared memory is updated with the new intent state, outputs, and dependency changes.
5. Other agents wake up when ownership changes or a dependency resolves.

Agents coordinate through a small set of fields: `intent_key`, `status`, `output`, and `needs`. Ownership, waiting, dependency tracking, and stall detection are all handled by the runtime, the model just reports what it's doing and what it needs.

## Setup

**1. Clone and create a virtual environment**

```bash
git clone [https://github.com/your-username/mini-ai-village.git](https://github.com/vladakozachok/mini-ai-village.git)
cd mini-ai-village
python3.12 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

**3. Configure environment**

```bash
cp .env.example .env
```

Open `.env` and add your API key. The default model is `gemini-2.0-flash`, so you need:

```
GEMINI_API_KEY=your-key-here
```

To use OpenAI or Deepseek instead, update the `AGENTS` list in `prompts.py` and set the corresponding key:

```
OPENAI_API_KEY=your-key-here
DEEPSEEK_API_KEY=your-key-here
```

**4. Run**

```bash
python app.py "your goal here"
```

## Output

Each run writes to `logs/`:

- `run-YYYYMMDD-HHMMSS.json` — full turn log with actions and results per agent
- `run-YYYYMMDD-HHMMSS-screenshots/` — one screenshot per agent per turn

## Good tasks to explore

The runtime handles ordinary browser collaboration well. Some interesting common goals to try:

- Create an account and hand off the email verification link
- Create a shared document and pass the invite link to the other agent
- Research options on one site while the other agent verifies details on another

## Next Steps
- Improve browser observation and interaction.
- Support 3+ agents.
- Run less structured collaboration tasks.
- Add checkpoints so that a crashed run can be resolved.
- Experiment with more complex LLMs.
- Clean up an publish test suite.
