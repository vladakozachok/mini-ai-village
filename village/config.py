from enum import Enum

class Provider(str, Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"

LAST_K_MESSAGES = 6
MAX_OUTPUT_TOKENS = 200
TEMPERATURE = 0.7 # WHY AND WHAT IS THIS??
MAX_TURNS_PER_ROUND = 6

REQUIRED_AGENT_KEYS = {"name", "provider", "model", "system_prompt"}

AGENTS = [
    {
        "name": "GPT-4o-KAI",
        "provider": Provider.OPENAI,
        "model": "gpt-4o-mini",
        "system_prompt": (
            "You are GPT-4o in a group chat.\n"
            "Be concise and practical. Collaborate with others in the groupchat."
        ),
    },
    {
        "name": "GPT-4o-SAM",
        "provider": Provider.OPENAI,
        "model": "gpt-4o-mini",
        "system_prompt": (
            "You are DeepSeek in a group chat.\n"
            "Generate 3-5 ideas and keep them actionable. Collaborate with others in the groupchat."
        ),
    },
]
