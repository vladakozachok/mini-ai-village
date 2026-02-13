import village.config as cfg

from village.types import AgentState, Message, RunState
from village.providers.openai_provider import generate_response as openai_generate
from village.providers.deepseek_provider import generate_response as deepseek_generate

PROVIDER_GEN = {
    cfg.Provider.OPENAI: openai_generate,
    cfg.Provider.DEEPSEEK: deepseek_generate,
}

def build_input_text(state: RunState, agent: AgentState) -> str:
    relevant_meesages = state.messages[-cfg.LAST_K_MESSAGES:]

    lines = []
    lines.append(f"GOAL: {state.goal}\n")
    lines.append(f"GROUP CHAT ACTIVITY: \n")
    
    for message in relevant_meesages:
        lines.append(f"{message.speaker}: {message.content}\n")

    return "".join(lines)



def run_turn(state: RunState) -> None:
    agent = state.get_next_agent()
    input_text = build_input_text(state, agent)
    generate_response_fn = PROVIDER_GEN[agent.provider]

    response = generate_response_fn(
        model = agent.model,
        instructions = agent.system_prompt,
        input_text=input_text
    )

    state.messages.append(
        Message(
            speaker = agent.name,
            content = response
        )
    )

    state.update_next_agent()


def run_group(state:RunState) -> None:

    while len(state.messages) < cfg.MAX_TURNS_PER_ROUND:
        run_turn(state)
