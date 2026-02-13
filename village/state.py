import village.config as cfg
from village.types import AgentConfig, AgentState, RunState
from typing import List

def build_agents() -> list[AgentState]:
    agents: List[AgentState] = []

    for agent_cfg in cfg.AGENTS:
        missing_keys = cfg.REQUIRED_AGENT_KEYS - agent_cfg.keys()
        
        if missing_keys:
            raise ValueError(f"Agent config is missing keys: {missing_keys}. Got: {list(agent_cfg.keys())}")

        agents.append(AgentState(config=AgentConfig(**agent_cfg)))

    return agents

def create_run_state(goal: str) -> RunState:
    agents = build_agents()

    return RunState(
        goal = goal, 
        agents = agents, 
    )
