import asyncio

import village.config as cfg
from village.types import AgentConfig, AgentState, RunState
from typing import List

async def build_agents() -> list[AgentState]:
    agents: List[AgentState] = []

    for agent_cfg in cfg.AGENTS:
        missing_keys = cfg.REQUIRED_AGENT_KEYS - agent_cfg.keys()
        
        if missing_keys:
            raise ValueError(f"Agent config is missing keys: {missing_keys}. Got: {list(agent_cfg.keys())}")

        agent = AgentState(config=AgentConfig(**agent_cfg))
        await agent.launch_browser()
        agents.append(agent)

    return agents

async def create_run_state(goal: str) -> RunState:
    agents = await build_agents()

    return RunState(
        goal = goal, 
        agents = agents, 
    )
