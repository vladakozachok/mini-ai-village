import asyncio
import logging
import os

import village.config as cfg
from village.types import AgentConfig, AgentState, RunState

logger = logging.getLogger(__name__)

_REQUIRED_ENV_VARS: dict[cfg.Provider, str] = {
    cfg.Provider.GEMINI: "GEMINI_API_KEY",
    cfg.Provider.OPENAI: "OPENAI_API_KEY",
    cfg.Provider.DEEPSEEK: "DEEPSEEK_API_KEY",
}


def _validate_api_keys() -> None:
    missing: list[str] = []
    for agent_cfg in cfg.AGENTS:
        provider = agent_cfg.get("provider")
        env_var = _REQUIRED_ENV_VARS.get(provider)
        if env_var and not os.environ.get(env_var):
            missing.append(f"{env_var} (required by agent '{agent_cfg.get('name')}')")

    if missing:
        raise EnvironmentError(
            "Missing required API key(s) — set these environment variables before running:\n"
            + "\n".join(f"  {m}" for m in missing)
        )


async def build_agents() -> list[AgentState]:
    agents: list[AgentState] = []

    for agent_cfg in cfg.AGENTS:
        missing_keys = cfg.REQUIRED_AGENT_KEYS - agent_cfg.keys()
        if missing_keys:
            raise ValueError(
                f"Agent config is missing keys: {missing_keys}. "
                f"Got: {list(agent_cfg.keys())}"
            )

        try:
            agent = AgentState(config=AgentConfig(**agent_cfg))
        except Exception as e:
            raise ValueError(f"Failed to create agent from config: {e}") from e

        try:
            await agent.launch_browser()
        except Exception as exc:
            logger.error(
                "Browser launch failed for agent '%s' (model=%s provider=%s); "
                "closing %d already-launched browser(s).",
                agent_cfg["name"],
                agent_cfg.get("model", "unknown"),
                agent_cfg.get("provider", "unknown"),
                len(agents),
                exc_info=True,
            )
            for launched in agents:
                try:
                    await launched.close_browser()
                except Exception:
                    logger.warning(
                        "Failed to close browser for agent '%s' during cleanup.",
                        launched.name,
                        exc_info=True,
                    )
            raise RuntimeError(
                f"Browser launch failed for agent '{agent_cfg['name']}': {exc}"
            ) from exc

        logger.info(
            "Agent '%s' ready (model=%s provider=%s).",
            agent.name,
            agent.model,
            agent.provider.value,
        )
        agents.append(agent)

    return agents


async def create_run_state(goal: str) -> RunState:
    _validate_api_keys()
    logger.info("Creating run state for goal: %r", goal)
    agents = await build_agents()
    return await RunState.create(goal=goal, agents=agents)
