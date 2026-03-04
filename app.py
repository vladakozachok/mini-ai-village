import asyncio
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import village.config as cfg
from village.engine import run_group
from village.coordination import parse_coordination_fields
from village.state import create_run_state
from village.types import AgentState, Message, RunState

LOG_DIR = Path("logs")
logger = logging.getLogger("village.run")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("village.run").setLevel(logging.INFO)
    logging.getLogger("village.state").setLevel(logging.INFO)
    logging.getLogger("village.engine").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("google.genai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI village group chat.")
    parser.add_argument("goal", help="Goal for the agents to collaborate on.")
    return parser.parse_args()


def _agent_log_identity(state: RunState, speaker: str) -> dict:
    for agent in state.agents:
        if agent.name == speaker:
            return {
                "name": agent.name,
                "provider": str(agent.provider),
                "model": agent.model,
            }
    return {
        "name": speaker,
        "provider": "unknown",
        "model": "unknown",
    }


def serialize_message(msg: Message, state: RunState) -> dict:
    payload = {
        "turn": msg.turn,
        "agent": _agent_log_identity(state, msg.speaker),
        "message": msg.message,
        "actions": [action.to_dict() for action in msg.actions],
        "action_results": [result.to_dict() for result in msg.action_results],
        "timestamp": msg.timestamp,
    }
    return payload


def init_log_data(state: RunState, run_id: str, start_time: datetime) -> dict:
    return {
        "run_id": run_id,
        "goal": state.goal,
        "started_at": start_time.isoformat(),
        "finished_at": None,
        "duration_seconds": None,
        "agents": [
            {
                "name": agent.name,
                "provider": str(agent.provider),
                "model": agent.model,
            }
            for agent in state.agents
        ],
        "messages": [],
    }


def write_json_log(json_path: Path, log_data: dict) -> None:
    json_path.write_text(json.dumps(log_data, indent=2))


def init_logs(state: RunState, run_id: str, start_time: datetime) -> tuple[Path, dict]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    json_path = LOG_DIR / f"{run_id}.json"
    log_data = init_log_data(state, run_id, start_time)
    write_json_log(json_path, log_data)

    return json_path, log_data


def append_message_to_json_log(
    json_path: Path, log_data: dict, msg: Message, state: RunState
) -> None:
    log_data["messages"].append(serialize_message(msg, state))
    write_json_log(json_path, log_data)


def log_turn_summary(state: RunState, msg: Message) -> None:
    agent_meta = _agent_log_identity(state, msg.speaker)
    fields = parse_coordination_fields(msg.message)

    logger.info(
        "turn=%s agent=%s provider=%s model=%s status=%s intent=%s",
        msg.turn,
        agent_meta["name"],
        agent_meta["provider"],
        agent_meta["model"],
        fields.get("status", "?"),
        fields.get("intent_key", "?"),
    )
    logger.info("message=%s", msg.message)

    if msg.actions:
        logger.info(
            "actions=%s", json.dumps([action.to_dict() for action in msg.actions])
        )
    else:
        logger.info("actions=[]")

    if msg.action_results:
        logger.info(
            "results=%s",
            json.dumps([result.to_dict() for result in msg.action_results]),
        )
    else:
        logger.info("results=[]")


def finalize_json_log(
    json_path: Path,
    log_data: dict,
    start_time: datetime,
    end_time: datetime,
    state: RunState,
) -> None:
    if len(log_data.get("messages", [])) < len(state.messages):
        log_data["messages"] = [serialize_message(msg, state) for msg in state.messages]
    log_data["finished_at"] = end_time.isoformat()
    log_data["duration_seconds"] = (end_time - start_time).total_seconds()
    write_json_log(json_path, log_data)


async def capture_screenshot(run_id: str, turn: int, agent: AgentState) -> Path | None:
    screenshots_dir = LOG_DIR / f"{run_id}-screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    safe_name = agent.name.replace(" ", "_")
    path = screenshots_dir / f"turn-{turn:02d}-{safe_name}.png"

    try:
        screenshot = await agent.get_browser().screenshot()
        path.write_bytes(screenshot)
        return path
    except Exception as exc:
        logger.warning(
            "screenshot_save_failed agent=%s turn=%s error=%s",
            agent.name,
            turn,
            exc,
        )
        return None


async def cleanup_agents(state: RunState | None) -> None:
    if not state:
        return

    await asyncio.gather(
        *(agent.close_browser() for agent in state.agents), return_exceptions=True
    )


async def main() -> None:
    configure_logging()
    args = parse_args()
    run_id = f"run-{datetime.now():%Y%m%d-%H%M%S}"
    state = None
    screenshots: list[Path] = []
    log_lock = asyncio.Lock()

    try:
        start_time = datetime.now()
        state = await create_run_state(args.goal)
        json_path, log_data = init_logs(state, run_id, start_time)

        async def after_turn(
            current_state: RunState, acting_agent: AgentState, latest_message: Message
        ) -> None:
            async with log_lock:
                append_message_to_json_log(
                    json_path, log_data, latest_message, current_state
                )
                log_turn_summary(current_state, latest_message)

                if latest_message.turn is None:
                    raise RuntimeError(
                        "Invariant violated: message turn was not assigned before after_turn callback."
                    )

                screenshot = await capture_screenshot(
                    run_id, latest_message.turn, acting_agent
                )
                if screenshot is not None:
                    screenshots.append(screenshot)

        await run_group(state, after_turn=after_turn)
        end_time = datetime.now()
        finalize_json_log(json_path, log_data, start_time, end_time, state)

        if screenshots:
            print(
                f"Saved run data to {json_path} and {len(screenshots)} screenshots to {screenshots[0].parent}"
            )
        else:
            print(f"Saved run data to {json_path}")

    finally:
        await cleanup_agents(state)


if __name__ == "__main__":
    asyncio.run(main())
