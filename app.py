import asyncio
import argparse
import json
import contextlib
from datetime import datetime
from pathlib import Path

import village.config as cfg
from village.engine import run_group
from village.state import create_run_state
from village.types import AgentState, Message, RunState


LOG_DIR = Path("logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI village group chat.")
    parser.add_argument("goal", help="Goal for the agents to collaborate on.")
    return parser.parse_args()


def _prepare_observation_for_log(observation: dict | None) -> dict | None:
    if observation is None:
        return None

    normalized: dict
    if isinstance(observation, dict):
        normalized = dict(observation)
    else:
        normalized = {"observation_repr": repr(observation)}

    elements = normalized.get("elements")
    if isinstance(elements, list) and cfg.MAX_OBSERVATION_LOG_ELEMENTS >= 0:
        normalized["elements"] = elements[: cfg.MAX_OBSERVATION_LOG_ELEMENTS]

    target_surfaces = normalized.get("target_surfaces")
    if isinstance(target_surfaces, list) and cfg.MAX_OBSERVATION_TARGET_SURFACES >= 0:
        clipped_surfaces = target_surfaces[: cfg.MAX_OBSERVATION_TARGET_SURFACES]
        for surface in clipped_surfaces:
            if not isinstance(surface, dict):
                continue
            anchors = surface.get("child_anchors")
            if isinstance(anchors, list) and cfg.MAX_OBSERVATION_SURFACE_ANCHORS >= 0:
                surface["child_anchors"] = anchors[: cfg.MAX_OBSERVATION_SURFACE_ANCHORS]
        normalized["target_surfaces"] = clipped_surfaces

    visible_text = normalized.get("visible_text")
    if isinstance(visible_text, str) and cfg.MAX_OBSERVATION_VISIBLE_TEXT_CHARS > 0:
        normalized["visible_text"] = visible_text[: cfg.MAX_OBSERVATION_VISIBLE_TEXT_CHARS]

    if cfg.MAX_OBSERVATION_LOG_CHARS > 0:
        serialized = json.dumps(normalized)
        if len(serialized) > cfg.MAX_OBSERVATION_LOG_CHARS:
            normalized = {"observation_excerpt": serialized[: cfg.MAX_OBSERVATION_LOG_CHARS]}

    return normalized


def serialize_message(msg: Message) -> dict:
    payload = {
        "turn": msg.turn,
        "speaker": msg.speaker,
        "raw_response": msg.raw_response,
        "message": msg.message,
        "actions": [action.to_dict() for action in msg.actions],
        "action_results": [result.to_dict() for result in msg.action_results],
        "timestamp": msg.timestamp,
    }
    if cfg.LOG_PROMPTS and msg.input_text is not None:
        if cfg.MAX_PROMPT_LOG_CHARS > 0:
            payload["input_text"] = msg.input_text[: cfg.MAX_PROMPT_LOG_CHARS]
        else:
            payload["input_text"] = msg.input_text
    if cfg.LOG_OBSERVATIONS and msg.observation is not None:
        payload["observation"] = _prepare_observation_for_log(msg.observation)
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
                **({"system_prompt": agent.system_prompt} if cfg.LOG_PROMPTS else {}),
            }
            for agent in state.agents
        ],
        "messages": [],
        "debug_events": [],
    }


def write_json_log(json_path: Path, log_data: dict) -> None:
    json_path.write_text(json.dumps(log_data, indent=2))


def init_logs(state: RunState, run_id: str, start_time: datetime) -> tuple[Path, dict]:
    """Initialize the JSON log so it exists during the run."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    json_path = LOG_DIR / f"{run_id}.json"
    log_data = init_log_data(state, run_id, start_time)
    write_json_log(json_path, log_data)

    return json_path, log_data


def append_message_to_json_log(json_path: Path, log_data: dict, msg: Message, state: RunState) -> None:
    log_data["messages"].append(serialize_message(msg))
    log_data["memory_snapshot"] = state.memory.get_snapshot()
    write_json_log(json_path, log_data)


def append_debug_event(json_path: Path, log_data: dict, event: dict) -> None:
    log_data.setdefault("debug_events", []).append(event)
    write_json_log(json_path, log_data)


def finalize_json_log(json_path: Path, log_data: dict, start_time: datetime, end_time: datetime, state: RunState) -> None:
    # Ensure any missed messages are serialized (e.g. if after_turn failed).
    if len(log_data.get("messages", [])) < len(state.messages):
        log_data["messages"] = [serialize_message(msg) for msg in state.messages]
    log_data["finished_at"] = end_time.isoformat()
    log_data["duration_seconds"] = (end_time - start_time).total_seconds()
    log_data["memory_final"] = state.memory.get_snapshot(max_events=50, max_failures=20)
    write_json_log(json_path, log_data)


async def capture_screenshot(run_id: str, turn: int, agent: AgentState) -> Path | None:
    """Save a screenshot from the agent's browser."""
    screenshots_dir = LOG_DIR / f"{run_id}-screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    
    safe_name = agent.name.replace(" ", "_")
    path = screenshots_dir / f"turn-{turn:02d}-{safe_name}.png"

    try:
        screenshot = await agent.get_browser().screenshot()
        path.write_bytes(screenshot)
        return path
    except Exception as e:
        print(f"Warning: couldn't save screenshot for {agent.name} at turn {turn}: {e}")
        return None


async def cleanup_agents(state: RunState | None) -> None:
    """Close all agent browsers."""
    if not state:
        return
    
    await asyncio.gather(
        *(agent.close_browser() for agent in state.agents),
        return_exceptions=True
    )


async def main() -> None:
    args = parse_args()
    run_id = f"run-{datetime.now():%Y%m%d-%H%M%S}"
    state = None
    screenshots: list[Path] = []
    log_lock = asyncio.Lock()

    try:
        start_time = datetime.now()
        state = await create_run_state(args.goal)
        json_path, log_data = init_logs(state, run_id, start_time)

        async def heartbeat() -> None:
            while state and not state.stop_event.is_set():
                await asyncio.sleep(cfg.DEBUG_HEARTBEAT_SECONDS)
                event = {
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                    "next_turn": state.next_turn,
                    "stop_event": state.stop_event.is_set(),
                    "last_status_by_agent": dict(state.memory.last_status_by_agent),
                    "active_intents": list(state.memory.active_intents.keys()),
                    "dependencies": len(state.memory.dependencies),
                    "blockers": len(state.memory.blockers),
                    "artifacts": len(state.memory.artifacts),
                }
                async with log_lock:
                    append_debug_event(json_path, log_data, event)

        async def after_turn(current_state: RunState, acting_agent: AgentState, latest_message: Message) -> None:
            async with log_lock:
                append_message_to_json_log(json_path, log_data, latest_message, current_state)

                if latest_message.turn is None:
                    raise RuntimeError("Invariant violated: message turn was not assigned before after_turn callback.")
                
                screenshot = await capture_screenshot(run_id, latest_message.turn, acting_agent)
                if screenshot:
                    screenshots.append(screenshot)
            
        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            await run_group(state, after_turn=after_turn)
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
        end_time = datetime.now()
        finalize_json_log(json_path, log_data, start_time, end_time, state)
        
        if screenshots:
            print(f"Saved run data to {json_path} and {len(screenshots)} screenshots to {screenshots[0].parent}")
        else:
            print(f"Saved run data to {json_path}")

    finally:
        await cleanup_agents(state)


if __name__ == "__main__":
    asyncio.run(main())
