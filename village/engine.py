import asyncio
import json
import re
from typing import Any, Awaitable, Callable


import village.config as cfg

from village.types import (
    AgentState,
    Artifact,
    Blocker,
    BlockerType,
    Dependency,
    IntentLease,
    MemoryEvent,
    Message,
    RunState,
)
from village.agent_web_use_orchestration.browser_env import ExecutionResult
from village.providers.openai_provider import generate_response as openai_generate
from village.providers.deepseek_provider import generate_response as deepseek_generate
from village.agent_web_use_orchestration.actions import Action, ActionType, parse_model_action, validate_action_dict

PROVIDER_GEN = {
    cfg.Provider.OPENAI: openai_generate,
    cfg.Provider.DEEPSEEK: deepseek_generate,
}

COORD_KEYS = ("INTENT_KEY", "TASK_ID", "INTENT", "OWNER", "STATUS", "OUTPUT", "NEEDS", "NEXT")


def parse_coordination_fields(message: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in message.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().upper()
        if key in COORD_KEYS:
            fields[key] = value.strip()
    return fields


def _set_coordination_field(message: str, key: str, value: str) -> str:
    pattern = rf"(?im)^\s*{re.escape(key)}\s*:\s*.*$"
    replacement = f"{key}: {value}"
    if re.search(pattern, message):
        return re.sub(pattern, replacement, message, count=1)
    return f"{replacement}\n{message}".strip()


def last_messages_for_each_agent(state: RunState) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for msg in reversed(list(state.messages)):
        if msg.speaker in latest:
            continue
        parsed = parse_coordination_fields(msg.message)
        if parsed:
            latest[msg.speaker] = parsed
    return latest


def find_active_intent_owner(state: RunState, intent: str, exclude_agent: str) -> str | None:
    if not intent:
        return None

    latest = last_messages_for_each_agent(state)
    normalized_intent = intent.strip().lower()

    for speaker, fields in latest.items():
        if speaker == exclude_agent:
            continue
        status = fields.get("STATUS", "").strip().lower()
        claimed_intent = fields.get("INTENT", "").strip().lower()
        if status == "in_progress" and claimed_intent == normalized_intent:
            return fields.get("OWNER") or speaker

    return None


def _normalize_intent_key(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    return "_".join(tokens[:2])[:120]


def _get_intent_key_from_fields(fields: dict[str, str]) -> str:
    raw_key = fields.get("INTENT_KEY", "")
    if raw_key:
        return _normalize_intent_key(raw_key)
    return _normalize_intent_key(fields.get("INTENT", ""))




def find_active_intent_owner_by_key(state: RunState, intent_key: str, exclude_agent: str) -> str | None:
    if not intent_key:
        return None
    latest = last_messages_for_each_agent(state)
    for speaker, fields in latest.items():
        if speaker == exclude_agent:
            continue
        status = fields.get("STATUS", "").strip().lower()
        claimed_key = _get_intent_key_from_fields(fields)
        if status == "in_progress" and claimed_key == intent_key:
            return fields.get("OWNER") or speaker
    return None


def build_coordination_summary(state: RunState) -> str:
    latest = last_messages_for_each_agent(state)
    if not latest:
        return "COORDINATION SUMMARY: none yet."

    lines = ["COORDINATION SUMMARY:"]
    for speaker, fields in latest.items():
        intent_key = _get_intent_key_from_fields(fields) or "none"
        task_id = fields.get("TASK_ID", "none")
        intent = fields.get("INTENT", "none")
        status = fields.get("STATUS", "unknown")
        output = fields.get("OUTPUT", "none")
        needs = fields.get("NEEDS", "none")
        lines.append(
            f"- {speaker}: intent_key={intent_key}; task_id={task_id}; intent={intent}; status={status}; output={output}; needs={needs}"
        )

    return "\n".join(lines)


def _truncate_value(value: str, limit: int = 180) -> str:
    value = (value or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _compact_memory_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    recent_events = []
    for event in (snapshot.get("recent_events") or [])[-4:]:
        if not isinstance(event, dict):
            continue
        recent_events.append(
            {
                "agent": event.get("agent", ""),
                "event_type": event.get("event_type", ""),
                "intent": event.get("intent", ""),
                "turn": event.get("turn", 0),
            }
        )

    active_intents = {}
    for key, value in list((snapshot.get("active_intents") or {}).items())[:6]:
        if not isinstance(value, dict):
            continue
        active_intents[key] = {
            "owner": value.get("owner", ""),
            "status": value.get("status", ""),
            "expires_at_turn": value.get("expires_at_turn", 0),
        }

    dependencies = []
    for dep in (snapshot.get("dependencies") or [])[:6]:
        if not isinstance(dep, dict):
            continue
        dependencies.append(
            {
                "from_agent": dep.get("from_agent", ""),
                "to_agent": dep.get("to_agent", ""),
                "intent": dep.get("intent", ""),
                "reason": _truncate_value(str(dep.get("reason", "")), 100),
            }
        )

    artifacts = {}
    for key, value in list((snapshot.get("artifacts") or {}).items())[:8]:
        if not isinstance(value, dict):
            continue
        artifacts[key] = {
            "intent": value.get("intent", ""),
            "value": _truncate_value(str(value.get("value", "")), 120),
            "by_agent": value.get("by_agent", ""),
            "turn": value.get("turn", 0),
        }

    blockers = {}
    for key, value in list((snapshot.get("blockers") or {}).items())[:6]:
        if not isinstance(value, dict):
            continue
        blockers[key] = {
            "intent": value.get("intent", ""),
            "type": value.get("type", ""),
            "by_agent": value.get("by_agent", ""),
            "description": _truncate_value(str(value.get("description", "")), 100),
        }

    recent_failures = dict(list((snapshot.get("recent_failures") or {}).items())[:6])

    return {
        "summary": snapshot.get("summary", ""),
        "active_intents": active_intents,
        "done_intents": snapshot.get("done_intents", {}),
        "dependencies": dependencies,
        "artifacts": artifacts,
        "blockers": blockers,
        "recent_failures": recent_failures,
        "last_status_by_agent": snapshot.get("last_status_by_agent", {}),
        "recent_events": recent_events,
    }


def build_input_text(state: RunState, agent: AgentState, observation: dict) -> str:
    summary = build_coordination_summary(state)
    memory_snapshot = _compact_memory_snapshot(state.memory.get_snapshot())
    last_actions, last_results = agent.get_last_action()

    lines = []
    lines.append(f"GOAL: {state.goal}\n")
    lines.append(f"LATEST MESSAGES SUMMARY: {summary}\n\n")
    lines.append(f"MEMORY_SNAPSHOT: {json.dumps(memory_snapshot)}\n\n")
    if last_actions:
        compact_last = []
        for action, result in list(zip(last_actions, last_results))[-2:]:
            state_changed = None
            if isinstance(result.data, dict):
                state_changed = result.data.get("state_changed")
            compact_last.append(
                f"{action.type.value}:success={result.success},state_changed={state_changed},error={_truncate_value(result.error_message or 'none', 80)}"
            )
        lines.append(f"LAST_ACTION_RESULT: {' | '.join(compact_last)}\n\n")
    lines.append(f"BROWSER OBSERVATION: \n{observation}\n")

    return "".join(lines)


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _is_none_like(value: str) -> bool:
    return _normalize_text(value) in {"", "none", "null", "n/a", "na"}


def _record_event(state: RunState, event_type: str, msg: Message, intent: str, details: dict[str, Any] | None = None) -> None:
    state.memory.events.append(
        MemoryEvent(
            agent=msg.speaker,
            event_type=event_type,
            turn=msg.turn or 0,
            intent=intent or "unspecified",
            details=details or {},
        )
    )


def _find_dependency_target(state: RunState, needs: str) -> str | None:
    lowered = needs.lower()
    for agent in state.agents:
        if agent.name.lower() in lowered:
            return agent.name
    return None


def _action_signature(action: Action) -> str:
    if action.type.value == "navigate":
        return f"navigate|{action.url or ''}"
    if action.type.value == "click":
        if action.selector:
            return f"click|{action.selector}"
        return f"click|{action.x},{action.y}"
    if action.type.value == "click_index":
        return f"click_index|{action.index}"
    if action.type.value == "click_relative":
        return f"click_relative|{action.selector}|{action.rel_x},{action.rel_y}"
    if action.type.value == "scroll":
        return f"scroll|{action.direction}|{action.amount}"
    if action.type.value == "keypress":
        return f"keypress|{','.join(action.keys or [])}"
    if action.type.value == "type":
        return f"type|{action.selector}|len={len(action.text or '')}"
    if action.type.value == "wait":
        return f"wait|{action.wait_ms}"
    return f"unknown|{action.type.value}"


def _resolve_click_index(action: Action, observation: dict) -> tuple[Action | None, str | None]:
    if action.type.value != "click_index":
        return action, None
    elements = observation.get("elements", [])
    if not isinstance(action.index, int):
        return None, "click_index requires an integer index."
    if action.index < 0 or action.index >= len(elements):
        return None, f"click_index out of range: {action.index}"
    element = elements[action.index] or {}
    selector = element.get("selector")
    if selector:
        return Action(type=ActionType.CLICK, selector=selector), None
    x = element.get("x")
    y = element.get("y")
    width = element.get("width")
    height = element.get("height")
    if all(isinstance(v, (int, float)) for v in (x, y, width, height)):
        center_x = int(round(x + (width / 2)))
        center_y = int(round(y + (height / 2)))
        return Action(type=ActionType.CLICK, x=center_x, y=center_y), None
    return None, "click_index element missing selector and bbox."


def update_shared_memory(state: RunState, msg: Message) -> None:
    if msg.turn is None:
        return

    parsed = parse_coordination_fields(msg.message)
    intent = _normalize_text(parsed.get("INTENT", ""))
    intent_key = _get_intent_key_from_fields(parsed)
    status = _normalize_text(parsed.get("STATUS", "unknown"))
    needs = parsed.get("NEEDS", "")
    output = parsed.get("OUTPUT", "")

    if status == "done" and _is_none_like(output):
        status = "in_progress"
        _record_event(state, "done_blocked_no_output", msg, intent_key)
    if status == "done" and intent_key in state.memory.artifacts and intent_key in state.memory.done_intents:
        previous = state.memory.artifacts[intent_key].value
        if output.strip() == previous.strip():
            status = "in_progress"
            _record_event(state, "done_blocked_output_unchanged", msg, intent_key)

    state.memory.last_status_by_agent[msg.speaker] = status

    if intent_key and status == "in_progress":
        state.memory.active_intents[intent_key] = IntentLease(
            owner=msg.speaker,
            status=status,
            expires_at_turn=msg.turn + cfg.LAST_K_MESSAGES,
        )
        _record_event(state, "intent_claimed", msg, intent_key)
    elif intent_key and status == "done":
        if intent_key in state.memory.active_intents:
            state.memory.active_intents.pop(intent_key)
        state.memory.done_intents[intent_key] = msg.turn
        _record_event(state, "intent_released", msg, intent_key, {"reason": "done"})

    if not _is_none_like(output):
        artifact_key = intent_key or f"artifact-{msg.turn}"
        state.memory.artifacts[artifact_key] = Artifact(
            intent=intent_key or "unspecified",
            value=output,
            turn=msg.turn,
            by_agent=msg.speaker,
        )
        _record_event(state, "artifact_created", msg, intent_key, {"artifact_key": artifact_key})

    # dependencies
    state.memory.dependencies = [d for d in state.memory.dependencies if d.from_agent != msg.speaker]
    # no automatic dependency creation from waiting
    # prune resolved dependencies
    if state.memory.dependencies:
        state.memory.dependencies = [
            d for d in state.memory.dependencies if not _dependency_is_resolved(state, d)
        ]

    # blockers
    if status in {"blocked", "blocked_external"}:
        blocker_key = intent_key or f"{msg.speaker}:{msg.turn}"
        state.memory.blockers[blocker_key] = Blocker(
            intent=intent_key or "unspecified",
            type=BlockerType.permission,
            since_turn=msg.turn,
            description=needs[:200] if not _is_none_like(needs) else "blocked",
            by_agent=msg.speaker,
        )
        _record_event(state, "blocker_added", msg, intent_key)

    # failures
    for action, result in zip(msg.actions, msg.action_results):
        signature = _action_signature(action)
        if result.success:
            continue
        state.memory.failure_counts[signature] = state.memory.failure_counts.get(signature, 0) + 1
        _record_event(
            state,
            "action_failed",
            msg,
            intent_key,
            {"signature": signature, "error": (result.error_message or "")[:120]},
        )

    state.memory.summary = (
        f"intents={len(state.memory.active_intents)} "
        f"artifacts={len(state.memory.artifacts)} "
        f"deps={len(state.memory.dependencies)} "
        f"blockers={len(state.memory.blockers)}"
    )


def _last_agent_status(state: RunState, agent_name: str) -> str | None:
    for message in reversed(state.messages):
        if message.speaker != agent_name:
            continue
        fields = parse_coordination_fields(message.message)
        return fields.get("STATUS", "").strip().lower() or None
    return None


def _last_agent_fields(state: RunState, agent_name: str) -> dict[str, str]:
    for message in reversed(state.messages):
        if message.speaker != agent_name:
            continue
        fields = parse_coordination_fields(message.message)
        if fields:
            return fields
    return {}


def _is_verification_only_actions(actions: list[Action]) -> bool:
    if not actions:
        return True
    return all(action.type.value == "get_value" for action in actions)


def _is_low_impact_actions(actions: list[Action]) -> bool:
    if not actions:
        return True
    low_impact_types = {"get_value", "type", "keypress"}
    return all(action.type.value in low_impact_types for action in actions)


def _last_agent_had_no_action_in_progress(state: RunState, agent_name: str) -> bool:
    for message in reversed(state.messages):
        if message.speaker != agent_name:
            continue
        fields = parse_coordination_fields(message.message)
        if not fields:
            return False
        status = fields.get("STATUS", "").strip().lower()
        if status != "in_progress":
            return False
        return _is_verification_only_actions(message.actions)
    return False


def _consecutive_verification_turns_for_intent(state: RunState, agent_name: str, intent_key: str) -> int:
    if not intent_key:
        return 0
    count = 0
    for message in reversed(state.messages):
        if message.speaker != agent_name:
            continue
        fields = parse_coordination_fields(message.message)
        if not fields:
            continue
        if _get_intent_key_from_fields(fields) != intent_key:
            break
        status = fields.get("STATUS", "").strip().lower()
        if status != "in_progress":
            break
        if _is_verification_only_actions(message.actions):
            count += 1
            continue
        break
    return count


def _consecutive_same_low_impact_plan_for_intent(
    state: RunState,
    agent_name: str,
    intent_key: str,
    current_actions: list[Action],
) -> int:
    if not intent_key or not _is_low_impact_actions(current_actions):
        return 0
    current_sig = [_action_signature(a) for a in current_actions]
    count = 0
    for message in reversed(state.messages):
        if message.speaker != agent_name:
            continue
        fields = parse_coordination_fields(message.message)
        if not fields:
            continue
        if _get_intent_key_from_fields(fields) != intent_key:
            break
        if fields.get("STATUS", "").strip().lower() != "in_progress":
            break
        if not _is_low_impact_actions(message.actions):
            break
        prev_sig = [_action_signature(a) for a in message.actions]
        if prev_sig != current_sig:
            break
        count += 1
    return count


def _has_actionable_elements(observation: dict) -> bool:
    elements = observation.get("elements", [])
    if not isinstance(elements, list):
        return False
    for el in elements:
        if not isinstance(el, dict):
            continue
        if el.get("kind") != "interactive":
            continue
        if el.get("disabled"):
            continue
        if el.get("selector"):
            return True
    return False


def _observation_fingerprint(observation: dict) -> tuple[str, str, str]:
    if not isinstance(observation, dict):
        return "", "", ""
    title = _normalize_text(str(observation.get("title", "")))
    focused = _normalize_text(str(observation.get("focused_text", "")))
    visible = _normalize_text(str(observation.get("visible_text", "")))
    return title[:200], focused[:240], visible[:240]


def _has_meaningful_observation_delta(before_obs: dict, after_obs: dict) -> bool:
    before = _observation_fingerprint(before_obs)
    after = _observation_fingerprint(after_obs)
    return before != after


def _dependency_is_resolved(state: RunState, dep: Dependency) -> bool:
    if dep.intent in state.memory.done_intents and state.memory.done_intents[dep.intent] >= dep.since_turn:
        return True

    for artifact in state.memory.artifacts.values():
        if artifact.turn < dep.since_turn:
            continue
        if artifact.by_agent == dep.to_agent:
            return True
        if dep.intent != "unspecified" and artifact.intent == dep.intent:
            return True

    for msg in state.messages:
        if msg.turn is None or msg.turn < dep.since_turn:
            continue
        if msg.speaker != dep.to_agent:
            continue
        fields = parse_coordination_fields(msg.message)
        status = fields.get("STATUS", "").strip().lower()
        output = fields.get("OUTPUT", "")
        if status == "done":
            return True
        if not _is_none_like(output):
            return True

    return False


def _has_unresolved_dependency(state: RunState, agent_name: str) -> bool:
    pending = [d for d in state.memory.dependencies if d.from_agent == agent_name]
    if not pending:
        return False
    for dep in pending:
        if not _dependency_is_resolved(state, dep):
            return True
    return False


def _is_wait_dependency_resolved(state: RunState, target_agent: str) -> bool:
    fields = _last_agent_fields(state, target_agent)
    if not fields:
        return False
    status = fields.get("STATUS", "").strip().lower()
    output = fields.get("OUTPUT", "")
    if status == "done":
        return True
    if not _is_none_like(output):
        return True
    return False


def _is_needed_by_waiting_peer(state: RunState, agent_name: str) -> bool:
    latest = last_messages_for_each_agent(state)
    for speaker, fields in latest.items():
        if speaker == agent_name:
            continue
        status = fields.get("STATUS", "").strip().lower()
        if status != "waiting":
            continue
        needs = fields.get("NEEDS", "")
        if not _is_none_like(needs) and agent_name.lower() in needs.lower():
            return True
    return False


def _mentioned_agents_in_needs(state: RunState, needs: str) -> list[str]:
    lowered = needs.lower()
    mentioned = [agent.name for agent in state.agents if agent.name.lower() in lowered]
    # Stable/deterministic ordering for tie-breaks.
    return sorted(set(mentioned))


def _deadlock_breaker_target_agent(state: RunState) -> str | None:
    # Prefer unresolved dependency targets first.
    unresolved_targets: list[str] = []
    for dep in state.memory.dependencies:
        if _dependency_is_resolved(state, dep):
            continue
        unresolved_targets.append(dep.to_agent)
    if unresolved_targets:
        counts: dict[str, int] = {}
        for agent in unresolved_targets:
            counts[agent] = counts.get(agent, 0) + 1
        # Most-blocked target first; stable tie-break by name.
        return sorted(counts.keys(), key=lambda name: (-counts[name], name))[0]

    # Look at the most recent coordination message with a concrete NEEDS field.
    for msg in reversed(state.messages):
        fields = parse_coordination_fields(msg.message)
        if not fields:
            continue
        needs = fields.get("NEEDS", "")
        if _is_none_like(needs):
            continue
        mentioned = _mentioned_agents_in_needs(state, needs)
        if len(mentioned) == 1:
            return mentioned[0]
        if len(mentioned) > 1:
            # If both/many are mentioned, nudge one at a time (deterministic alternation).
            idx = state.next_turn % len(mentioned)
            return mentioned[idx]
    return None


def _should_break_wait_deadlock(state: RunState, agent_name: str) -> bool:
    statuses = state.memory.last_status_by_agent
    if not statuses:
        return False
    if not all(status == "waiting" for status in statuses.values()):
        return False
    # Prefer the target requested by the most recent NEEDS message.
    leader = _deadlock_breaker_target_agent(state)
    if not leader:
        # Fallback: alphabetically smallest waiting agent proceeds.
        leader = sorted(statuses.keys())[0]
    return agent_name == leader


async def _attempt_modal_dismiss(page) -> bool:
    try:
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(300)
        return True
    except Exception:
        return False

def extract_actions(response: str) -> list[Action]:
    match = re.search(r"(?im)^\s*ACTION\s*:\s*", response)
    if not match:
        return []

    payload = response[match.end():].strip()
    if not payload:
        return []

    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload, flags=re.IGNORECASE)
        payload = payload.split("```", 1)[0].strip()

    if payload.lower().startswith("null"):
        return []

    decoder = json.JSONDecoder()
    candidates = [payload]
    starts = [idx for idx in (payload.find("["), payload.find("{")) if idx != -1]
    candidates.extend(payload[idx:] for idx in sorted(set(starts)))

    for candidate in candidates:
        try:
            action_obj, _ = decoder.raw_decode(candidate)
            if isinstance(action_obj, dict):
                if validate_action_dict(action_obj):
                    return [parse_model_action(action_obj)]
                return []

            if isinstance(action_obj, list):
                parsed: list[Action] = []
                for item in action_obj:
                    if not isinstance(item, dict) or not validate_action_dict(item):
                        return []
                    parsed.append(parse_model_action(item))
                return parsed[:cfg.MAX_ACTIONS_PER_TURN]
        except (json.JSONDecodeError, TypeError):
            continue

    return []

def extract_message(response: str) -> str | None:
    message_match = re.search(r"(?im)^\s*MESSAGE\s*:\s*", response)
    if not message_match:
        return None

    action_match = re.search(r"(?im)^\s*ACTION\s*:\s*", response)
    start = message_match.end()
    end = action_match.start() if action_match and action_match.start() > start else len(response)
    payload = response[start:end].strip()
    return payload or None

async def run_agent_cycle(state: RunState, agent: AgentState) -> Message:
    generate_response_fn = PROVIDER_GEN[agent.provider]

    # Get the current page observation for the agent's browser
    observation = await agent.get_browser().build_observation()
    input_text = build_input_text(state, agent, observation)

    try:
        response = await asyncio.wait_for(
            asyncio.to_thread(
                generate_response_fn,
                model=agent.model,
                instructions=agent.system_prompt,
                input_text=input_text,
            ),
            timeout=cfg.LLM_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"LLM call timed out after {cfg.LLM_TIMEOUT_SECONDS}s for {agent.name}") from exc

    for attempt in range(2):
        message_text = extract_message(response) or response
        parsed_fields = parse_coordination_fields(message_text)
        claimed_intent = parsed_fields.get("INTENT", "")
        claimed_intent_key = _get_intent_key_from_fields(parsed_fields)
        if claimed_intent_key:
            message_text = _set_coordination_field(message_text, "INTENT_KEY", claimed_intent_key)
            parsed_fields["INTENT_KEY"] = claimed_intent_key
        claimed_status = parsed_fields.get("STATUS", "").strip().lower()
        claimed_needs = parsed_fields.get("NEEDS", "")
        reported_output = parsed_fields.get("OUTPUT", "none")
        if claimed_status == "done" and _is_none_like(reported_output):
            claimed_status = "in_progress"
            message_text = _set_coordination_field(message_text, "STATUS", "in_progress")
            message_text = _set_coordination_field(
                message_text,
                "NEXT",
                "Provide a concrete OUTPUT artifact before marking done.",
            )
        if (
            claimed_status == "done"
            and claimed_intent_key in state.memory.artifacts
            and claimed_intent_key in state.memory.done_intents
            and not _is_none_like(reported_output)
        ):
            previous = state.memory.artifacts[claimed_intent_key].value
            if reported_output.strip() == previous.strip():
                claimed_status = "in_progress"
                message_text = _set_coordination_field(message_text, "STATUS", "in_progress")
                message_text = _set_coordination_field(
                    message_text,
                    "NEXT",
                    "OUTPUT unchanged; produce new evidence before marking done.",
                )
        if claimed_status == "waiting" and not _is_none_like(claimed_needs):
            target = _find_dependency_target(state, claimed_needs)
            if target:
                dep = Dependency(
                    from_agent=agent.name,
                    to_agent=target,
                    intent=claimed_intent_key or "unspecified",
                    since_turn=state.next_turn,
                    reason=claimed_needs[:200],
                )
                if _dependency_is_resolved(state, dep):
                    claimed_status = "in_progress"
                    message_text = _set_coordination_field(message_text, "STATUS", "in_progress")
                    message_text = _set_coordination_field(message_text, "NEEDS", "none")
                    message_text = _set_coordination_field(
                        message_text,
                        "NEXT",
                        "Dependency resolved; proceed with the next concrete step.",
                    )
        if claimed_status == "waiting" and _is_none_like(claimed_needs):
            claimed_status = "in_progress"
            message_text = _set_coordination_field(message_text, "STATUS", "in_progress")
            message_text = _set_coordination_field(
                message_text,
                "NEXT",
                "Proceed with the next concrete step for this intent.",
            )
        if claimed_status == "waiting" and _is_needed_by_waiting_peer(state, agent.name):
            claimed_status = "in_progress"
            message_text = _set_coordination_field(message_text, "STATUS", "in_progress")
            message_text = _set_coordination_field(message_text, "NEEDS", "none")
            message_text = _set_coordination_field(
                message_text,
                "NEXT",
                "Another agent is waiting on you; perform a concrete action now.",
            )
        if claimed_status == "waiting" and _should_break_wait_deadlock(state, agent.name):
            claimed_status = "in_progress"
            message_text = _set_coordination_field(message_text, "STATUS", "in_progress")
            message_text = _set_coordination_field(message_text, "NEEDS", "none")
            message_text = _set_coordination_field(
                message_text,
                "NEXT",
                "Deadlock breaker: proceed with the next concrete step.",
            )
        actions = extract_actions(response)
        needed_by_waiting_peer = _is_needed_by_waiting_peer(state, agent.name)
        verification_only_actions = _is_verification_only_actions(actions)
        consecutive_verification_turns = _consecutive_verification_turns_for_intent(
            state=state,
            agent_name=agent.name,
            intent_key=claimed_intent_key,
        )
        has_actionable_elements = _has_actionable_elements(observation)
        same_low_impact_repeats = _consecutive_same_low_impact_plan_for_intent(
            state=state,
            agent_name=agent.name,
            intent_key=claimed_intent_key,
            current_actions=actions,
        )
        if (
            attempt == 0
            and claimed_status == "in_progress"
            and _is_none_like(claimed_needs)
            and has_actionable_elements
            and verification_only_actions
            and consecutive_verification_turns >= cfg.MAX_CONSECUTIVE_VERIFICATION_TURNS
        ):
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_response_fn,
                    model=agent.model,
                    instructions=agent.system_prompt,
                    input_text=input_text
                    + "\nENFORCEMENT: Verification-only cap reached. Perform at least one concrete UI action now (prefer click_index).",
                ),
                timeout=cfg.LLM_TIMEOUT_SECONDS,
            )
            continue
        if (
            attempt == 0
            and claimed_status == "in_progress"
            and needed_by_waiting_peer
            and _is_low_impact_actions(actions)
        ):
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_response_fn,
                    model=agent.model,
                    instructions=agent.system_prompt,
                    input_text=input_text
                    + "\nENFORCEMENT: Another agent is waiting on you. Take a concrete UI action now (not just monitoring/chat/get_value).",
                ),
                timeout=cfg.LLM_TIMEOUT_SECONDS,
            )
            continue
        if (
            attempt == 0
            and claimed_status == "in_progress"
            and _is_none_like(claimed_needs)
            and has_actionable_elements
            and same_low_impact_repeats >= cfg.MAX_CONSECUTIVE_LOW_IMPACT_REPEATS
        ):
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_response_fn,
                    model=agent.model,
                    instructions=agent.system_prompt,
                    input_text=input_text
                    + "\nENFORCEMENT: Repeated low-impact action plan detected. Choose a different concrete UI action now (prefer click_index on a non-input interactive element).",
                ),
                timeout=cfg.LLM_TIMEOUT_SECONDS,
            )
            continue
        if (
            attempt == 0
            and claimed_status == "in_progress"
            and _is_none_like(claimed_needs)
            and verification_only_actions
            and _last_agent_had_no_action_in_progress(state, agent.name)
        ):
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_response_fn,
                    model=agent.model,
                    instructions=agent.system_prompt,
                    input_text=input_text
                    + "\nENFORCEMENT: You must include at least one ACTION this turn.",
                ),
                timeout=cfg.LLM_TIMEOUT_SECONDS,
            )
            continue
        break
    if (
        claimed_status == "in_progress"
        and _is_none_like(claimed_needs)
        and _is_verification_only_actions(actions)
        and _last_agent_had_no_action_in_progress(state, agent.name)
    ):
        claimed_status = "blocked"
        message_text = _set_coordination_field(message_text, "STATUS", "blocked")
        message_text = _set_coordination_field(message_text, "NEEDS", "idle_action_required")
        message_text = _set_coordination_field(
            message_text,
            "NEXT",
            "No actions taken for two turns; provide at least one concrete ACTION.",
        )
    active_owner = find_active_intent_owner_by_key(
        state=state,
        intent_key=claimed_intent_key,
        exclude_agent=agent.name,
    )

    executed_actions: list[Action] = []
    action_results: list[ExecutionResult] = []
    observation_before_actions = observation
    if claimed_intent_key and claimed_intent_key in state.memory.done_intents and claimed_status != "done":
        alternate_key = _normalize_intent_key(claimed_intent)
        if not alternate_key or alternate_key == claimed_intent_key:
            alternate_key = f"{claimed_intent_key}_{agent.name.lower().replace(' ', '')}"[:120]
        message_text = _set_coordination_field(message_text, "INTENT_KEY", alternate_key)
        claimed_intent_key = alternate_key
    elif active_owner and claimed_status == "in_progress":
        message_text = (
            f"INTENT_KEY: {claimed_intent_key or _normalize_intent_key(claimed_intent)}\n"
            f"INTENT: {claimed_intent}\n"
            f"OWNER: {agent.name}\n"
            "STATUS: waiting\n"
            "OUTPUT: none\n"
            f"NEEDS: {active_owner} to finish current active ownership of this intent.\n"
            "NEXT: support verification or choose a complementary subtask."
        )
    else:
        for action in actions:
            resolved_action = action
            if action.type.value == "click_index":
                resolved_action, error = _resolve_click_index(action, observation)
                if not resolved_action:
                    executed_actions.append(action)
                    action_results.append(ExecutionResult(success=False, error_message=error or "click_index failed"))
                    if cfg.STOP_ON_ACTION_FAILURE:
                        break
                    continue
            signature = _action_signature(resolved_action)
            if state.memory.failure_counts.get(signature, 0) >= cfg.MAX_REPEAT_FAILURES_PER_ACTION:
                executed_actions.append(resolved_action)
                action_results.append(
                    ExecutionResult(
                        success=False,
                        error_message=f"Skipped action due to repeated failures: {signature}",
                    )
                )
                continue
            result = await agent.get_browser().execute(resolved_action)
            if (
                not result.success
                and result.error_message
                and "intercepts pointer events" in result.error_message
            ):
                dismissed = await _attempt_modal_dismiss(agent.get_browser().page)
                if dismissed:
                    result = await agent.get_browser().execute(resolved_action)
            executed_actions.append(resolved_action)
            action_results.append(result)
            if cfg.STOP_ON_ACTION_FAILURE and not result.success:
                break
        reported_output = parsed_fields.get("OUTPUT", "none")
        if _is_none_like(reported_output):
            for action, result in zip(executed_actions, action_results):
                if action.type.value != "get_value" or not result.success or not result.data:
                    continue
                value = str(result.data.get("value", "")).strip()
                if not value:
                    continue
                message_text = _set_coordination_field(message_text, "OUTPUT", value)
                break
        if _is_none_like(parse_coordination_fields(message_text).get("OUTPUT", "none")) and executed_actions:
            state_changed_signal = False
            for result in action_results:
                if not result.success:
                    continue
                data = result.data or {}
                if isinstance(data, dict) and data.get("state_changed") is True:
                    state_changed_signal = True
                    break
            observation_after_actions = await agent.get_browser().build_observation()
            if state_changed_signal or _has_meaningful_observation_delta(
                observation_before_actions,
                observation_after_actions,
            ):
                message_text = _set_coordination_field(message_text, "OUTPUT", "observed_state_transition")
    agent.add_action(executed_actions, action_results)

    return Message(
        speaker = agent.name,
        raw_response = response,
        message = message_text,
        actions = executed_actions,
        action_results = action_results,
        input_text = input_text,
        observation = observation,
    )
    
async def worker(state: RunState, agent: AgentState, after_turn: Callable[[RunState, AgentState, Message], Awaitable[None]] | None = None) -> None:
    while not state.stop_event.is_set():
        last_fields = _last_agent_fields(state, agent.name)
        last_status = (last_fields.get("STATUS", "") or _last_agent_status(state, agent.name) or "").strip().lower()
        last_needs = last_fields.get("NEEDS", "")
        if last_status == "waiting" and not _is_none_like(last_needs):
            target = _find_dependency_target(state, last_needs)
            if _is_needed_by_waiting_peer(state, agent.name):
                pass
            elif target and not _is_wait_dependency_resolved(state, target):
                if _should_break_wait_deadlock(state, agent.name):
                    pass
                else:
                    await asyncio.sleep(0.5)
                    continue
        if last_status in {"blocked", "blocked_external"}:
            if last_status == "blocked" and "idle_action_required" in (last_needs or ""):
                pass
            else:
                await asyncio.sleep(0.5)
                continue
        try:
            msg = await run_agent_cycle(state, agent)
        except Exception as exc:
            msg = Message(
                speaker=agent.name,
                raw_response=f"WORKER_ERROR: {exc}",
                message=f"I hit an internal error this turn: {exc}",
                actions=[],
                action_results=[],
            )
        end_execution = False

        async with state.message_lock:
            if state.next_turn >= cfg.MAX_TURNS_PER_ROUND:
                state.stop_event.set()
                break

            state.next_turn += 1
            msg.turn = state.next_turn
            state.messages.append(msg)
            update_shared_memory(state, msg)
            print(f"[TURN {msg.turn}] {msg.speaker}: {msg.message}")

            if state.next_turn >= cfg.MAX_TURNS_PER_ROUND:
                state.stop_event.set()
                end_execution = True
        
        if after_turn:
            try:
                await after_turn(state, agent, msg)
            except Exception:
                print(f"Error in after_turn callback for agent {agent.name}")
        
        if end_execution:
            break


async def run_group(
    state: RunState,
    after_turn: Callable[[RunState, AgentState, Message], Awaitable[None]] | None = None,
) -> None:
    tasks = [asyncio.create_task(worker(state, agent, after_turn)) for agent in state.agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Worker {state.agents[i].name} crashed: {result}")
