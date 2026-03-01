"""
memory_ops.py
-------------
Functions that read or write RunState.memory.
No async, no browser interaction.
"""
from typing import Any

import village.config as cfg
from village.types import (
    Artifact,
    Blocker,
    BlockerType,
    Dependency,
    IntentLease,
    Message,
    RunState,
)
from village.coordination import (
    _can_handoff_done_intent,
    _get_intent_key_from_fields,
    _is_none_like,
    _normalize_text,
    _set_coordination_field,
    _truncate_value,
    parse_coordination_fields,
)


# ---------------------------------------------------------------------------
# Message history helpers
# ---------------------------------------------------------------------------

def last_messages_for_each_agent(state: RunState) -> dict[str, dict[str, str]]:
    """Return the most recent parsed coordination fields for every agent that has spoken."""
    latest: dict[str, dict[str, str]] = {}
    for msg in reversed(list(state.messages)):
        if msg.speaker in latest:
            continue
        parsed = parse_coordination_fields(msg.message)
        if parsed:
            latest[msg.speaker] = parsed
    return latest


def _last_agent_fields(state: RunState, agent_name: str) -> dict[str, str]:
    return last_messages_for_each_agent(state).get(agent_name, {})


def _last_known_agent_status(state: RunState, agent_name: str, fields: dict[str, str] | None = None) -> str:
    if fields is None:
        fields = _last_agent_fields(state, agent_name)
    status = _normalize_text(fields.get("STATUS", ""))
    if status:
        return status
    return _normalize_text(state.memory.last_status_by_agent.get(agent_name, "") or "")


# ---------------------------------------------------------------------------
# Dependency resolution
# ---------------------------------------------------------------------------

def _dependency_is_resolved(state: RunState, dependency: Dependency) -> bool:
    """Checks whether a dependency has been satisfied since it was created."""
    for artifact in state.memory.artifacts.values():
        if artifact.turn < dependency.since_turn:
            continue
        if artifact.by_agent == dependency.to_agent:
            return True

    for msg in state.messages:
        if msg.turn is None or msg.turn < dependency.since_turn:
            continue
        if msg.speaker != dependency.to_agent:
            continue
        fields = parse_coordination_fields(msg.message)
        status = _normalize_text(fields.get("STATUS", ""))
        output = fields.get("OUTPUT", "")
        if status == "done":
            return True
        if not _is_none_like(output):
            return True

    return False


# ---------------------------------------------------------------------------
# Coordination summary (used to build LLM prompt input)
# ---------------------------------------------------------------------------

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
            f"- {speaker}: intent_key={intent_key}; task_id={task_id}; intent={intent}; "
            f"status={status}; output={output}; needs={needs}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Memory snapshot compaction (used to build LLM prompt input)
# ---------------------------------------------------------------------------

def _compact_dict(
    source: dict | None,
    limit: int,
    fields: list[str],
    truncate: dict[str, int] | None = None,
) -> dict:
    """Slice a snapshot dict to `limit` entries, keeping only `fields`, with optional truncation."""
    out = {}
    for key, value in list((source or {}).items())[:limit]:
        if not isinstance(value, dict):
            continue
        row = {f: value.get(f, "") for f in fields}
        for f, max_len in (truncate or {}).items():
            if f in row:
                row[f] = _truncate_value(str(row[f]), max_len)
        out[key] = row
    return out


def _compact_memory_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    # Show the 8 most recent events so agents have enough context to understand
    # what their peers have been doing across the last several turns.
    recent_events = [
        {f: event.get(f, "" if f != "turn" else 0) for f in ("agent", "event_type", "intent", "turn")}
        for event in (snapshot.get("recent_events") or [])[-8:]
        if isinstance(event, dict)
    ]

    dependencies = [
        {
            "from_agent": dep.get("from_agent", ""),
            "to_agent": dep.get("to_agent", ""),
            "intent": dep.get("intent", ""),
            "reason": _truncate_value(str(dep.get("reason", "")), 100),
        }
        for dep in (snapshot.get("dependencies") or [])[:6]
        if isinstance(dep, dict)
    ]

    return {
        "summary": snapshot.get("summary", ""),
        "active_intents": _compact_dict(
            snapshot.get("active_intents"), 6,
            ["owner", "status", "expires_at_turn"],
        ),
        "done_intents": snapshot.get("done_intents", {}),
        "dependencies": dependencies,
        "artifacts": _compact_dict(
            snapshot.get("artifacts"), 8,
            ["intent", "value", "by_agent", "turn"],
            truncate={"value": 120},
        ),
        "blockers": _compact_dict(
            snapshot.get("blockers"), 6,
            ["intent", "type", "by_agent", "description"],
            truncate={"description": 100},
        ),
        "recent_failures": dict(list((snapshot.get("recent_failures") or {}).items())[:6]),
        "last_status_by_agent": snapshot.get("last_status_by_agent", {}),
        "recent_events": recent_events,
    }


# ---------------------------------------------------------------------------
# Shared memory mutation
# ---------------------------------------------------------------------------

def update_shared_memory(state: RunState, msg: Message) -> None:
    # Deferred import to avoid circular dependency at module level.
    from village.agent_web_use_orchestration.browser_ops import _action_signature

    if msg.turn is None:
        return

    parsed = parse_coordination_fields(msg.message)
    intent_key = _get_intent_key_from_fields(parsed)
    status = _normalize_text(parsed.get("STATUS", "unknown"))
    needs = parsed.get("NEEDS", "")
    output = parsed.get("OUTPUT", "")
    preserve_existing_artifact = False

    # Patch msg.message when we override STATUS so agents reading history see
    # the corrected value rather than the stale one from the LLM response.
    if status == "done" and _is_none_like(output):
        status = "in_progress"
        msg.message = _set_coordination_field(msg.message, "STATUS", "in_progress")
        state.memory.record_event(
            agent=msg.speaker,
            event_type="done_blocked_no_output",
            turn=msg.turn or 0,
            intent=intent_key,
        )

    if status == "done" and intent_key in state.memory.artifacts and intent_key in state.memory.done_intents:
        previous = state.memory.artifacts[intent_key].value
        previous_agent = state.memory.artifacts[intent_key].by_agent
        allow_handoff = _can_handoff_done_intent(
            previous_output=previous,
            new_output=output,
            previous_agent=previous_agent,
            current_agent=msg.speaker,
        )
        if allow_handoff:
            state.memory.record_event(
                agent=msg.speaker,
                event_type="done_handoff",
                turn=msg.turn or 0,
                intent=intent_key,
                details={"from_agent": previous_agent},
            )
        elif output.strip() == previous.strip():
            preserve_existing_artifact = True
            msg.message = _set_coordination_field(
                msg.message,
                "NEXT",
                "This intent is already complete. Keep it done and use a new INTENT_KEY for any follow-up.",
            )
            state.memory.record_event(
                agent=msg.speaker,
                event_type="done_reaffirmed",
                turn=msg.turn or 0,
                intent=intent_key,
            )
        else:
            preserve_existing_artifact = True
            msg.message = _set_coordination_field(
                msg.message,
                "NEXT",
                "This intent is already complete. Use a new INTENT_KEY for a new artifact or follow-up.",
            )
            state.memory.record_event(
                agent=msg.speaker,
                event_type="done_reopen_blocked",
                turn=msg.turn or 0,
                intent=intent_key,
                details={
                    "existing_output": _truncate_value(previous, 80),
                    "new_output": _truncate_value(output, 80),
                },
            )

    state.memory.last_status_by_agent[msg.speaker] = status

    if intent_key and status == "in_progress":
        state.memory.active_intents[intent_key] = IntentLease(
            owner=msg.speaker,
            status=status,
            expires_at_turn=msg.turn + cfg.LAST_K_MESSAGES,
        )
        state.memory.record_event(
            agent=msg.speaker,
            event_type="intent_claimed",
            turn=msg.turn or 0,
            intent=intent_key,
        )
    elif intent_key and status == "done":
        state.memory.active_intents.pop(intent_key, None)
        if not preserve_existing_artifact:
            state.memory.done_intents[intent_key] = msg.turn
            state.memory.record_event(
                agent=msg.speaker,
                event_type="intent_released",
                turn=msg.turn or 0,
                intent=intent_key,
                details={"reason": "done"},
            )

    if not _is_none_like(output) and not preserve_existing_artifact:
        artifact_key = intent_key or f"artifact-{msg.turn}"
        state.memory.artifacts[artifact_key] = Artifact(
            intent=intent_key or "unspecified",
            value=output,
            turn=msg.turn,
            by_agent=msg.speaker,
        )
        state.memory.record_event(
            agent=msg.speaker,
            event_type="artifact_created",
            turn=msg.turn or 0,
            intent=intent_key,
            details={"artifact_key": artifact_key},
        )

    # Only clear this agent's deps when it is no longer waiting.
    # Wiping them on every message would drop valid deps mid-wait.
    if status != "waiting":
        state.memory.dependencies = [d for d in state.memory.dependencies if d.from_agent != msg.speaker]
    # Prune any globally resolved dependencies.
    state.memory.dependencies = [
        d for d in state.memory.dependencies if not _dependency_is_resolved(state, d)
    ]

    # Do NOT evict the lease of a waiting agent just because it hasn't spoken
    # in LAST_K_MESSAGES turns. A waiting agent is deliberately paused; silently
    # dropping its lease would let another agent claim the same intent and create
    # duplicate work. Only evict leases for agents that are genuinely absent.
    state.memory.active_intents = {
        k: v for k, v in state.memory.active_intents.items()
        if v.expires_at_turn > msg.turn
        or state.memory.last_status_by_agent.get(v.owner) == "waiting"
    }

    if status in {"blocked", "blocked_external"}:
        blocker_key = intent_key or f"{msg.speaker}:{msg.turn}"
        state.memory.blockers[blocker_key] = Blocker(
            intent=intent_key or "unspecified",
            type=BlockerType.permission,
            since_turn=msg.turn,
            description=needs[:200] if not _is_none_like(needs) else "blocked",
            by_agent=msg.speaker,
        )
        state.memory.record_event(
            agent=msg.speaker,
            event_type="blocker_added",
            turn=msg.turn or 0,
            intent=intent_key,
        )

    for action, result in zip(msg.actions, msg.action_results):
        if result.success:
            continue
        signature = _action_signature(action)
        state.memory.failure_counts[signature] = state.memory.failure_counts.get(signature, 0) + 1
        state.memory.record_event(
            agent=msg.speaker,
            event_type="action_failed",
            turn=msg.turn or 0,
            intent=intent_key,
            details={"signature": signature, "error": (result.error_message or "")[:120]},
        )

    state.memory.summary = (
        f"intents={len(state.memory.active_intents)} "
        f"artifacts={len(state.memory.artifacts)} "
        f"deps={len(state.memory.dependencies)} "
        f"blockers={len(state.memory.blockers)}"
    )