import re
from typing import Any

import village.config as cfg
from village.types import (
    Blocker,
    BlockerType,
    Dependency,
    IntentRecord,
    Message,
    RunState,
)
from village.coordination import (
    _get_intent_key_from_fields,
    _is_none_like,
    _normalize_text,
    _set_coordination_field,
    _truncate_value,
    parse_coordination_fields,
)


def _record_activity_turn(rec: IntentRecord) -> int:
    return max(rec.turn_completed or 0, rec.turn_claimed or 0)


def _latest_agent_record(
    state: RunState,
    agent_name: str,
    *,
    statuses: set[str] | None = None,
) -> IntentRecord | None:
    candidates = [
        rec
        for rec in state.memory.intents.values()
        if rec.owner == agent_name and (statuses is None or rec.status in statuses)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda rec: (_record_activity_turn(rec), rec.intent_key))


def _last_known_agent_status(state: RunState, agent_name: str) -> str:
    return _normalize_text(state.memory.last_status_by_agent.get(agent_name, "") or "")


def _last_agent_needs(state: RunState, agent_name: str) -> str:
    rec = _latest_agent_record(state, agent_name, statuses={"waiting", "blocked"})
    if rec is None:
        return ""
    return rec.needs or ""


def _last_agent_intent_key(state: RunState, agent_name: str) -> str:
    rec = _latest_agent_record(state, agent_name)
    return rec.intent_key if rec else ""


def _last_agent_done_turn(state: RunState, agent_name: str) -> int:
    rec = _latest_agent_record(state, agent_name, statuses={"done"})
    if rec is None:
        return 0
    return rec.turn_completed or rec.turn_claimed or 0


def _mentioned_agents_in_needs(state: RunState, needs: str) -> list[str]:
    if _is_none_like(needs):
        return []
    lowered = needs.lower()
    return sorted(
        {agent.name for agent in state.agents if agent.name.lower() in lowered}
    )


def _find_dependency_target(state: RunState, needs: str) -> str | None:
    mentioned = _mentioned_agents_in_needs(state, needs)
    return mentioned[0] if mentioned else None


_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_CODE_RE = re.compile(r"\b\d{4,8}\b")
_VERIFIABLE_UI_ACTION_TYPES = {
    "navigate",
    "click",
    "click_index",
    "click_relative",
    "keypress",
    "type",
}


def _dependency_output_satisfies_reason(output: str, reason: str) -> bool:
    if _is_none_like(output):
        return False

    normalized_reason = _normalize_text(reason)
    normalized_output = str(output).strip()
    lowered_output = _normalize_text(normalized_output)

    if any(token in normalized_reason for token in ("link", "url", "invite")):
        return bool(_URL_RE.search(normalized_output))
    if "email" in normalized_reason:
        return "@" in normalized_output or any(
            token in lowered_output for token in ("sent", "emailed", "delivered")
        )
    if any(
        token in normalized_reason for token in ("code", "otp", "pin", "verification")
    ):
        return bool(_CODE_RE.search(normalized_output))
    if any(
        token in normalized_reason
        for token in ("message", "reply", "response", "answer")
    ):
        return not _is_none_like(normalized_output)

    return False


def _dependency_expects_artifact(reason: str) -> bool:
    normalized_reason = _normalize_text(reason)
    return any(
        token in normalized_reason
        for token in (
            "link",
            "url",
            "invite",
            "email",
            "code",
            "otp",
            "pin",
            "verification",
            "message",
            "reply",
            "response",
            "answer",
        )
    )


def _message_attempted_verifiable_ui_action(msg: Message) -> bool:
    return any(
        action.type.value in _VERIFIABLE_UI_ACTION_TYPES for action in msg.actions
    )


def _message_has_verified_ui_progress(msg: Message) -> bool:
    for action, result in zip(msg.actions, msg.action_results):
        if action.type.value not in _VERIFIABLE_UI_ACTION_TYPES:
            continue
        if not result.success or not isinstance(result.data, dict):
            continue
        if result.data.get("state_changed") is True:
            return True
    return False


def _message_retrieved_concrete_value(msg: Message) -> bool:
    for action, result in zip(msg.actions, msg.action_results):
        if action.type.value != "get_value":
            continue
        if not result.success or not isinstance(result.data, dict):
            continue
        value = str(result.data.get("value", "")).strip()
        if value:
            return True
    return False


def _dependency_is_resolved(state: RunState, dependency: Dependency) -> bool:
    target = dependency.to_agent
    normalized_reason = _normalize_text(dependency.reason)
    expects_artifact = _dependency_expects_artifact(dependency.reason)

    if "active ownership of this intent" in normalized_reason:
        owner = state.memory.get_active_owner(dependency.intent, exclude="")
        return owner != target

    for rec in state.memory.intents.values():
        if rec.owner != target:
            continue
        if rec.status != "done":
            continue
        if (
            rec.turn_completed is not None
            and rec.turn_completed < dependency.since_turn
        ):
            continue
        if rec.turn_completed is None and rec.turn_claimed < dependency.since_turn:
            continue
        if _dependency_output_satisfies_reason(rec.output or "", dependency.reason):
            return True

    for msg in state.messages:
        if msg.turn is None or msg.turn < dependency.since_turn:
            continue
        if msg.speaker != target:
            continue
        fields = parse_coordination_fields(msg.message)
        status = _normalize_text(fields.get("status", ""))
        output = fields.get("output", fields.get("OUTPUT", ""))
        if _dependency_output_satisfies_reason(output, dependency.reason):
            return True
        if (
            not expects_artifact
            and status in {"done", "waiting", "blocked", "blocked_external"}
            and _message_has_verified_ui_progress(msg)
        ):
            return True

    return False


def build_coordination_summary(state: RunState) -> str:
    if not state.memory.intents:
        return "COORDINATION SUMMARY: none yet."

    lines = ["COORDINATION SUMMARY:"]
    for rec in state.memory.intents.values():
        output_preview = _truncate_value(rec.output or "none", 120)
        lines.append(
            f"- {rec.owner}: intent_key={rec.intent_key}; intent={rec.intent}; "
            f"status={rec.status}; output={output_preview}; needs={rec.needs or 'none'}"
        )
    return "\n".join(lines)


def _compact_memory_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    recent_events = [
        {
            f: event.get(f, "" if f != "turn" else 0)
            for f in ("agent", "event_type", "intent", "turn")
        }
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

    all_intents = snapshot.get("intents", {})
    active_intents = {}
    done_intents = {}
    for key, rec in list(all_intents.items()):
        if not isinstance(rec, dict):
            continue
        status = rec.get("status", "")
        entry = {
            "owner": rec.get("owner", ""),
            "intent": rec.get("intent", ""),
            "status": status,
            "output": _truncate_value(str(rec.get("output") or "none"), 120),
            "needs": rec.get("needs") or "none",
        }
        if status == "done":
            done_intents[key] = entry
        else:
            active_intents[key] = entry

    return {
        "summary": snapshot.get("summary", ""),
        "active_intents": dict(list(active_intents.items())[:6]),
        "done_intents": dict(list(done_intents.items())[:8]),
        "dependencies": dependencies,
        "blockers": {
            k: {f: v.get(f, "") for f in ("intent", "type", "by_agent", "description")}
            for k, v in list((snapshot.get("blockers") or {}).items())[:6]
            if isinstance(v, dict)
        },
        "recent_failures": dict(
            list((snapshot.get("recent_failures") or {}).items())[:6]
        ),
        "last_status_by_agent": snapshot.get("last_status_by_agent", {}),
        "recent_events": recent_events,
    }


def update_shared_memory(state: RunState, msg: Message) -> None:
    from village.agent_web_use_orchestration.browser_ops import _action_signature

    if msg.turn is None:
        raise RuntimeError(
            "Message turn must be assigned before updating shared memory."
        )

    parsed = parse_coordination_fields(msg.message)
    intent_key = _get_intent_key_from_fields(parsed)
    status = _normalize_text(parsed.get("status", "unknown"))
    output = parsed.get("output", "") or ""
    needs = parsed.get("needs", "") or ""
    existing = state.memory.get_record(msg.speaker, intent_key) if intent_key else None

    if status == "done" and _is_none_like(output):
        status = "in_progress"
        msg.message = _set_coordination_field(msg.message, "status", "in_progress")
        state.memory.record_event(
            agent=msg.speaker,
            event_type="done_blocked_no_output",
            turn=msg.turn,
            intent=intent_key,
        )

    if (
        status == "done"
        and _message_attempted_verifiable_ui_action(msg)
        and not _message_has_verified_ui_progress(msg)
        and not _message_retrieved_concrete_value(msg)
        and not (existing and existing.status == "done")
    ):
        status = "in_progress"
        output = "none"
        msg.message = _set_coordination_field(msg.message, "status", "in_progress")
        msg.message = _set_coordination_field(msg.message, "output", "none")
        msg.message = _set_coordination_field(
            msg.message,
            "next",
            "The completion was not verified in the interface. Try a different target or re-check the page before reporting done.",
        )
        state.memory.record_event(
            agent=msg.speaker,
            event_type="done_blocked_no_state_change",
            turn=msg.turn,
            intent=intent_key,
        )

    state.memory.last_status_by_agent[msg.speaker] = status

    if intent_key:
        state.memory.last_intent_key_by_agent[msg.speaker] = intent_key

    if intent_key:
        record_key = state.memory.intent_record_key(msg.speaker, intent_key)
        latest_done = state.memory.latest_record_for_intent(
            intent_key, statuses={"done"}
        )
        skip_record_write = False

        if status == "done":
            if latest_done and latest_done.status == "done":
                same_output = output.strip() == (latest_done.output or "").strip()
                same_agent = latest_done.owner == msg.speaker

                if same_agent or same_output:
                    if same_agent:
                        state.memory.intents[record_key] = IntentRecord(
                            intent_key=intent_key,
                            intent=parsed.get(
                                "intent", existing.intent if existing else intent_key
                            ),
                            owner=msg.speaker,
                            status="done",
                            output=output,
                            needs=needs if not _is_none_like(needs) else None,
                            turn_claimed=(
                                existing.turn_claimed if existing else msg.turn
                            ),
                            turn_completed=msg.turn,
                            expires_at_turn=0,
                        )
                    msg.message = _set_coordination_field(
                        msg.message,
                        "next",
                        "Intent already complete. Use a new intent_key for follow-up.",
                    )
                    event = "done_reaffirmed" if same_output else "done_reopen_blocked"
                    state.memory.record_event(
                        agent=msg.speaker,
                        event_type=event,
                        turn=msg.turn,
                        intent=intent_key,
                        details={
                            "existing_output": _truncate_value(
                                latest_done.output or "", 80
                            )
                        },
                    )
                    skip_record_write = True

                elif latest_done.owner != msg.speaker:
                    state.memory.record_event(
                        agent=msg.speaker,
                        event_type="done_handoff",
                        turn=msg.turn,
                        intent=intent_key,
                        details={"from_agent": latest_done.owner},
                    )

            if not skip_record_write:
                state.memory.intents[record_key] = IntentRecord(
                    intent_key=intent_key,
                    intent=parsed.get(
                        "intent", existing.intent if existing else intent_key
                    ),
                    owner=msg.speaker,
                    status="done",
                    output=output,
                    needs=needs if not _is_none_like(needs) else None,
                    turn_claimed=existing.turn_claimed if existing else msg.turn,
                    turn_completed=msg.turn,
                    expires_at_turn=0,
                )
                state.last_progress_turn = msg.turn
                state.memory.record_event(
                    agent=msg.speaker,
                    event_type="intent_released",
                    turn=msg.turn,
                    intent=intent_key,
                    details={"reason": "done"},
                )

        elif status == "in_progress":
            state.memory.intents[record_key] = IntentRecord(
                intent_key=intent_key,
                intent=parsed.get(
                    "intent", existing.intent if existing else intent_key
                ),
                owner=msg.speaker,
                status="in_progress",
                output=(
                    output
                    if not _is_none_like(output)
                    else (existing.output if existing else None)
                ),
                needs=needs if not _is_none_like(needs) else None,
                turn_claimed=existing.turn_claimed if existing else msg.turn,
                turn_completed=None,
                expires_at_turn=msg.turn + cfg.LAST_K_MESSAGES,
            )
            state.memory.record_event(
                agent=msg.speaker,
                event_type="intent_claimed",
                turn=msg.turn,
                intent=intent_key,
            )

        elif status == "waiting":
            if existing:
                existing.status = "waiting"
                existing.needs = needs if not _is_none_like(needs) else existing.needs
            else:
                state.memory.intents[record_key] = IntentRecord(
                    intent_key=intent_key,
                    intent=parsed.get("intent", intent_key),
                    owner=msg.speaker,
                    status="waiting",
                    needs=needs if not _is_none_like(needs) else None,
                    turn_claimed=msg.turn,
                    expires_at_turn=msg.turn + cfg.LAST_K_MESSAGES,
                )

            target = _find_dependency_target(state, needs)
            if target:
                dep = Dependency(
                    from_agent=msg.speaker,
                    to_agent=target,
                    intent=intent_key or "unspecified",
                    since_turn=msg.turn,
                    reason=needs[:200],
                )
                already_tracked = any(
                    d.from_agent == dep.from_agent
                    and d.to_agent == dep.to_agent
                    and d.intent == dep.intent
                    and d.reason == dep.reason
                    for d in state.memory.dependencies
                )
                if not already_tracked:
                    state.memory.dependencies.append(dep)

        elif status in {"blocked", "blocked_external"}:
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
                turn=msg.turn,
                intent=intent_key,
            )

    state.memory.intents = {
        k: v
        for k, v in state.memory.intents.items()
        if v.status != "in_progress"
        or v.expires_at_turn > msg.turn
        or state.memory.last_status_by_agent.get(v.owner) == "waiting"
    }

    if status != "waiting":
        state.memory.dependencies = [
            d for d in state.memory.dependencies if d.from_agent != msg.speaker
        ]
    state.memory.dependencies = [
        d for d in state.memory.dependencies if not _dependency_is_resolved(state, d)
    ]

    _update_failure_counts(state, msg, intent_key, _action_signature)
    _update_summary(state)


def _update_failure_counts(state, msg, intent_key, action_signature_fn):
    turn_has_verified_ui_progress = any(
        action.type.value in {"click", "click_index", "click_relative"}
        and result.success
        and isinstance(result.data, dict)
        and result.data.get("state_changed") is True
        for action, result in zip(msg.actions, msg.action_results)
    )

    for action, result in zip(msg.actions, msg.action_results):
        sig = action_signature_fn(action)
        if result.success:
            data = result.data if isinstance(result.data, dict) else {}
            if (
                action.type.value in {"click", "click_index", "click_relative"}
                and turn_has_verified_ui_progress
            ) or data.get("state_changed") is True:
                state.memory.failure_counts.pop(sig, None)
                continue
            if action.type.value == "get_value" and str(data.get("value", "")).strip():
                state.memory.failure_counts.pop(sig, None)
                continue

        if not result.success:
            state.memory.failure_counts[sig] = (
                state.memory.failure_counts.get(sig, 0) + 1
            )
            state.memory.record_event(
                agent=msg.speaker,
                event_type="action_failed",
                turn=msg.turn or 0,
                intent=intent_key,
                details={"signature": sig, "error": (result.error_message or "")[:120]},
            )
            continue

        if action.type.value not in {"click", "click_index", "click_relative"}:
            continue
        if turn_has_verified_ui_progress:
            continue
        if (
            not isinstance(result.data, dict)
            or result.data.get("state_changed") is not False
        ):
            continue

        state.memory.failure_counts[sig] = state.memory.failure_counts.get(sig, 0) + 1
        state.memory.record_event(
            agent=msg.speaker,
            event_type="action_noop",
            turn=msg.turn or 0,
            intent=intent_key,
            details={"signature": sig},
        )

    if len(state.memory.failure_counts) > cfg.MAX_FAILURE_COUNT_ENTRIES:
        keep = sorted(
            state.memory.failure_counts.items(), key=lambda kv: kv[1], reverse=True
        )[: cfg.MAX_FAILURE_COUNT_ENTRIES]
        state.memory.failure_counts = dict(keep)


def _update_summary(state: RunState) -> None:
    active = sum(1 for v in state.memory.intents.values() if v.status == "in_progress")
    done = sum(1 for v in state.memory.intents.values() if v.status == "done")
    state.memory.summary = (
        f"intents_active={active} intents_done={done} "
        f"deps={len(state.memory.dependencies)} "
        f"blockers={len(state.memory.blockers)}"
    )
