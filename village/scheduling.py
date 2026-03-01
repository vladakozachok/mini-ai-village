"""
scheduling.py
-------------
Agent coordination logic: deadlock detection, dependency tracking,
turn-gate decisions, and enforcement policy.
No async, no browser interaction.
"""
from collections import Counter
from typing import Callable

import village.config as cfg
from village.types import Dependency, Message, RunState, AgentState
from village.coordination import (
    _get_intent_key_from_fields,
    _is_none_like,
    _normalize_text,
    _set_coordination_field,
    parse_coordination_fields,
)
from village.memory_ops import (
    _dependency_is_resolved,
    last_messages_for_each_agent,
)
from village.agent_web_use_orchestration.browser_ops import (
    _action_signature,
    _is_low_impact_actions,
)


# ---------------------------------------------------------------------------
# Intent ownership
# ---------------------------------------------------------------------------

def find_active_intent_owner_by_key(state: RunState, intent_key: str, exclude_agent: str) -> str | None:
    if not intent_key:
        return None
    latest = last_messages_for_each_agent(state)
    for speaker, fields in latest.items():
        if speaker == exclude_agent:
            continue
        status = _normalize_text(fields.get("STATUS", ""))
        claimed_key = _get_intent_key_from_fields(fields)
        if status == "in_progress" and claimed_key == intent_key:
            return fields.get("OWNER") or speaker
    return None


# ---------------------------------------------------------------------------
# Dependency targeting
# ---------------------------------------------------------------------------

def _mentioned_agents_in_needs(state: RunState, needs: str) -> list[str]:
    if _is_none_like(needs):
        return []
    lowered = needs.lower()
    mentioned = [agent.name for agent in state.agents if agent.name.lower() in lowered]
    return sorted(set(mentioned))


def _find_dependency_target(state: RunState, needs: str) -> str | None:
    mentioned = _mentioned_agents_in_needs(state, needs)
    return mentioned[0] if mentioned else None


def _wait_dependency_resolved(
    state: RunState,
    agent_name: str,
    intent_key: str,
    needs: str,
) -> bool:
    target = _find_dependency_target(state, needs)
    if target is None:
        return True

    # Engine-generated ownership waits should clear as soon as the target is
    # no longer actively holding the same intent, even if they did not mark it
    # done. Otherwise a released owner can sit in STATUS=waiting and still keep
    # the next agent asleep forever.
    if "active ownership of this intent" in needs.lower():
        latest = last_messages_for_each_agent(state).get(target, {})
        target_intent_key = _get_intent_key_from_fields(latest)
        target_status = _normalize_text(latest.get("STATUS", ""))
        if target_intent_key != (intent_key or "unspecified") or target_status != "in_progress":
            return True

    tracked = [
        dep
        for dep in state.memory.dependencies
        if dep.from_agent == agent_name
        and dep.to_agent == target
        and dep.intent in {intent_key or "unspecified", "unspecified"}
    ]
    if tracked:
        return all(_dependency_is_resolved(state, dep) for dep in tracked)

    since_turn = 0
    for msg in reversed(state.messages):
        if msg.speaker != agent_name or msg.turn is None:
            continue
        fields = parse_coordination_fields(msg.message)
        if _normalize_text(fields.get("STATUS", "")) != "waiting":
            continue
        if target not in _mentioned_agents_in_needs(state, fields.get("NEEDS", "")):
            continue
        since_turn = msg.turn
        break

    fallback = Dependency(
        from_agent=agent_name,
        to_agent=target,
        intent=intent_key or "unspecified",
        since_turn=since_turn,
        reason=needs[:200],
    )
    return _dependency_is_resolved(state, fallback)


# ---------------------------------------------------------------------------
# Peer wait / deadlock detection
# ---------------------------------------------------------------------------

def _is_needed_by_peer(state: RunState, agent_name: str) -> bool:
    latest = last_messages_for_each_agent(state)
    for speaker, fields in latest.items():
        if speaker == agent_name:
            continue
        needs = fields.get("NEEDS", "")
        if _is_none_like(needs):
            continue
        if agent_name in _mentioned_agents_in_needs(state, needs):
            return True
    return False


def _deadlock_breaker_leader(state: RunState) -> str | None:
    statuses = state.memory.last_status_by_agent

    # Only fire if every registered agent has reported in and all are waiting.
    # Absent agents haven't started yet — they are not waiting.
    all_agent_names = {a.name for a in state.agents}
    if not all_agent_names.issubset(statuses.keys()):
        return None
    if not all(statuses[name] == "waiting" for name in all_agent_names):
        return None

    # Prefer the agent most-needed by unresolved dependencies.
    unresolved_targets = [
        dep.to_agent for dep in state.memory.dependencies
        if not _dependency_is_resolved(state, dep)
    ]
    if unresolved_targets:
        counts = Counter(unresolved_targets)
        return sorted(counts.keys(), key=lambda name: (-counts[name], name))[0]

    # Fall back to the most recent message with a concrete NEEDS.
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
            idx = state.next_turn % len(mentioned)
            return mentioned[idx]

    return sorted(statuses.keys())[0]


# ---------------------------------------------------------------------------
# Waiting status resolution
# ---------------------------------------------------------------------------

def _resolve_waiting_status(
    state: RunState,
    agent: AgentState,
    claimed_intent_key: str,
    claimed_needs: str,
    message_text: str,
) -> tuple[str, str]:
    """
    Evaluate whether a 'waiting' status should be overridden to 'in_progress'.
    Returns the (possibly updated) status string and message_text.

    Side-effect: when a genuine unresolved dependency is detected it is persisted
    into state.memory.dependencies so that deadlock detection and
    _wait_dependency_resolved can find it on future turns.
    """
    next_message: str | None = None

    if not _is_none_like(claimed_needs):
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
                # Dependency already satisfied — clear the wait and move on.
                message_text = _set_coordination_field(message_text, "NEEDS", "none")
                next_message = "Dependency resolved; proceed with the next concrete step."
            else:
                # Dependency is real and unresolved — persist it so that
                # _deadlock_breaker_leader can weight it correctly and
                # _wait_dependency_resolved sees it on future turns.
                # Avoid duplicates.
                already_tracked = any(
                    d.from_agent == agent.name
                    and d.to_agent == target
                    and d.intent == (claimed_intent_key or "unspecified")
                    for d in state.memory.dependencies
                )
                if not already_tracked:
                    state.memory.dependencies.append(dep)
        # If needs is set but names no known agent, allow waiting to stand.
        # Fall through with next_message=None → status stays "waiting".

    elif _is_needed_by_peer(state, agent.name):
        next_message = "Another agent explicitly needs you; perform a concrete action now."
    elif _deadlock_breaker_leader(state) == agent.name:
        next_message = "Deadlock breaker: proceed with the next concrete step."
    # No else: an agent that says STATUS: waiting with no NEEDS and no waiting peers
    # is allowed to stay waiting — it may be pausing intentionally before re-checking.

    if next_message is not None:
        message_text = _set_coordination_field(message_text, "STATUS", "in_progress")
        message_text = _set_coordination_field(message_text, "NEXT", next_message)
        return "in_progress", message_text

    return "waiting", message_text


# ---------------------------------------------------------------------------
# Consecutive-turn pattern detection
# ---------------------------------------------------------------------------

def _last_agent_had_no_action_in_progress(state: RunState, agent_name: str) -> bool:
    """Returns True if the agent's most recent in-progress turn had no concrete actions."""
    for message in reversed(state.messages):
        if message.speaker != agent_name:
            continue
        fields = parse_coordination_fields(message.message)
        if not fields:
            return False
        if _normalize_text(fields.get("STATUS", "")) != "in_progress":
            return False
        return _is_low_impact_actions(message.actions, strict=True)
    return False


def _consecutive_turns_matching(
    state: RunState,
    agent_name: str,
    intent_key: str,
    predicate: Callable[[Message], bool],
) -> int:
    """Count consecutive recent in-progress turns for agent+intent where predicate holds."""
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
        if _normalize_text(fields.get("STATUS", "")) != "in_progress":
            break
        if predicate(message):
            count += 1
        else:
            break
    return count


def _consecutive_verification_turns_for_intent(state: RunState, agent_name: str, intent_key: str) -> int:
    """Count consecutive in-progress turns where the agent only used get_value (or no actions)."""
    return _consecutive_turns_matching(
        state, agent_name, intent_key,
        lambda msg: _is_low_impact_actions(msg.actions, strict=True),
    )


def _consecutive_same_low_impact_plan_for_intent(
    state: RunState,
    agent_name: str,
    intent_key: str,
    current_actions: list,
) -> int:
    """Count consecutive in-progress turns where the agent repeated the exact same low-impact action plan."""
    if not intent_key or not _is_low_impact_actions(current_actions):
        return 0
    current_sig = [_action_signature(a) for a in current_actions]
    return _consecutive_turns_matching(
        state, agent_name, intent_key,
        lambda msg: (
            _is_low_impact_actions(msg.actions)
            and [_action_signature(a) for a in msg.actions] == current_sig
        ),
    )


# ---------------------------------------------------------------------------
# Enforcement policy
# ---------------------------------------------------------------------------

def _enforce_verification_cap(
    *, claimed_status, claimed_needs, has_actionable_elements,
    verification_only_actions, consecutive_verification_turns, **_
) -> bool:
    return (
        claimed_status == "in_progress"
        and _is_none_like(claimed_needs)
        and has_actionable_elements
        and verification_only_actions
        and consecutive_verification_turns >= cfg.MAX_CONSECUTIVE_VERIFICATION_TURNS
    )


def _enforce_waiting_peer(*, claimed_status, needed_by_peer, actions, **_) -> bool:
    return (
        claimed_status == "in_progress"
        and needed_by_peer
        and _is_low_impact_actions(actions)
    )


def _enforce_low_impact_repeat(
    *, claimed_status, claimed_needs, has_actionable_elements, same_low_impact_repeats, **_
) -> bool:
    return (
        claimed_status == "in_progress"
        and _is_none_like(claimed_needs)
        and has_actionable_elements
        and same_low_impact_repeats >= cfg.MAX_CONSECUTIVE_LOW_IMPACT_REPEATS
    )


# Ordered list of (predicate, enforcement_message). First match wins.
_ENFORCEMENT_CHECKS: list[tuple] = [
    (
        _enforce_verification_cap,
        "\nENFORCEMENT: Verification-only cap reached. Perform at least one concrete UI action now (prefer click_index).",
    ),
    (
        _enforce_waiting_peer,
        "\nENFORCEMENT: Another agent explicitly needs you. Take a concrete UI action now (not just monitoring/chat/get_value).",
    ),
    (
        _enforce_low_impact_repeat,
        "\nENFORCEMENT: Repeated low-impact action plan detected. Choose a different concrete UI action now (prefer click_index on a non-input interactive element).",
    ),
]


def _get_enforcement_suffix(**ctx) -> str | None:
    """Return the enforcement suffix for a retry, or None if no retry is needed."""
    for predicate, message in _ENFORCEMENT_CHECKS:
        if predicate(**ctx):
            return message
    return None
