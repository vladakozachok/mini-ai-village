import logging
from collections import Counter
from dataclasses import dataclass, field
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
from village.memory_ops import _dependency_is_resolved
from village.agent_web_use_orchestration.browser_ops import (
    _action_signature,
    _is_low_impact_actions,
)

logger = logging.getLogger(__name__)


def find_active_intent_owner_by_key(
    state: RunState, intent_key: str, exclude_agent: str
) -> str | None:
    return state.memory.get_active_owner(intent_key, exclude=exclude_agent)


def _mentioned_agents_in_needs(state: RunState, needs: str) -> list[str]:
    if _is_none_like(needs):
        return []
    lowered = needs.lower()
    return sorted({a.name for a in state.agents if a.name.lower() in lowered})


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

    if "active ownership of this intent" in needs.lower():
        owner = state.memory.get_active_owner(intent_key, exclude="")
        if owner != target:
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
        if _normalize_text(fields.get("status", "")) != "waiting":
            continue
        if target not in _mentioned_agents_in_needs(state, fields.get("needs", "")):
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


def _is_needed_by_peer(state: RunState, agent_name: str) -> bool:
    for rec in state.memory.intents.values():
        if rec.owner == agent_name:
            continue
        if state.memory.last_intent_key_by_agent.get(rec.owner) != rec.intent_key:
            continue
        if rec.status not in {"waiting", "blocked", "done"}:
            continue
        needs = rec.needs or ""
        if _is_none_like(needs):
            continue
        if agent_name not in _mentioned_agents_in_needs(state, needs):
            continue
        dep = Dependency(
            from_agent=rec.owner,
            to_agent=agent_name,
            intent=rec.intent_key or "unspecified",
            since_turn=rec.turn_completed or rec.turn_claimed or 0,
            reason=needs[:200],
        )
        if not _dependency_is_resolved(state, dep):
            return True
    return False


def _peer_progress_since_done(state: RunState, agent_name: str) -> bool:
    from village.memory_ops import _last_agent_done_turn

    done_turn = _last_agent_done_turn(state, agent_name)
    if done_turn <= 0:
        return False

    for msg in reversed(state.messages):
        if msg.turn is None:
            continue
        if msg.turn <= done_turn:
            break
        if msg.speaker == agent_name:
            continue
        return True

    return False


def _deadlock_breaker_leader(state: RunState) -> str | None:
    statuses = state.memory.last_status_by_agent
    all_agent_names = {a.name for a in state.agents}

    if not all_agent_names.issubset(statuses.keys()):
        return None
    if not all(statuses[name] == "waiting" for name in all_agent_names):
        return None

    unresolved_targets = [
        dep.to_agent
        for dep in state.memory.dependencies
        if not _dependency_is_resolved(state, dep)
    ]
    if unresolved_targets:
        counts = Counter(unresolved_targets)
        return sorted(counts.keys(), key=lambda name: (-counts[name], name))[0]

    for msg in reversed(state.messages):
        fields = parse_coordination_fields(msg.message)
        if not fields:
            continue
        needs = fields.get("needs", "")
        if _is_none_like(needs):
            continue
        mentioned = _mentioned_agents_in_needs(state, needs)
        if len(mentioned) == 1:
            return mentioned[0]
        if len(mentioned) > 1:
            return mentioned[state.next_turn % len(mentioned)]

    return sorted(statuses.keys())[0]


def _resolve_waiting_status(
    state: RunState,
    agent: AgentState,
    claimed_intent_key: str,
    claimed_needs: str,
    message_text: str,
) -> tuple[str, str]:
    next_message: str | None = None

    if not _is_none_like(claimed_needs):
        target = _find_dependency_target(state, claimed_needs)
        if target:
            dep = Dependency(
                from_agent=agent.name,
                to_agent=target,
                intent=claimed_intent_key or "unspecified",
                since_turn=state.next_turn + 1,
                reason=claimed_needs[:200],
            )
            if _dependency_is_resolved(state, dep):
                message_text = _set_coordination_field(message_text, "NEEDS", "none")
                next_message = (
                    "Dependency resolved; proceed with the next concrete step."
                )
            else:
                already_tracked = any(
                    d.from_agent == agent.name
                    and d.to_agent == target
                    and d.intent == (claimed_intent_key or "unspecified")
                    for d in state.memory.dependencies
                )
                if not already_tracked:
                    state.memory.dependencies.append(dep)

    elif _is_needed_by_peer(state, agent.name):
        next_message = (
            "Another agent explicitly needs you; perform a concrete action now."
        )
    elif _deadlock_breaker_leader(state) == agent.name:
        next_message = "Deadlock breaker: proceed with the next concrete step."

    if next_message is not None:
        message_text = _set_coordination_field(message_text, "STATUS", "in_progress")
        message_text = _set_coordination_field(message_text, "NEXT", next_message)
        return "in_progress", message_text

    return "waiting", message_text


def _consecutive_turns_matching(
    state: RunState,
    agent_name: str,
    intent_key: str,
    predicate: Callable[[Message], bool],
) -> int:
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
        if _normalize_text(fields.get("status", "")) != "in_progress":
            break
        if predicate(message):
            count += 1
        else:
            break
    return count


def _consecutive_verification_turns_for_intent(
    state: RunState,
    agent_name: str,
    intent_key: str,
) -> int:
    return _consecutive_turns_matching(
        state,
        agent_name,
        intent_key,
        lambda msg: _is_low_impact_actions(msg.actions, strict=True),
    )


def _consecutive_same_low_impact_plan_for_intent(
    state: RunState,
    agent_name: str,
    intent_key: str,
    current_actions: list,
) -> int:
    if not intent_key or not _is_low_impact_actions(current_actions):
        return 0
    current_sig = [_action_signature(a) for a in current_actions]
    return _consecutive_turns_matching(
        state,
        agent_name,
        intent_key,
        lambda msg: (
            _is_low_impact_actions(msg.actions)
            and [_action_signature(a) for a in msg.actions] == current_sig
        ),
    )


def _message_had_noop_ui_attempt(msg: Message) -> bool:
    if not msg.actions or not msg.action_results:
        return False

    saw_ui_action = False
    for action, result in zip(msg.actions, msg.action_results):
        action_type = action.type.value
        if action_type not in {"click", "click_index", "click_relative", "navigate"}:
            continue
        saw_ui_action = True
        if not result.success:
            return False
        if not isinstance(result.data, dict):
            return False
        if result.data.get("state_changed") is not False:
            return False
    return saw_ui_action


def _consecutive_noop_ui_turns_for_intent(
    state: RunState,
    agent_name: str,
    intent_key: str,
) -> int:
    return _consecutive_turns_matching(
        state,
        agent_name,
        intent_key,
        _message_had_noop_ui_attempt,
    )


def _last_turn_had_screenshot_for_intent(
    state: RunState,
    agent_name: str,
    intent_key: str,
) -> bool:
    if not intent_key:
        return False
    for message in reversed(state.messages):
        if message.speaker != agent_name:
            continue
        fields = parse_coordination_fields(message.message)
        if _get_intent_key_from_fields(fields) != intent_key:
            continue
        return any(action.type.value == "screenshot" for action in message.actions)
    return False


@dataclass
class EnforcementContext:
    needed_by_peer: bool
    consecutive_verification_turns: int
    same_low_impact_repeats: int
    consecutive_noop_ui_turns: int
    has_actionable_elements: bool
    last_turn_had_screenshot: bool = False

    claimed_status: str = "in_progress"
    claimed_needs: str = "none"
    verification_only_actions: bool = False
    actions: list = field(default_factory=list)


def build_enforcement_context(
    *,
    state: RunState,
    agent_name: str,
    intent_key: str,
    observation: dict,
) -> EnforcementContext:
    from village.agent_web_use_orchestration.browser_ops import _has_actionable_elements

    return EnforcementContext(
        needed_by_peer=_is_needed_by_peer(state, agent_name),
        consecutive_verification_turns=_consecutive_verification_turns_for_intent(
            state=state,
            agent_name=agent_name,
            intent_key=intent_key,
        ),
        same_low_impact_repeats=_consecutive_same_low_impact_plan_for_intent(
            state=state,
            agent_name=agent_name,
            intent_key=intent_key,
            current_actions=[],
        ),
        consecutive_noop_ui_turns=_consecutive_noop_ui_turns_for_intent(
            state=state,
            agent_name=agent_name,
            intent_key=intent_key,
        ),
        has_actionable_elements=_has_actionable_elements(observation),
        last_turn_had_screenshot=_last_turn_had_screenshot_for_intent(
            state=state,
            agent_name=agent_name,
            intent_key=intent_key,
        ),
    )


def enforcement_prompt_hint(ctx: EnforcementContext) -> str | None:
    if (
        ctx.has_actionable_elements
        and ctx.consecutive_noop_ui_turns >= cfg.MAX_CONSECUTIVE_NOOP_UI_TURNS
    ):
        lead = "Another agent is waiting on you. " if ctx.needed_by_peer else ""
        if ctx.last_turn_had_screenshot:
            return (
                f"\nENFORCEMENT GUIDANCE: {lead}Your last {ctx.consecutive_noop_ui_turns} concrete UI turns "
                "did not complete the task, and you already took a screenshot on the previous turn. "
                "Do not take another screenshot yet. Use the current page state to choose a different "
                "concrete action now."
            )
        return (
            f"\nENFORCEMENT GUIDANCE: {lead}Your last {ctx.consecutive_noop_ui_turns} concrete UI turns "
            "did not change page state. Do not repeat the same click target or coordinates. "
            "If the page is ambiguous, take one screenshot first, then use it to choose a different "
            "concrete action in the same turn."
        )

    if ctx.has_actionable_elements and ctx.needed_by_peer:
        return (
            "\nENFORCEMENT GUIDANCE: Another agent is waiting on you. "
            "You must make concrete UI progress this turn. If the page is ambiguous, "
            "you may take one screenshot first, but follow it with a different concrete "
            "UI action in the same turn."
        )

    if (
        ctx.has_actionable_elements
        and ctx.consecutive_verification_turns >= cfg.MAX_CONSECUTIVE_VERIFICATION_TURNS
    ):
        return (
            f"\nENFORCEMENT GUIDANCE: You have done {ctx.consecutive_verification_turns} "
            "consecutive verification-only turns. You must perform at least one concrete "
            "UI action this turn (prefer click_index)."
        )

    if (
        ctx.has_actionable_elements
        and ctx.same_low_impact_repeats >= cfg.MAX_CONSECUTIVE_LOW_IMPACT_REPEATS
    ):
        return (
            f"\nENFORCEMENT GUIDANCE: You have repeated the same low-impact action "
            f"{ctx.same_low_impact_repeats} times. Choose a different concrete UI action "
            "(prefer click_index on a non-input interactive element)."
        )

    return None
