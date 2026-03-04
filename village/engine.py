"""
Run logic
"""

import asyncio
import json
import logging
from typing import Awaitable, Callable

import village.config as cfg
from village.types import AgentState, Message, RunState
from village.agent_web_use_orchestration.actions import Action, ActionType
from village.agent_web_use_orchestration.browser_env import ExecutionResult
from village.providers.openai_provider import generate_response as openai_generate
from village.providers.deepseek_provider import generate_response as deepseek_generate
from village.providers.gemini_provider import generate_response as gemini_generate
from village.llm_client import call_llm_with_retry
from village.coordination import (
    _get_intent_key_from_fields,
    _is_none_like,
    _normalize_text,
    _set_coordination_field,
    _truncate_value,
    parse_coordination_fields,
    ParsedResponse,
    extract_actions,
    rewrite_coordination_message,
)
from village.memory_ops import (
    _compact_memory_snapshot,
    _last_agent_intent_key,
    _last_agent_needs,
    _last_known_agent_status,
    build_coordination_summary,
    update_shared_memory,
)
from village.scheduling import (
    _deadlock_breaker_leader,
    _is_needed_by_peer,
    _peer_progress_since_done,
    _resolve_waiting_status,
    _wait_dependency_resolved,
    build_enforcement_context,
    enforcement_prompt_hint,
    find_active_intent_owner_by_key,
)
from village.agent_web_use_orchestration.browser_ops import (
    _action_signature,
    _attempt_modal_dismiss,
    _drop_screenshot_actions,
    _has_meaningful_observation_delta,
    _limit_screenshot_actions,
    _observation_looks_waiting,
    _observation_needs_screenshot,
    _resolve_click_relative,
    _resolve_click_index,
)

logger = logging.getLogger(__name__)

PROVIDER_GEN = {
    cfg.Provider.OPENAI: openai_generate,
    cfg.Provider.DEEPSEEK: deepseek_generate,
    cfg.Provider.GEMINI: gemini_generate,
}


def build_input_text(
    state: RunState,
    agent: AgentState,
    observation: dict,
    enforcement_hint: str | None = None,
) -> str:
    summary = build_coordination_summary(state)
    memory_snapshot = _compact_memory_snapshot(state.memory.get_snapshot())
    last_actions, last_results = agent.get_last_action()

    last_status = _last_known_agent_status(state, agent.name)

    screenshot_hint = "no"
    screenshot_reason = "none"
    is_waiting_now = last_status == "waiting" or _observation_looks_waiting(observation)

    if is_waiting_now:
        screenshot_reason = (
            "Current page indicates waiting; do not take a screenshot while waiting."
        )
    elif last_actions and last_results:
        last_action = last_actions[-1]
        last_result = last_results[-1]
        last_state_changed = (
            last_result.data.get("state_changed")
            if isinstance(last_result.data, dict)
            else None
        )
        if any(action.type.value == "screenshot" for action in last_actions):
            screenshot_reason = "A screenshot was already taken on the previous turn; act from it before taking another one."
        elif (
            last_action.type.value in {"click", "click_index", "click_relative"}
            and last_state_changed is False
        ):
            screenshot_hint = "yes"
            screenshot_reason = "The previous click did not change page state."
        elif "repeated failures" in str(last_result.error_message or "").lower():
            screenshot_hint = "yes"
            screenshot_reason = "The previous target was skipped after repeated failures; reassess the page before trying a different action."

    if (
        screenshot_hint == "no"
        and not is_waiting_now
        and _observation_needs_screenshot(observation)
    ):
        screenshot_hint = "yes"
        screenshot_reason = "The current page appears visually ambiguous and click targets are weakly identified."

    parts = [
        f"YOUR OVERARCHING GOAL: {state.goal}\n",
        f"LATEST MESSAGES SUMMARY: {summary}\n\n",
        f"MEMORY_SNAPSHOT: {json.dumps(memory_snapshot)}\n\n",
    ]
    if last_actions:
        compact_last = []
        for action, result in zip(last_actions[-2:], last_results[-2:]):
            state_changed = (
                result.data.get("state_changed")
                if isinstance(result.data, dict)
                else None
            )
            compact_last.append(
                f"{action.type.value}:success={result.success},state_changed={state_changed}"
                f",error={result.error_message or 'none'}"
            )
        parts.append(f"LAST_ACTION_RESULT: {' | '.join(compact_last)}\n\n")
    parts.append(f"SCREENSHOT_RECOMMENDED: {screenshot_hint} ({screenshot_reason})\n\n")
    if enforcement_hint:
        parts.append(f"{enforcement_hint}\n\n")
    parts.append(f"BROWSER OBSERVATION: \n{json.dumps(observation)}\n")

    return "".join(parts)


def _parse_response(
    response: str, state: RunState, agent: AgentState
) -> ParsedResponse:
    parsed_fields = parse_coordination_fields(response)
    message_text = response

    claimed_intent = parsed_fields.get("intent", "")
    claimed_intent_key = _get_intent_key_from_fields(parsed_fields)
    claimed_status = _normalize_text(parsed_fields.get("status", ""))
    claimed_needs = parsed_fields.get("needs", "")
    reported_output = parsed_fields.get("output", "none")

    if claimed_intent_key:
        parsed_fields["intent_key"] = claimed_intent_key
        message_text = _set_coordination_field(
            message_text, "intent_key", claimed_intent_key
        )

    if claimed_status == "done" and _is_none_like(reported_output):
        claimed_status = "in_progress"
        parsed_fields["status"] = "in_progress"
        parsed_fields["next"] = (
            "Provide a concrete output artifact or completion evidence before marking done."
        )
        message_text = _set_coordination_field(message_text, "status", "in_progress")
        message_text = _set_coordination_field(
            message_text, "next", parsed_fields["next"]
        )

    if (
        claimed_status != "done"
        and claimed_intent_key
        and state.memory.is_done(claimed_intent_key)
        and not state.memory.can_reenter(claimed_intent_key, agent.name)
    ):
        parsed_fields["next"] = (
            "You finished this intent. Use a new intent_key for your next action."
        )
        message_text = _set_coordination_field(
            message_text, "next", parsed_fields["next"]
        )

    if claimed_status == "waiting":
        claimed_status, message_text = _resolve_waiting_status(
            state, agent, claimed_intent_key, claimed_needs, message_text
        )
        parsed_fields = parse_coordination_fields(message_text)

    return ParsedResponse(
        message_text=message_text,
        parsed_fields=parsed_fields,
        claimed_intent=parsed_fields.get("intent", claimed_intent),
        claimed_intent_key=_get_intent_key_from_fields(parsed_fields)
        or claimed_intent_key,
        claimed_status=_normalize_text(parsed_fields.get("status", claimed_status)),
        claimed_needs=parsed_fields.get("needs", claimed_needs),
        actions=extract_actions(response),
    )


def _default_transition_output(
    action_results: list[ExecutionResult], observation_after: dict
) -> str:
    for result in reversed(action_results):
        data = result.data if isinstance(result.data, dict) else None
        if not data:
            continue
        post_state = (
            data.get("post_state") if isinstance(data.get("post_state"), dict) else None
        )
        if not post_state:
            continue
        title = _normalize_text(str(post_state.get("title", "")))
        excerpt = _normalize_text(str(post_state.get("body_text_excerpt", "")))
        parts = [p for p in (title, excerpt[:140]) if p]
        if parts:
            return _truncate_value(" | ".join(parts), 180)

    title = _normalize_text(str(observation_after.get("title", "")))
    focused = _normalize_text(str(observation_after.get("focused_text", "")))
    visible = _normalize_text(str(observation_after.get("visible_text", "")))
    parts = [p for p in (title, focused[:100], visible[:140]) if p]
    if parts:
        return _truncate_value(" | ".join(parts), 180)
    return "observed_state_transition"


def _extract_screenshot_b64(action_results: list) -> str | None:
    for result in reversed(action_results):
        if not result.success or not isinstance(result.data, dict):
            continue
        b64 = result.data.get("screenshot_base64")
        if b64:
            return b64
    return None


async def _execute_actions(
    *,
    browser,
    observation: dict,
    actions: list,
    state: RunState,
) -> tuple[list, list, dict]:
    executed_actions: list = []
    action_results: list = []

    for action in actions:
        resolved_action = action
        if action.type.value == "click_index":
            resolved_action, error = _resolve_click_index(action, observation)
            if not resolved_action:
                executed_actions.append(action)
                action_results.append(
                    ExecutionResult(
                        success=False,
                        error_message=error or "click_index failed",
                    )
                )
                if cfg.STOP_ON_ACTION_FAILURE:
                    break
                continue
        elif action.type.value == "click_relative":
            resolved_action, error = _resolve_click_relative(action, observation)
            if not resolved_action:
                executed_actions.append(action)
                action_results.append(
                    ExecutionResult(
                        success=False,
                        error_message=error or "click_relative failed",
                    )
                )
                if cfg.STOP_ON_ACTION_FAILURE:
                    break
                continue

        signature = _action_signature(resolved_action)
        if (
            state.memory.failure_counts.get(signature, 0)
            >= cfg.MAX_REPEAT_FAILURES_PER_ACTION
        ):
            executed_actions.append(resolved_action)
            action_results.append(
                ExecutionResult(
                    success=False,
                    error_message=f"Skipped action due to repeated failures: {signature}",
                )
            )
            continue

        result = await browser.execute(resolved_action)
        if (
            not result.success
            and result.error_message
            and "intercepts pointer events" in result.error_message
        ):
            if await _attempt_modal_dismiss(browser.page):
                result = await browser.execute(resolved_action)

        if (
            action.type.value == "click_index"
            and resolved_action.selector
            and not result.success
            and result.error_message
            and "intercepts pointer events" in result.error_message
        ):
            fallback_action, fallback_error = _resolve_click_index(
                action,
                observation,
                prefer_coordinates=True,
            )
            if fallback_action:
                fallback_sig = _action_signature(fallback_action)
                if (
                    fallback_sig != signature
                    and state.memory.failure_counts.get(fallback_sig, 0)
                    < cfg.MAX_REPEAT_FAILURES_PER_ACTION
                ):
                    fallback_result = await browser.execute(fallback_action)
                    fallback_result.data = dict(fallback_result.data or {})
                    fallback_result.data["fallback_from_signature"] = signature
                    fallback_result.data["fallback_reason"] = "pointer_intercepted"
                    resolved_action = fallback_action
                    result = fallback_result
            elif fallback_error and not result.error_message:
                result = ExecutionResult(success=False, error_message=fallback_error)

        executed_actions.append(resolved_action)
        action_results.append(result)
        if cfg.STOP_ON_ACTION_FAILURE and not result.success:
            break

    observation_after = (
        await browser.build_observation() if executed_actions else observation
    )
    return executed_actions, action_results, observation_after


def _apply_waiting_patch(
    pr: ParsedResponse,
    observation_after: dict,
    executed_actions: list,
    action_results: list[ExecutionResult],
    state: RunState,
    agent_name: str,
) -> ParsedResponse:
    if not executed_actions:
        return pr
    if not _observation_looks_waiting(observation_after):
        return pr
    current_status = _normalize_text(
        parse_coordination_fields(pr.message_text).get("status", "")
    )
    if current_status != "in_progress":
        return pr

    verified_progress = any(
        result.success
        and isinstance(result.data, dict)
        and result.data.get("state_changed") is True
        for result in action_results
    )

    if verified_progress:
        fields = parse_coordination_fields(pr.message_text)
        if _is_none_like(fields.get("output", "none")):
            pr.message_text = _set_coordination_field(
                pr.message_text,
                "output",
                _default_transition_output(action_results, observation_after),
            )
        pr.message_text = _set_coordination_field(pr.message_text, "status", "done")
        if _is_none_like(
            parse_coordination_fields(pr.message_text).get("needs", "none")
        ):
            handoff_agent = _single_other_agent_name(state, agent_name)
            if handoff_agent:
                pr.message_text = _set_coordination_field(
                    pr.message_text,
                    "needs",
                    f"{handoff_agent} to take the next complementary step in their own browser.",
                )
        pr.message_text = _set_coordination_field(
            pr.message_text,
            "next",
            "This step completed and the page is now waiting. Start a new intent when the next concrete step is available.",
        )
        pr.parsed_fields = parse_coordination_fields(pr.message_text)
        pr.claimed_status = "done"
        pr.claimed_needs = pr.parsed_fields.get("needs", pr.claimed_needs)
        return pr

    pr.message_text = _set_coordination_field(pr.message_text, "status", "waiting")
    if _is_none_like(parse_coordination_fields(pr.message_text).get("needs", "none")):
        handoff_agent = _single_other_agent_name(state, agent_name)
        if handoff_agent:
            pr.message_text = _set_coordination_field(
                pr.message_text,
                "needs",
                f"{handoff_agent} to take the next complementary step in their own browser.",
            )
    pr.message_text = _set_coordination_field(
        pr.message_text,
        "next",
        "Current page indicates you should wait before acting again.",
    )
    pr.parsed_fields = parse_coordination_fields(pr.message_text)
    pr.claimed_status = "waiting"
    pr.claimed_needs = pr.parsed_fields.get("needs", pr.claimed_needs)
    return pr


def _single_other_agent_name(state: RunState, agent_name: str) -> str | None:
    others = [a.name for a in state.agents if a.name != agent_name]
    return others[0] if len(others) == 1 else None


def _backfill_output(
    pr: ParsedResponse,
    executed_actions: list,
    action_results: list,
    observation: dict,
    observation_after: dict,
) -> ParsedResponse:
    if not _is_none_like(pr.parsed_fields.get("output", "none")):
        return pr

    for action, result in zip(executed_actions, action_results):
        if action.type.value != "get_value" or not result.success or not result.data:
            continue
        value = str(result.data.get("value", "")).strip()
        if value:
            pr.message_text = _set_coordination_field(pr.message_text, "output", value)
            pr.parsed_fields["output"] = value
            return pr

    if executed_actions:
        state_changed = any(
            isinstance(r.data, dict) and r.data.get("state_changed") is True
            for r in action_results
            if r.success
        )
        if state_changed or _has_meaningful_observation_delta(
            observation, observation_after
        ):
            val = _default_transition_output(action_results, observation_after)
            pr.message_text = _set_coordination_field(pr.message_text, "output", val)
            pr.parsed_fields["output"] = val

    return pr


async def run_agent_cycle(state: RunState, agent: AgentState) -> Message:
    generate_fn = PROVIDER_GEN[agent.provider]
    browser = agent.get_browser()

    observation = await browser.build_observation()

    last_intent_key = _last_agent_intent_key(state, agent.name)
    enforcement_ctx = build_enforcement_context(
        state=state,
        agent_name=agent.name,
        intent_key=last_intent_key,
        observation=observation,
    )
    enforcement_hint = enforcement_prompt_hint(enforcement_ctx)
    input_text = build_input_text(
        state, agent, observation, enforcement_hint=enforcement_hint
    )

    raw_response = await call_llm_with_retry(
        generate_fn,
        model=agent.model,
        instructions=agent.system_prompt,
        input_text=input_text,
        agent_name=agent.name,
    )
    pr = _parse_response(raw_response, state, agent)

    if pr.claimed_intent_key and pr.claimed_status == "in_progress":
        state.memory.pending_claims[pr.claimed_intent_key] = agent.name

    active_owner = find_active_intent_owner_by_key(
        state=state,
        intent_key=pr.claimed_intent_key,
        exclude_agent=agent.name,
    )

    executed_actions: list = []
    action_results: list = []
    should_execute_actions = True

    if active_owner and pr.claimed_status == "in_progress":
        state.memory.pending_claims.pop(pr.claimed_intent_key, None)
        pr.claimed_status = "waiting"
        pr.claimed_needs = (
            f"{active_owner} to finish current active ownership of this intent."
        )
        pr.actions = []
        should_execute_actions = False
        pr.message_text = rewrite_coordination_message(
            pr,
            agent_name=agent.name,
            status="waiting",
            output="none",
            needs=pr.claimed_needs,
            next_step="Support verification or choose a complementary subtask.",
        )

    if pr.claimed_status == "waiting":
        pr.actions = _drop_screenshot_actions(pr.actions)
    else:
        pr.actions = _limit_screenshot_actions(pr.actions, 1)
        if enforcement_ctx.last_turn_had_screenshot:
            non_screenshot_actions = _drop_screenshot_actions(pr.actions)
            if non_screenshot_actions:
                pr.actions = non_screenshot_actions
        if (
            enforcement_ctx.consecutive_noop_ui_turns
            >= cfg.MAX_CONSECUTIVE_NOOP_UI_TURNS
            and pr.actions
            and pr.actions[0].type != ActionType.SCREENSHOT
            and not enforcement_ctx.last_turn_had_screenshot
        ):
            pr.actions = [Action(type=ActionType.SCREENSHOT)]

    if should_execute_actions:
        new_actions, new_results, observation_after = await _execute_actions(
            browser=browser,
            observation=observation,
            actions=pr.actions,
            state=state,
        )
        executed_actions.extend(new_actions)
        action_results.extend(new_results)
        pr = _backfill_output(
            pr, executed_actions, action_results, observation, observation_after
        )
        pr = _apply_waiting_patch(
            pr,
            observation_after,
            executed_actions,
            action_results,
            state,
            agent.name,
        )

    screenshot_b64 = _extract_screenshot_b64(action_results)
    current_status = _normalize_text(
        parse_coordination_fields(pr.message_text).get("status", "")
    )
    vision_response: str | None = None

    if screenshot_b64 and current_status != "waiting":
        try:
            vision_response = await call_llm_with_retry(
                generate_fn,
                model=agent.model,
                instructions=agent.system_prompt,
                input_text=(
                    input_text
                    + "\n\nSCREENSHOT_TAKEN: The screenshot above shows the current browser state. "
                    "Use it to identify the correct coordinates or elements before acting."
                ),
                image_b64=screenshot_b64,
                agent_name=agent.name,
            )
            pr = _parse_response(vision_response, state, agent)
            pr.actions = _drop_screenshot_actions(pr.actions)

            if pr.actions:
                latest_observation = await browser.build_observation()
                new_actions, new_results, observation_after = await _execute_actions(
                    browser=browser,
                    observation=latest_observation,
                    actions=pr.actions,
                    state=state,
                )
                executed_actions.extend(new_actions)
                action_results.extend(new_results)
                pr = _backfill_output(
                    pr, new_actions, new_results, latest_observation, observation_after
                )
                pr = _apply_waiting_patch(
                    pr,
                    observation_after,
                    new_actions,
                    new_results,
                    state,
                    agent.name,
                )

        except TypeError:
            vision_response = None

    agent.add_action(executed_actions, action_results)

    return Message(
        speaker=agent.name,
        raw_response=raw_response,
        message=pr.message_text,
        engine_response=vision_response,
        actions=executed_actions,
        action_results=action_results,
        input_text=input_text,
        observation=observation,
    )


async def _notify_state_change(state: RunState) -> None:
    async with state.state_change:
        state.state_change_seq += 1
        state.state_change.notify_all()


async def _wait_for_state_change(state: RunState, observed_seq: int) -> None:
    async with state.state_change:
        await state.state_change.wait_for(
            lambda: state.stop_event.is_set() or state.state_change_seq != observed_seq
        )


async def worker(
    state: RunState,
    agent: AgentState,
    after_turn: (
        Callable[[RunState, AgentState, Message], Awaitable[None]] | None
    ) = None,
) -> None:
    consecutive_errors = 0

    while not state.stop_event.is_set():
        observed_seq = state.state_change_seq
        last_status = _last_known_agent_status(state, agent.name)
        last_needs = _last_agent_needs(state, agent.name)
        last_intent_key = _last_agent_intent_key(state, agent.name)

        if last_status == "waiting" and not _is_none_like(last_needs):
            needed_by_peer = _is_needed_by_peer(state, agent.name)
            dep_resolved = _wait_dependency_resolved(
                state=state,
                agent_name=agent.name,
                intent_key=last_intent_key,
                needs=last_needs,
            )
            is_deadlock_leader = _deadlock_breaker_leader(state) == agent.name

            if not (needed_by_peer or dep_resolved or is_deadlock_leader):
                await _wait_for_state_change(state, observed_seq)
                continue

        if last_status in {"blocked", "blocked_external"}:
            idle_action_blocked = (
                last_status == "blocked"
                and "idle_action_required" in (last_needs or "")
            )
            if not idle_action_blocked:
                await _wait_for_state_change(state, observed_seq)
                continue

        if (
            last_status == "done"
            and not _is_needed_by_peer(state, agent.name)
            and not _peer_progress_since_done(state, agent.name)
        ):
            await _wait_for_state_change(state, observed_seq)
            continue

        try:
            msg = await run_agent_cycle(state, agent)
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            logger.error(
                "Agent '%s' (model=%s) cycle failed (%d/%d): %s",
                agent.name,
                agent.model,
                consecutive_errors,
                cfg.MAX_CONSECUTIVE_WORKER_ERRORS,
                exc,
                exc_info=True,
            )
            if consecutive_errors >= cfg.MAX_CONSECUTIVE_WORKER_ERRORS:
                logger.error(
                    "Agent '%s' (model=%s) exceeded consecutive error limit — stopping run.",
                    agent.name,
                    agent.model,
                )
                state.stop_event.set()
                await _notify_state_change(state)
                break
            continue

        committed_key = _get_intent_key_from_fields(
            parse_coordination_fields(msg.message)
        )

        end_execution = False
        async with state.message_lock:
            state.next_turn += 1
            msg.turn = state.next_turn
            msg.observation = None

            update_shared_memory(state, msg)
            state.messages.append(msg)
            if len(state.messages) > cfg.MAX_MESSAGES_RETAINED:
                state.messages = state.messages[-cfg.MAX_MESSAGES_RETAINED :]

            if committed_key:
                state.memory.pending_claims.pop(committed_key, None)

            logger.info(
                "[TURN %d] %s (model=%s) status=%s intent=%s",
                state.next_turn,
                agent.name,
                agent.model,
                parse_coordination_fields(msg.message).get("status", "?"),
                committed_key or "?",
            )

            turn_limit_hit = state.next_turn >= cfg.MAX_TURNS_PER_ROUND

            stall_limit_hit = (
                cfg.MAX_STALLED_TURNS > 0
                and state.last_progress_turn == 0
                and state.next_turn > cfg.MAX_STALLED_TURNS
            ) or (
                cfg.MAX_STALLED_TURNS > 0
                and state.last_progress_turn > 0
                and state.next_turn - state.last_progress_turn > cfg.MAX_STALLED_TURNS
            )

            if turn_limit_hit or stall_limit_hit:
                if stall_limit_hit and not turn_limit_hit:
                    logger.warning(
                        "Stall circuit breaker triggered after %d turns without "
                        "any intent reaching 'done' (last_progress_turn=%d).",
                        cfg.MAX_STALLED_TURNS,
                        state.last_progress_turn,
                    )
                state.stop_event.set()
                end_execution = True

        await _notify_state_change(state)

        if after_turn:
            try:
                await after_turn(state, agent, msg)
            except Exception as exc:
                logger.warning(
                    "after_turn callback error for agent '%s' (model=%s): %s",
                    agent.name,
                    agent.model,
                    exc,
                    exc_info=True,
                )

        if end_execution:
            break

        if cfg.AGENT_TURN_COOLDOWN_SECONDS > 0:
            await asyncio.sleep(cfg.AGENT_TURN_COOLDOWN_SECONDS)


async def run_group(
    state: RunState,
    after_turn: (
        Callable[[RunState, AgentState, Message], Awaitable[None]] | None
    ) = None,
) -> None:
    tasks = [
        asyncio.create_task(worker(state, agent, after_turn)) for agent in state.agents
    ]

    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                agent = state.agents[i]
                logger.error(
                    "Worker for agent '%s' (model=%s) crashed: %s",
                    agent.name,
                    agent.model,
                    result,
                    exc_info=result,
                )
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("run_group interrupted — signalling workers to stop.")
        state.stop_event.set()
        await _notify_state_change(state)
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        logger.info("Closing browsers for %d agent(s).", len(state.agents))
        close_results = await asyncio.gather(
            *[agent.close_browser() for agent in state.agents],
            return_exceptions=True,
        )
        for agent, result in zip(state.agents, close_results):
            if isinstance(result, Exception):
                logger.warning(
                    "Failed to close browser for agent '%s' (model=%s): %s",
                    agent.name,
                    agent.model,
                    result,
                )
        logger.info("run_group finished. total_turns=%d", state.next_turn)
