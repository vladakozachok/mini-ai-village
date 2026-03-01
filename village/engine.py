"""
engine.py
---------
Orchestration layer: builds LLM prompts, runs agent cycles, and manages
the worker/group execution loop. Imports everything from the sub-modules.
"""
import asyncio
import json
from typing import Awaitable, Callable

import village.config as cfg
from village.types import AgentState, Message, RunState
from village.agent_web_use_orchestration.browser_env import ExecutionResult
from village.providers.openai_provider import generate_response as openai_generate
from village.providers.deepseek_provider import generate_response as deepseek_generate
from village.providers.gemini_provider import generate_response as gemini_generate

from village.coordination import (
    COORD_KEYS,
    _can_handoff_done_intent,
    _get_intent_key_from_fields,
    _is_none_like,
    _normalize_text,
    _set_coordination_field,
    _truncate_value,
    parse_coordination_fields,
    ParsedResponse,
    extract_actions,
    extract_message,
)
from village.memory_ops import (
    _compact_memory_snapshot,
    _last_agent_fields,
    _last_known_agent_status,
    build_coordination_summary,
    update_shared_memory,
)
from village.scheduling import (
    _consecutive_same_low_impact_plan_for_intent,
    _consecutive_verification_turns_for_intent,
    _deadlock_breaker_leader,
    _get_enforcement_suffix,
    _is_needed_by_peer,
    _resolve_waiting_status,
    _wait_dependency_resolved,
    find_active_intent_owner_by_key,
)
from village.agent_web_use_orchestration.browser_ops import (
    _action_signature,
    _attempt_modal_dismiss,
    _has_actionable_elements,
    _has_meaningful_observation_delta,
    _is_low_impact_actions,
    _resolve_click_index,
)

PROVIDER_GEN = {
    cfg.Provider.OPENAI: openai_generate,
    cfg.Provider.DEEPSEEK: deepseek_generate,
    cfg.Provider.GEMINI: gemini_generate,
}


def _observation_text(observation: dict) -> str:
    if not isinstance(observation, dict):
        return ""
    return _normalize_text(
        " ".join(
            str(observation.get(key, ""))
            for key in ("title", "focused_text", "visible_text")
        )
    )


def _observation_looks_waiting(observation: dict) -> bool:
    text = _observation_text(observation)
    return any(needle in text for needle in ("waiting for", "wait for", "please wait"))


def _single_other_agent_name(state: RunState, agent_name: str) -> str | None:
    others = [a.name for a in state.agents if a.name != agent_name]
    return others[0] if len(others) == 1 else None


def _observation_needs_screenshot(observation: dict) -> bool:
    if not isinstance(observation, dict):
        return False
    adapters = observation.get("adapters", {})
    board_like = adapters.get("board_like", {}) if isinstance(adapters, dict) else {}
    if not (isinstance(board_like, dict) and board_like.get("enabled")):
        return False

    elements = observation.get("elements", [])
    if not isinstance(elements, list):
        return False

    anonymous = 0
    for element in elements:
        if not isinstance(element, dict) or element.get("kind") != "interactive":
            continue
        selector = str(element.get("selector", ""))
        text = _normalize_text(str(element.get("text", "")))
        label = _normalize_text(str(element.get("label", "")))
        if ":nth-of-type(" in selector and not (text or label):
            anonymous += 1

    return anonymous >= 4


def build_input_text(state: RunState, agent: AgentState, observation: dict) -> str:
    summary = build_coordination_summary(state)
    memory_snapshot = _compact_memory_snapshot(state.memory.get_snapshot())
    last_actions, last_results = agent.get_last_action()
    screenshot_hint = "no"
    screenshot_reason = "none"

    if last_actions and last_results:
        last_action = last_actions[-1]
        last_result = last_results[-1]
        last_state_changed = last_result.data.get("state_changed") if isinstance(last_result.data, dict) else None
        if last_action.type.value == "screenshot":
            screenshot_reason = "A screenshot was already taken on the previous turn; act from it before taking another one."
        elif last_action.type.value in {"click", "click_index", "click_relative"} and last_state_changed is False:
            screenshot_hint = "yes"
            screenshot_reason = "The previous click did not change page state."

    if screenshot_hint == "no" and _observation_needs_screenshot(observation):
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
            state_changed = result.data.get("state_changed") if isinstance(result.data, dict) else None
            compact_last.append(
                f"{action.type.value}:success={result.success},state_changed={state_changed}"
                f",error={result.error_message or 'none'}"
            )
        parts.append(f"LAST_ACTION_RESULT: {' | '.join(compact_last)}\n\n")
    parts.append(f"SCREENSHOT_RECOMMENDED: {screenshot_hint} ({screenshot_reason})\n\n")

    parts.append(f"BROWSER OBSERVATION: \n{json.dumps(observation)}\n")

    return "".join(parts)


def _build_coordination_message(
    *,
    intent_key: str,
    task_id: str,
    intent: str,
    owner: str,
    status: str,
    output: str,
    needs: str,
    next_step: str,
) -> str:
    fields = {
        "INTENT_KEY": intent_key or "none",
        "TASK_ID": task_id or "none",
        "INTENT": intent or "none",
        "OWNER": owner or "none",
        "STATUS": status or "in_progress",
        "OUTPUT": output or "none",
        "NEEDS": needs or "none",
        "NEXT": next_step or "none",
    }
    return "\n".join(f"{key}: {fields[key]}" for key in COORD_KEYS)


def _rewrite_coordination_message(
    pr: ParsedResponse,
    *,
    agent_name: str,
    status: str,
    output: str | None = None,
    needs: str | None = None,
    next_step: str | None = None,
) -> str:
    fields = pr.parsed_fields
    return _build_coordination_message(
        intent_key=pr.claimed_intent_key or _get_intent_key_from_fields(fields),
        task_id=fields.get("TASK_ID", "none"),
        intent=pr.claimed_intent or fields.get("INTENT", "none"),
        owner=agent_name,
        status=status,
        output=fields.get("OUTPUT", "none") if output is None else output,
        needs=fields.get("NEEDS", "none") if needs is None else needs,
        next_step=fields.get("NEXT", "none") if next_step is None else next_step,
    )


def _parse_response(response: str, state: RunState, agent: AgentState) -> ParsedResponse:
    """Extract and normalise all coordination fields from a raw LLM response."""
    message_text = extract_message(response) or response
    parsed_fields = parse_coordination_fields(message_text)
    claimed_intent = parsed_fields.get("INTENT", "")
    claimed_intent_key = _get_intent_key_from_fields(parsed_fields)
    if claimed_intent_key:
        message_text = _set_coordination_field(message_text, "INTENT_KEY", claimed_intent_key)
        parsed_fields["INTENT_KEY"] = claimed_intent_key
    claimed_status = _normalize_text(parsed_fields.get("STATUS", ""))
    claimed_needs = parsed_fields.get("NEEDS", "")
    reported_output = parsed_fields.get("OUTPUT", "none")

    actions = extract_actions(response)

    # Auto-promote to done only when the agent is repeating an output it already
    # produced for this intent (i.e. it is spinning without making progress).
    # We do NOT auto-promote on the first output — the agent may legitimately want
    # to stay in_progress and continue working before handing off.
    if (
        claimed_status in {"in_progress", "waiting"}
        and not _is_none_like(reported_output)
        and not actions
        and claimed_intent_key
    ):
        previous_artifact = state.memory.artifacts.get(claimed_intent_key)
        if (
            previous_artifact is not None
            and previous_artifact.value.strip() == reported_output.strip()
        ):
            claimed_status = "done"
            message_text = _set_coordination_field(message_text, "STATUS", "done")
            message_text = _set_coordination_field(
                message_text,
                "NEXT",
                "This artifact completes the current intent. Use a new INTENT_KEY for the next phase.",
            )

    if claimed_status == "done":
        previous_artifact = state.memory.artifacts.get(claimed_intent_key) if claimed_intent_key else None
        # Use last_done_agent from done_intents tuple (not artifact.by_agent) so
        # turn-taking is based on who last *finished* the intent, not who wrote
        # the most recent artifact (these diverge when agents work in parallel).
        _done_entry = state.memory.done_intents.get(claimed_intent_key) if claimed_intent_key else None
        _last_done_agent = (
            _done_entry[1] if isinstance(_done_entry, tuple)
            else (previous_artifact.by_agent if previous_artifact else None)
        )
        allow_done_handoff = (
            previous_artifact is not None
            and _can_handoff_done_intent(
                previous_output=previous_artifact.value,
                new_output=reported_output,
                previous_agent=_last_done_agent,
                current_agent=agent.name,
            )
        )
        if (
            claimed_intent_key in state.memory.done_intents
            and previous_artifact is not None
            and not allow_done_handoff
        ):
            message_text = _set_coordination_field(
                message_text,
                "NEXT",
                "This intent is already marked as done and the artifact is unchanged. "
                "Produce a new artifact or move onto a different task.",
            )
        elif _is_none_like(reported_output):
            claimed_status = "in_progress"
            message_text = _set_coordination_field(message_text, "STATUS", "in_progress")
            message_text = _set_coordination_field(
                message_text,
                "NEXT",
                "Provide a concrete OUTPUT artifact before marking done.",
            )

    if claimed_status == "waiting":
        claimed_status, message_text = _resolve_waiting_status(
            state, agent, claimed_intent_key, claimed_needs, message_text
        )

    final_fields = parse_coordination_fields(message_text)

    return ParsedResponse(
        message_text=message_text,
        parsed_fields=final_fields,
        claimed_intent=final_fields.get("INTENT", claimed_intent),
        claimed_intent_key=_get_intent_key_from_fields(final_fields) or claimed_intent_key,
        claimed_status=_normalize_text(final_fields.get("STATUS", claimed_status)),
        claimed_needs=final_fields.get("NEEDS", claimed_needs),
        actions=actions,
    )


def _default_transition_output(action_results: list[ExecutionResult], observation_after: dict) -> str:
    """Build a compact artifact from the post-action state instead of a placeholder string."""
    for result in reversed(action_results):
        data = result.data if isinstance(result.data, dict) else None
        if not data:
            continue
        post_state = data.get("post_state") if isinstance(data.get("post_state"), dict) else None
        if not post_state:
            continue
        title = _normalize_text(str(post_state.get("title", "")))
        excerpt = _normalize_text(str(post_state.get("body_text_excerpt", "")))
        parts = [part for part in (title, excerpt[:140]) if part]
        if parts:
            return _truncate_value(" | ".join(parts), 180)

    title = _normalize_text(str(observation_after.get("title", "")))
    focused = _normalize_text(str(observation_after.get("focused_text", "")))
    visible = _normalize_text(str(observation_after.get("visible_text", "")))
    parts = [part for part in (title, focused[:100], visible[:140]) if part]
    if parts:
        return _truncate_value(" | ".join(parts), 180)
    return "observed_state_transition"


# ---------------------------------------------------------------------------
# Screenshot extraction
# ---------------------------------------------------------------------------

def _extract_screenshot_b64(action_results: list) -> str | None:
    """Return the base64 image from the most recent successful screenshot action, or None."""
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
                action_results.append(ExecutionResult(success=False, error_message=error or "click_index failed"))
                if cfg.STOP_ON_ACTION_FAILURE:
                    break
                continue

        signature = _action_signature(resolved_action)
        if state.memory.failure_counts.get(signature, 0) >= cfg.MAX_REPEAT_FAILURES_PER_ACTION:
            executed_actions.append(resolved_action)
            action_results.append(ExecutionResult(
                success=False,
                error_message=f"Skipped action due to repeated failures: {signature}",
            ))
            continue

        result = await browser.execute(resolved_action)
        if (
            not result.success
            and result.error_message
            and "intercepts pointer events" in result.error_message
        ):
            if await _attempt_modal_dismiss(browser.page):
                result = await browser.execute(resolved_action)

        executed_actions.append(resolved_action)
        action_results.append(result)
        if cfg.STOP_ON_ACTION_FAILURE and not result.success:
            break

    observation_after = await browser.build_observation() if executed_actions else observation
    return executed_actions, action_results, observation_after


# ---------------------------------------------------------------------------
# Agent cycle
# ---------------------------------------------------------------------------

async def run_agent_cycle(state: RunState, agent: AgentState) -> Message:
    generate_response_fn = PROVIDER_GEN[agent.provider]
    browser = agent.get_browser()

    observation = await browser.build_observation()
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

    pr = _parse_response(response, state, agent)

    enforcement_suffix = _get_enforcement_suffix(
        claimed_status=pr.claimed_status,
        claimed_needs=pr.claimed_needs,
        has_actionable_elements=_has_actionable_elements(observation),
        verification_only_actions=_is_low_impact_actions(pr.actions, strict=True),
        consecutive_verification_turns=_consecutive_verification_turns_for_intent(
            state=state,
            agent_name=agent.name,
            intent_key=pr.claimed_intent_key,
        ),
        needed_by_peer=_is_needed_by_peer(state, agent.name),
        actions=pr.actions,
        same_low_impact_repeats=_consecutive_same_low_impact_plan_for_intent(
            state=state,
            agent_name=agent.name,
            intent_key=pr.claimed_intent_key,
            current_actions=pr.actions,
        ),
    )

    if enforcement_suffix:
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_response_fn,
                    model=agent.model,
                    instructions=agent.system_prompt,
                    input_text=input_text + enforcement_suffix,
                ),
                timeout=cfg.LLM_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"LLM call timed out after {cfg.LLM_TIMEOUT_SECONDS}s for {agent.name}") from exc
        pr = _parse_response(response, state, agent)

    active_owner = find_active_intent_owner_by_key(
        state=state,
        intent_key=pr.claimed_intent_key,
        exclude_agent=agent.name,
    )

    executed_actions: list = []
    action_results: list = []
    should_execute_actions = True

    # Allow re-entry on a done intent only if a *different* agent was the last
    # to mark it done.  We read from done_intents (which stores the agent who
    # last marked done) rather than the artifact's by_agent — those can differ,
    # which caused the chess stall: Sam writes artifact then Kai acts on a
    # different intent, Sam's artifact is still "last writer" so Sam was
    # incorrectly blocked from picking up make_move again.
    if (
        pr.claimed_intent_key
        and pr.claimed_intent_key in state.memory.done_intents
        and pr.claimed_status != "done"
    ):
        previous_artifact = state.memory.artifacts.get(pr.claimed_intent_key)
        _done_entry = state.memory.done_intents[pr.claimed_intent_key]
        last_done_agent = (
            _done_entry[1] if isinstance(_done_entry, tuple)
            else (previous_artifact.by_agent if previous_artifact else None)
        )
        same_agent_last_finished = last_done_agent == agent.name
        if same_agent_last_finished:
            pr.message_text = _set_coordination_field(
                pr.message_text,
                "NEXT",
                "This intent already has an artifact from you. "
                "Use a new INTENT_KEY for your next action.",
            )
    if should_execute_actions and active_owner and pr.claimed_status == "in_progress":
        pr.claimed_status = "waiting"
        pr.claimed_needs = f"{active_owner} to finish current active ownership of this intent."
        pr.actions = []
        should_execute_actions = False
        pr.message_text = _rewrite_coordination_message(
            pr,
            agent_name=agent.name,
            status="waiting",
            output="none",
            needs=pr.claimed_needs,
            next_step="Support verification or choose a complementary subtask.",
        )

    if should_execute_actions:
        new_actions, new_results, observation_after = await _execute_actions(
            browser=browser,
            observation=observation,
            actions=pr.actions,
            state=state,
        )
        executed_actions.extend(new_actions)
        action_results.extend(new_results)

        # Backfill OUTPUT from get_value result if not already set.
        reported_output = pr.parsed_fields.get("OUTPUT", "none")
        if _is_none_like(reported_output):
            for action, result in zip(executed_actions, action_results):
                if action.type.value != "get_value" or not result.success or not result.data:
                    continue
                value = str(result.data.get("value", "")).strip()
                if not value:
                    continue
                pr.message_text = _set_coordination_field(pr.message_text, "OUTPUT", value)
                break

        # Backfill OUTPUT from state transition if still not set.
        if _is_none_like(parse_coordination_fields(pr.message_text).get("OUTPUT", "none")) and executed_actions:
            state_changed_signal = any(
                isinstance(r.data, dict) and r.data.get("state_changed") is True
                for r in action_results if r.success
            )
            if state_changed_signal or _has_meaningful_observation_delta(observation, observation_after):
                pr.message_text = _set_coordination_field(
                    pr.message_text,
                    "OUTPUT",
                    _default_transition_output(action_results, observation_after),
                )

        if (
            executed_actions
            and _observation_looks_waiting(observation_after)
            and _normalize_text(parse_coordination_fields(pr.message_text).get("STATUS", "")) == "in_progress"
        ):
            pr.message_text = _set_coordination_field(pr.message_text, "STATUS", "waiting")
            if _is_none_like(parse_coordination_fields(pr.message_text).get("NEEDS", "none")):
                handoff_agent = _single_other_agent_name(state, agent.name)
                if handoff_agent:
                    pr.message_text = _set_coordination_field(
                        pr.message_text,
                        "NEEDS",
                        f"{handoff_agent} to take the next complementary step in their own browser.",
                    )
            pr.message_text = _set_coordination_field(
                pr.message_text,
                "NEXT",
                "Current page indicates you should wait before acting again.",
            )

    # If the agent took a screenshot, re-call the LLM with the image attached
    # so it can actually see the browser state and derive correct coordinates.
    # We pass image_b64 as a keyword arg; providers that don't support vision
    # simply ignore it via **kwargs or the explicit optional param.
    screenshot_b64 = _extract_screenshot_b64(action_results)
    if screenshot_b64:
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    generate_response_fn,
                    model=agent.model,
                    instructions=agent.system_prompt,
                    input_text=(
                        input_text
                        + "\n\nSCREENSHOT_TAKEN: The screenshot above shows the current browser state. "
                        + "Use it to identify the correct coordinates or elements before acting."
                    ),
                    image_b64=screenshot_b64,
                ),
                timeout=cfg.LLM_TIMEOUT_SECONDS,
            )
            pr = _parse_response(response, state, agent)
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

                if _is_none_like(parse_coordination_fields(pr.message_text).get("OUTPUT", "none")) and new_actions:
                    state_changed_signal = any(
                        isinstance(r.data, dict) and r.data.get("state_changed") is True
                        for r in new_results if r.success
                    )
                    if state_changed_signal or _has_meaningful_observation_delta(latest_observation, observation_after):
                        pr.message_text = _set_coordination_field(
                            pr.message_text,
                            "OUTPUT",
                            _default_transition_output(new_results, observation_after),
                        )

                if (
                    new_actions
                    and _observation_looks_waiting(observation_after)
                    and _normalize_text(parse_coordination_fields(pr.message_text).get("STATUS", "")) == "in_progress"
                ):
                    pr.message_text = _set_coordination_field(pr.message_text, "STATUS", "waiting")
                    if _is_none_like(parse_coordination_fields(pr.message_text).get("NEEDS", "none")):
                        handoff_agent = _single_other_agent_name(state, agent.name)
                        if handoff_agent:
                            pr.message_text = _set_coordination_field(
                                pr.message_text,
                                "NEEDS",
                                f"{handoff_agent} to take the next complementary step in their own browser.",
                            )
                    pr.message_text = _set_coordination_field(
                        pr.message_text,
                        "NEXT",
                        "Current page indicates you should wait before acting again.",
                    )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"LLM call timed out after {cfg.LLM_TIMEOUT_SECONDS}s for {agent.name}") from exc
        except TypeError:
            # Provider does not accept image_b64 yet — skip vision call gracefully.
            pass

    agent.add_action(executed_actions, action_results)

    return Message(
        speaker=agent.name,
        raw_response=response,
        message=pr.message_text,
        actions=executed_actions,
        action_results=action_results,
        input_text=input_text,
        observation=observation,
    )


# ---------------------------------------------------------------------------
# Worker and run group
# ---------------------------------------------------------------------------

async def worker(
    state: RunState,
    agent: AgentState,
    after_turn: Callable[[RunState, AgentState, Message], Awaitable[None]] | None = None,
) -> None:
    consecutive_errors = 0

    while not state.stop_event.is_set():
        last_fields = _last_agent_fields(state, agent.name)
        last_status = _last_known_agent_status(state, agent.name, fields=last_fields)
        last_needs = last_fields.get("NEEDS", "")

        # Waiting: sleep unless dependency is resolved or a peer needs us.
        if last_status == "waiting" and not _is_none_like(last_needs):
            needed_by_peer = _is_needed_by_peer(state, agent.name)
            dep_resolved = _wait_dependency_resolved(
                state=state,
                agent_name=agent.name,
                intent_key=_get_intent_key_from_fields(last_fields),
                needs=last_needs,
            )
            is_deadlock_leader = _deadlock_breaker_leader(state) == agent.name

            should_proceed = needed_by_peer or dep_resolved or is_deadlock_leader
            if not should_proceed:
                await asyncio.sleep(0.5)
                continue

        # Blocked: only proceed for the recoverable idle-action variant.
        if last_status in {"blocked", "blocked_external"}:
            idle_action_blocked = last_status == "blocked" and "idle_action_required" in (last_needs or "")
            if not idle_action_blocked:
                await asyncio.sleep(0.5)
                continue

        try:
            msg = await run_agent_cycle(state, agent)
            consecutive_errors = 0
        except Exception as exc:
            consecutive_errors += 1
            print(f"[ERROR] {agent.name} cycle failed ({consecutive_errors}/{cfg.MAX_CONSECUTIVE_WORKER_ERRORS}): {exc}")
            if consecutive_errors >= cfg.MAX_CONSECUTIVE_WORKER_ERRORS:
                print(f"[ERROR] {agent.name} exceeded consecutive error limit — stopping worker.")
                state.stop_event.set()
                break
            await asyncio.sleep(0.5)
            continue

        end_execution = False
        async with state.message_lock:
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
            except Exception as exc:
                print(f"Error in after_turn callback for agent {agent.name}: {exc}")

        if end_execution:
            break

        if cfg.AGENT_TURN_COOLDOWN_SECONDS > 0:
            await asyncio.sleep(cfg.AGENT_TURN_COOLDOWN_SECONDS)


async def run_group(
    state: RunState,
    after_turn: Callable[[RunState, AgentState, Message], Awaitable[None]] | None = None,
) -> None:
    tasks = [asyncio.create_task(worker(state, agent, after_turn)) for agent in state.agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Worker {state.agents[i].name} crashed: {result}")
