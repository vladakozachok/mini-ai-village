"""Helpers for turning model actions into concrete browser operations."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from village.agent_web_use_orchestration.actions import Action, ActionType

if TYPE_CHECKING:
    from playwright.async_api import Page


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


_LOW_IMPACT_TYPES = frozenset({"get_value", "type", "keypress"})
_VERIFICATION_ONLY = frozenset({"get_value"})


def _is_low_impact_actions(actions: list[Action], *, strict: bool = False) -> bool:
    if not actions:
        return True
    bucket = _VERIFICATION_ONLY if strict else _LOW_IMPACT_TYPES
    return all(a.type.value in bucket for a in actions)


def _action_signature(action: Action) -> str:
    t = action.type.value
    if t == "navigate":
        return f"navigate|{action.url or ''}"
    if t == "click":
        return (
            f"click|{action.selector}"
            if action.selector
            else f"click|{action.x},{action.y}"
        )
    if t == "click_index":
        return f"click_index|{action.index}"
    if t == "click_relative":
        return f"click_relative|{action.selector}|{action.rel_x},{action.rel_y}"
    if t == "scroll":
        return f"scroll|{action.direction}|{action.amount}"
    if t == "keypress":
        return f"keypress|{','.join(action.keys or [])}"
    if t == "type":
        return f"type|{action.selector}|len={len(action.text or '')}"
    return f"{t}|"


def _click_from_bbox(bbox: dict, observation: dict) -> Action | None:
    cx, cy = bbox.get("center_x"), bbox.get("center_y")
    if cx is None or cy is None:
        return None
    vw = observation.get("viewport_width", 1280)
    vh = observation.get("viewport_height", 720)
    return Action(
        type=ActionType.CLICK,
        x=int(round(float(cx) * vw)),
        y=int(round(float(cy) * vh)),
    )


def _relative_click(selector: str, rel_x: float, rel_y: float) -> Action:
    return Action(
        type=ActionType.CLICK_RELATIVE,
        selector=selector,
        rel_x=round(float(rel_x), 4),
        rel_y=round(float(rel_y), 4),
    )


def _default_surface_selector(observation: dict) -> str | None:
    adapters = observation.get("adapters", {})
    if isinstance(adapters, dict):
        game_like = adapters.get("game_like", {})
        if isinstance(game_like, dict):
            selector = str(game_like.get("surface_selector", "")).strip()
            if selector:
                return selector

    for surface in observation.get("target_surfaces") or []:
        if not isinstance(surface, dict):
            continue
        selector = str(surface.get("selector", "")).strip()
        tag = str(surface.get("tag", "")).lower()
        if selector and tag not in {"body", "html"}:
            return selector
    return None


def _resolve_click_relative(
    action: Action,
    observation: dict,
) -> tuple[Action | None, str | None]:
    if action.type != ActionType.CLICK_RELATIVE:
        return action, None
    if action.selector:
        return action, None
    if action.rel_x is None or action.rel_y is None:
        return None, "click_relative requires rel_x and rel_y."

    selector = _default_surface_selector(observation)
    if not selector:
        return None, "Relative click needs a target surface, but none was detected."
    return _relative_click(selector, action.rel_x, action.rel_y), None


def _resolve_click_index(
    action: Action,
    observation: dict,
    *,
    prefer_coordinates: bool = False,
) -> tuple[Action | None, str | None]:
    if action.type != ActionType.CLICK_INDEX:
        return action, None

    if not isinstance(action.index, int):
        return None, "click_index requires an integer index."

    elements = observation.get("elements") or []
    if not (0 <= action.index < len(elements)):
        return None, f"click_index {action.index} out of range (0–{len(elements) - 1})."

    el = elements[action.index]
    if not isinstance(el, dict):
        return None, f"Element at index {action.index} is malformed."

    bbox = el.get("normalized_bbox") or {}
    bbox_action = _click_from_bbox(bbox, observation)
    if prefer_coordinates and bbox_action:
        return bbox_action, None

    click_selector = el.get("click_selector")
    click_rel_x = el.get("click_rel_x")
    click_rel_y = el.get("click_rel_y")
    if (
        isinstance(click_selector, str)
        and click_selector.strip()
        and isinstance(click_rel_x, (int, float))
        and isinstance(click_rel_y, (int, float))
    ):
        return (
            _relative_click(click_selector, float(click_rel_x), float(click_rel_y)),
            None,
        )

    selector = el.get("selector")
    if selector:
        return Action(type=ActionType.CLICK, selector=selector), None

    if bbox_action:
        return bbox_action, None

    return None, f"Element at index {action.index} has no selector or bounding box."


def _observation_fingerprint(obs: dict) -> tuple[str, str, str]:
    if not isinstance(obs, dict):
        return "", "", ""
    return (
        _normalize_text(obs.get("title", ""))[:200],
        _normalize_text(obs.get("focused_text", ""))[:240],
        _normalize_text(obs.get("visible_text", ""))[:240],
    )


def _has_meaningful_observation_delta(before: dict, after: dict) -> bool:
    return _observation_fingerprint(before) != _observation_fingerprint(after)


def _has_actionable_elements(obs: dict) -> bool:
    for el in obs.get("elements") or []:
        if (
            isinstance(el, dict)
            and el.get("kind") == "interactive"
            and not el.get("disabled")
            and el.get("selector")
        ):
            return True
    return False


async def _attempt_modal_dismiss(page: "Page") -> bool:
    try:
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(300)
        return True
    except Exception:
        return False


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
    anonymous = sum(
        1
        for element in elements
        if isinstance(element, dict)
        and element.get("kind") == "interactive"
        and ":nth-of-type(" in str(element.get("selector", ""))
        and not (
            _normalize_text(str(element.get("text", "")))
            or _normalize_text(str(element.get("label", "")))
        )
    )
    return anonymous >= 4


def _drop_screenshot_actions(actions: list) -> list:
    return [a for a in actions if a.type.value != "screenshot"]


def _limit_screenshot_actions(actions: list, max_screenshots: int = 1) -> list:
    kept: list = []
    count = 0
    for action in actions:
        if action.type.value == "screenshot":
            if count >= max_screenshots:
                continue
            count += 1
        kept.append(action)
    return kept
