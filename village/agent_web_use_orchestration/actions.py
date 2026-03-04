from __future__ import annotations

import enum
import time
from dataclasses import asdict, dataclass, field


class ActionType(str, enum.Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    CLICK_INDEX = "click_index"
    CLICK_RELATIVE = "click_relative"
    SCROLL = "scroll"
    KEYPRESS = "keypress"
    TYPE = "type"
    GET_VALUE = "get_value"
    SCREENSHOT = "screenshot"


@dataclass
class Action:
    type: ActionType
    url: str | None = None
    selector: str | None = None
    x: int | None = None
    y: int | None = None
    index: int | None = None
    rel_x: float | None = None
    rel_y: float | None = None
    direction: str | None = None
    amount: int | None = None
    keys: list[str] | None = None
    text: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


def _is_str(v: object) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_unit_float(v: object) -> bool:
    return isinstance(v, float) and 0.0 <= v <= 1.0


def _is_pixel_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _looks_normalised(x: object, y: object) -> bool:
    return _is_unit_float(x) and _is_unit_float(y)


def validate_action_dict(d: dict) -> bool:
    try:
        action_type = ActionType(d.get("type"))
    except ValueError:
        return False

    if action_type == ActionType.NAVIGATE:
        return _is_str(d.get("url"))

    if action_type == ActionType.CLICK:
        x, y = d.get("x"), d.get("y")
        if _looks_normalised(x, y):
            return True
        return _is_str(d.get("selector")) or (_is_pixel_int(x) and _is_pixel_int(y))

    if action_type == ActionType.CLICK_INDEX:
        v = d.get("index")
        return isinstance(v, int) and not isinstance(v, bool) and v >= 0

    if action_type == ActionType.CLICK_RELATIVE:
        return (
            _is_str(d.get("selector"))
            and _is_unit_float(d.get("rel_x"))
            and _is_unit_float(d.get("rel_y"))
        )

    if action_type == ActionType.SCROLL:
        return d.get("direction") in {"up", "down", "left", "right"} and _is_number(
            d.get("amount")
        )

    if action_type == ActionType.KEYPRESS:
        keys = d.get("keys")
        return isinstance(keys, list) and bool(keys) and all(_is_str(k) for k in keys)

    if action_type == ActionType.TYPE:
        return _is_str(d.get("selector")) and _is_str(d.get("text"))

    if action_type == ActionType.GET_VALUE:
        return _is_str(d.get("selector"))

    if action_type == ActionType.SCREENSHOT:
        return True

    return False


def parse_model_action(d: dict) -> Action:
    action_type = ActionType(d["type"])

    if action_type == ActionType.CLICK:
        x, y, selector = d.get("x"), d.get("y"), d.get("selector")
        if _looks_normalised(x, y):
            return Action(
                type=ActionType.CLICK_RELATIVE,
                selector=selector if _is_str(selector) else None,
                rel_x=float(x),
                rel_y=float(y),
            )

    return Action(
        type=action_type,
        url=d.get("url"),
        selector=d.get("selector"),
        x=int(d["x"]) if _is_pixel_int(d.get("x")) else None,
        y=int(d["y"]) if _is_pixel_int(d.get("y")) else None,
        index=d.get("index"),
        rel_x=float(d["rel_x"]) if _is_unit_float(d.get("rel_x")) else None,
        rel_y=float(d["rel_y"]) if _is_unit_float(d.get("rel_y")) else None,
        direction=d.get("direction"),
        amount=int(d["amount"]) if _is_number(d.get("amount")) else None,
        keys=d.get("keys"),
        text=d.get("text"),
    )
