import json
import logging
import re
from dataclasses import dataclass

from village.agent_web_use_orchestration.actions import (
    Action,
    parse_model_action,
    validate_action_dict,
)
import village.config as cfg

logger = logging.getLogger(__name__)


@dataclass
class ParsedResponse:
    message_text: str
    parsed_fields: dict
    claimed_intent: str
    claimed_intent_key: str
    claimed_status: str
    claimed_needs: str
    actions: list[Action]


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _is_none_like(value: str) -> bool:
    return _normalize_text(str(value)) in {"", "none", "null", "n/a", "na"}


def _truncate_value(value: str, limit: int = 180) -> str:
    value = (value or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def parse_response_json(response: str) -> dict:
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = text.split("```", 1)[0].strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


def parse_coordination_fields(message: str) -> dict[str, str]:
    return parse_response_json(message)


def _set_coordination_field(message: str, key: str, value: str) -> str:
    try:
        data = json.loads(message)
        if isinstance(data, dict):
            data[key.lower()] = value
            return json.dumps(data)
    except (json.JSONDecodeError, ValueError):
        pass
    return message


def _normalize_intent_key(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    return "_".join(tokens[:2])[:120]


def _get_intent_key_from_fields(fields: dict) -> str:
    raw_key = fields.get("intent_key", "") or fields.get("INTENT_KEY", "")
    if raw_key:
        return _normalize_intent_key(raw_key)
    intent = fields.get("intent", "") or fields.get("INTENT", "")
    return _normalize_intent_key(intent)


def extract_actions(response: str) -> list[Action]:
    data = parse_response_json(response)
    if not data:
        return []

    action_obj = data.get("action") or data.get("ACTION")
    if action_obj is None:
        return []

    if isinstance(action_obj, dict):
        if validate_action_dict(action_obj):
            return [parse_model_action(action_obj)]
        logger.warning("Invalid single action dict in LLM response; dropping.")
        return []

    if isinstance(action_obj, list):
        parsed: list[Action] = []
        for item in action_obj:
            if not isinstance(item, dict) or not validate_action_dict(item):
                logger.warning("Skipping invalid action item: %s", item)
                continue
            parsed.append(parse_model_action(item))
        return parsed[: cfg.MAX_ACTIONS_PER_TURN]

    return []


def build_coordination_message(
    *,
    intent_key: str,
    intent: str,
    owner: str,
    status: str,
    output: str,
    needs: str,
    next_step: str,
) -> str:
    return json.dumps(
        {
            "intent_key": intent_key or "none",
            "intent": intent or "none",
            "owner": owner or "none",
            "status": status or "in_progress",
            "output": output or "none",
            "needs": needs or "none",
            "next": next_step or "none",
            "action": None,
        }
    )


def rewrite_coordination_message(
    pr: "ParsedResponse",
    *,
    agent_name: str,
    status: str,
    output: str | None = None,
    needs: str | None = None,
    next_step: str | None = None,
) -> str:
    fields = pr.parsed_fields
    return build_coordination_message(
        intent_key=pr.claimed_intent_key or _get_intent_key_from_fields(fields),
        intent=pr.claimed_intent or fields.get("intent", "none"),
        owner=agent_name,
        status=status,
        output=fields.get("output", "none") if output is None else output,
        needs=fields.get("needs", "none") if needs is None else needs,
        next_step=fields.get("next", "none") if next_step is None else next_step,
    )
