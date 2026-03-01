"""
coordination.py
---------------
Pure functions for parsing and writing coordination fields on message text.
No async, no state mutation, no browser interaction.
"""
import json
import re
from dataclasses import dataclass

from village.agent_web_use_orchestration.actions import Action, parse_model_action, validate_action_dict
import village.config as cfg

COORD_KEYS = ("INTENT_KEY", "TASK_ID", "INTENT", "OWNER", "STATUS", "OUTPUT", "NEEDS", "NEXT")


@dataclass
class ParsedResponse:
    """All coordination fields extracted from a single LLM response."""
    message_text: str
    parsed_fields: dict[str, str]
    claimed_intent: str
    claimed_intent_key: str
    claimed_status: str
    claimed_needs: str
    actions: list[Action]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _is_none_like(value: str) -> bool:
    return _normalize_text(value) in {"", "none", "null", "n/a", "na"}


def _truncate_value(value: str, limit: int = 180) -> str:
    value = (value or "").strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


# ---------------------------------------------------------------------------
# Coordination field parsing and mutation
# ---------------------------------------------------------------------------

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


def _normalize_intent_key(value: str) -> str:
    # Keep up to 4 tokens so that multi-word intents stay distinct across agents.
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    return "_".join(tokens[:4])[:120]


def _get_intent_key_from_fields(fields: dict[str, str]) -> str:
    raw_key = fields.get("INTENT_KEY", "")
    if raw_key:
        return _normalize_intent_key(raw_key)
    return _normalize_intent_key(fields.get("INTENT", ""))


def _can_handoff_done_intent(
    *,
    previous_output: str,
    new_output: str,
    previous_agent: str,
    current_agent: str,
) -> bool:
    return (
        current_agent != previous_agent
        and not _is_none_like(new_output)
        and previous_output.strip() != new_output.strip()
    )


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------

def extract_message(response: str) -> str | None:
    message_match = re.search(r"(?im)^\s*MESSAGE\s*:\s*", response)
    if not message_match:
        return None
    action_match = re.search(r"(?im)^\s*ACTION\s*:\s*", response)
    start = message_match.end()
    end = action_match.start() if action_match and action_match.start() > start else len(response)
    payload = response[start:end].strip()
    return payload or None


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