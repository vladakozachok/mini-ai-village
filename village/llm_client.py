from __future__ import annotations

import asyncio
import logging
import random

import village.config as cfg

logger = logging.getLogger(__name__)

_RETRYABLE_SUBSTRINGS = (
    "rate limit",
    "rate_limit",
    "429",
    "503",
    "502",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
)


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in _RETRYABLE_SUBSTRINGS)


def _backoff_delay(attempt: int) -> float:
    base = cfg.LLM_RETRY_BASE_DELAY_S * (2**attempt)
    capped = min(base, cfg.LLM_RETRY_MAX_DELAY_S)
    return capped * random.uniform(1 - cfg.LLM_RETRY_JITTER, 1 + cfg.LLM_RETRY_JITTER)


async def call_llm_with_retry(
    generate_fn,
    *,
    model: str,
    instructions: str,
    input_text: str,
    agent_name: str,
    image_b64: str | None = None,
) -> str:
    kwargs: dict = dict(model=model, instructions=instructions, input_text=input_text)
    if image_b64 is not None:
        kwargs["image_b64"] = image_b64

    last_exc: Exception | None = None
    for attempt in range(cfg.LLM_MAX_RETRIES + 1):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(generate_fn, **kwargs),
                timeout=cfg.LLM_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            last_exc = RuntimeError(
                f"LLM call timed out after {cfg.LLM_TIMEOUT_SECONDS}s "
                f"for {agent_name} (model={model})"
            )
            last_exc.__cause__ = exc
        except TypeError:
            raise
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            last_exc = exc

        if attempt < cfg.LLM_MAX_RETRIES:
            delay = _backoff_delay(attempt)
            logger.warning(
                "LLM call failed for %s (model=%s), attempt %d/%d — retrying in %.1fs: %s",
                agent_name,
                model,
                attempt + 1,
                cfg.LLM_MAX_RETRIES,
                delay,
                last_exc,
            )
            await asyncio.sleep(delay)

    raise last_exc
