import asyncio
import logging
import time
import enum
from dataclasses import dataclass, field, InitVar
from typing import Any

import village.config as cfg
from village.agent_web_use_orchestration.browser_env import (
    BrowserEnvironment,
    ExecutionResult,
)
from village.agent_web_use_orchestration.actions import Action

logger = logging.getLogger(__name__)


class BlockerType(str, enum.Enum):
    captcha = "captcha"
    permission = "permission"
    rate_limit = "rate_limit"


@dataclass(frozen=True)
class AgentConfig:
    name: str
    provider: cfg.Provider
    model: str
    system_prompt: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent name cannot be empty.")
        if not self.model.strip():
            raise ValueError("Agent model cannot be empty.")
        if not self.system_prompt.strip():
            raise ValueError("Agent system_prompt cannot be empty.")


@dataclass
class AgentState:
    config: InitVar[AgentConfig]

    _config: AgentConfig = field(init=False, repr=False)
    browser: BrowserEnvironment = field(init=False)
    actions: list[tuple[list[Action], list[ExecutionResult]]] = field(
        default_factory=list
    )

    def __post_init__(self, config: AgentConfig) -> None:
        self._config = config
        self.browser = BrowserEnvironment(headless=True)

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def provider(self) -> cfg.Provider:
        return self._config.provider

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def system_prompt(self) -> str:
        return self._config.system_prompt

    async def launch_browser(self) -> None:
        await self.browser.launch()

    def get_browser(self) -> BrowserEnvironment:
        return self.browser

    async def close_browser(self) -> None:
        if self.browser:
            await self.browser.close()

    def add_action(self, actions: list[Action], results: list[ExecutionResult]) -> None:
        self.actions.append((actions, results))

    def get_last_action(self) -> tuple[list[Action], list[ExecutionResult]]:
        return self.actions[-1] if self.actions else ([], [])


@dataclass
class Message:
    speaker: str
    raw_response: str
    message: str
    engine_response: str | None = None
    turn: int | None = None
    actions: list[Action] = field(default_factory=list)
    action_results: list[ExecutionResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    input_text: str | None = None
    observation: dict[str, Any] | None = None


@dataclass
class IntentRecord:
    intent_key: str
    intent: str
    owner: str
    status: str
    output: str | None = None
    needs: str | None = None
    turn_claimed: int = 0
    turn_completed: int | None = None
    expires_at_turn: int = 0


@dataclass
class Dependency:
    from_agent: str
    to_agent: str
    intent: str
    since_turn: int
    reason: str


@dataclass
class Blocker:
    intent: str
    type: BlockerType
    since_turn: int
    description: str
    by_agent: str


@dataclass
class MemoryEvent:
    agent: str
    event_type: str
    turn: int
    intent: str
    details: dict[str, Any]


@dataclass
class SharedMemory:
    intents: dict[str, IntentRecord] = field(default_factory=dict)
    pending_claims: dict[str, str] = field(default_factory=dict)
    last_intent_key_by_agent: dict[str, str] = field(default_factory=dict)
    events: list[MemoryEvent] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    blockers: dict[str, Blocker] = field(default_factory=dict)
    failure_counts: dict[str, int] = field(default_factory=dict)
    last_status_by_agent: dict[str, str] = field(default_factory=dict)
    summary: str = ""

    @staticmethod
    def intent_record_key(agent: str, intent_key: str) -> str:
        return f"{agent}:{intent_key}"

    def get_record(self, agent: str, intent_key: str) -> IntentRecord | None:
        if not intent_key:
            return None
        return self.intents.get(self.intent_record_key(agent, intent_key))

    def records_for_intent(
        self,
        intent_key: str,
        *,
        statuses: set[str] | None = None,
        exclude_agent: str | None = None,
    ) -> list[IntentRecord]:
        if not intent_key:
            return []
        return [
            rec
            for rec in self.intents.values()
            if rec.intent_key == intent_key
            and (exclude_agent is None or rec.owner != exclude_agent)
            and (statuses is None or rec.status in statuses)
        ]

    def latest_record_for_intent(
        self,
        intent_key: str,
        *,
        statuses: set[str] | None = None,
        exclude_agent: str | None = None,
    ) -> IntentRecord | None:
        candidates = self.records_for_intent(
            intent_key,
            statuses=statuses,
            exclude_agent=exclude_agent,
        )
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda rec: (
                rec.turn_completed or 0,
                rec.turn_claimed or 0,
                rec.owner,
            ),
        )

    def get_active_owner(self, intent_key: str, exclude: str) -> str | None:
        pending = self.pending_claims.get(intent_key)
        if pending and pending != exclude:
            return pending
        rec = self.latest_record_for_intent(
            intent_key,
            statuses={"in_progress"},
            exclude_agent=exclude,
        )
        if rec:
            return rec.owner
        return None

    def is_done(self, intent_key: str) -> bool:
        rec = self.latest_record_for_intent(intent_key, statuses={"done"})
        return rec is not None and rec.status == "done"

    def last_done_agent(self, intent_key: str) -> str | None:
        rec = self.latest_record_for_intent(intent_key, statuses={"done"})
        return rec.owner if (rec and rec.status == "done") else None

    def can_reenter(self, intent_key: str, agent: str) -> bool:
        rec = self.latest_record_for_intent(intent_key, statuses={"done"})
        if not rec or rec.status != "done":
            return True
        return rec.owner != agent

    def get_output(self, intent_key: str) -> str | None:
        rec = self.latest_record_for_intent(intent_key, statuses={"done"})
        if rec is None:
            rec = self.latest_record_for_intent(intent_key)
        return rec.output if rec else None

    def active_intent_keys(self) -> list[str]:
        keys = [
            self.last_intent_key_by_agent.get(agent, "")
            for agent, status in self.last_status_by_agent.items()
            if status == "in_progress"
        ]
        return list(dict.fromkeys(k for k in keys if k))

    def done_intent_keys(self) -> list[str]:
        return list(
            dict.fromkeys(
                rec.intent_key for rec in self.intents.values() if rec.status == "done"
            )
        )

    def record_event(
        self,
        *,
        agent: str,
        event_type: str,
        turn: int,
        intent: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            MemoryEvent(
                agent=agent,
                event_type=event_type,
                turn=turn,
                intent=intent or "unspecified",
                details=details or {},
            )
        )

    def get_snapshot(
        self,
        max_events: int = cfg.LAST_K_MESSAGES,
        max_failures: int = cfg.LAST_K_MESSAGES,
    ) -> dict:
        recent_failures = sorted(
            self.failure_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:max_failures]
        return {
            "recent_events": [e.__dict__ for e in self.events[-max_events:]],
            "intents": {k: v.__dict__ for k, v in self.intents.items()},
            "dependencies": [d.__dict__ for d in self.dependencies],
            "blockers": {k: v.__dict__ for k, v in self.blockers.items()},
            "recent_failures": dict(recent_failures),
            "last_status_by_agent": dict(self.last_status_by_agent),
            "last_intent_key_by_agent": dict(self.last_intent_key_by_agent),
            "summary": self.summary,
        }


@dataclass
class RunState:
    goal: str
    agents: list[AgentState]
    memory: SharedMemory = field(default_factory=SharedMemory)
    messages: list[Message] = field(default_factory=list)
    message_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    state_change: asyncio.Condition = field(default_factory=asyncio.Condition)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    state_change_seq: int = 0
    next_turn: int = 0
    last_progress_turn: int = 0
    task_complete: bool = False

    @classmethod
    async def create(cls, goal: str, agents: list[AgentState]) -> "RunState":
        return cls(
            goal=goal,
            agents=agents,
            memory=SharedMemory(),
            messages=[],
            message_lock=asyncio.Lock(),
            state_change=asyncio.Condition(),
            stop_event=asyncio.Event(),
        )
