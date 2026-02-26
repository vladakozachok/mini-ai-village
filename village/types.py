import asyncio

from dataclasses import dataclass, field
import village.config as cfg
import time
import enum
from typing import Any

from village.agent_web_use_orchestration.browser_env import BrowserEnvironment, ExecutionResult
from village.agent_web_use_orchestration.actions import Action

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
    config: AgentConfig
    browser: BrowserEnvironment = field(init=False)
    actions: list[tuple[list[Action], list[ExecutionResult]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.browser = BrowserEnvironment(headless=True)
    
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

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def provider(self) -> cfg.Provider:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def system_prompt(self) -> str:
        return self.config.system_prompt

@dataclass
class Message:
    speaker: str
    raw_response: str
    message: str
    turn: int | None = None
    actions: list[Action] = field(default_factory=list)
    action_results: list[ExecutionResult] = field(default_factory=list) 
    timestamp: float = field(default_factory=time.time)
    input_text: str | None = None
    observation: dict[str, Any] | None = None

@dataclass
class IntentLease:
    owner: str
    status: str
    expires_at_turn: int

@dataclass
class Artifact:
    intent: str
    value: str
    turn: int
    by_agent: str

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
    events: list[MemoryEvent] = field(default_factory=list)
    active_intents: dict[str, IntentLease] = field(default_factory=dict)
    done_intents: dict[str, int] = field(default_factory=dict)
    dependencies: list[Dependency] = field(default_factory=list)
    blockers: dict[str, Blocker] = field(default_factory=dict)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    failure_counts: dict[str, int] = field(default_factory=dict)
    last_status_by_agent: dict[str, str] = field(default_factory=dict)
    summary: str = ""

    def get_snapshot(
        self,
        max_events: int = cfg.LAST_K_MESSAGES,
        max_failures: int = cfg.LAST_K_MESSAGES,
    ) -> dict:
        recent_failures = sorted(
            self.failure_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:max_failures]
        return {
            "recent_events": [e.__dict__ for e in self.events[-max_events:]],
            "active_intents": {k: v.__dict__ for k, v in self.active_intents.items()},
            "done_intents": dict(self.done_intents),
            "dependencies": [d.__dict__ for d in self.dependencies],
            "blockers": {k: v.__dict__ for k, v in self.blockers.items()},
            "artifacts": {k: v.__dict__ for k, v in self.artifacts.items()},
            "recent_failures": dict(recent_failures),
            "last_status_by_agent": dict(self.last_status_by_agent),
            "summary": self.summary,
        }


@dataclass
class RunState:
    goal: str
    agents: list[AgentState]
    memory: SharedMemory = field(default_factory=SharedMemory)
    messages: list[Message] = field(default_factory=list)
    message_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    next_turn: int = 0
    task_complete: bool = False
