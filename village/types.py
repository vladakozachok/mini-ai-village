from dataclasses import dataclass, field
import village.config as cfg
import time

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
    memory: str = ""

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
    content: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class RunState:
    goal: str
    agents: list[AgentState]
    messages: list[Message] = field(default_factory=list)
    next_agent_index: int = 0
    task_complete: bool = False

    def get_next_agent(self) -> AgentState:
        return self.agents[self.next_agent_index]
    
    def update_next_agent(self) -> None:
        self.next_agent_index = (self.next_agent_index + 1) % len(self.agents)
