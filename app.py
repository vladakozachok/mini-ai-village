from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from village.engine import run_group
from village.state import create_run_state
from village.types import RunState


LOG_DIR = Path("logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI village group chat.")
    parser.add_argument("goal", help="Goal for the agents to collaborate on.")
    return parser.parse_args()


def format_transcript(state: RunState) -> str:
    lines: list[str] = [f"GOAL: {state.goal}", ""]
    for i, message in enumerate(state.messages, start=1):
        lines.append(f"[{i}] {message.speaker}:")
        lines.append(message.content)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_log(state: RunState) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = LOG_DIR / f"run-{timestamp}.txt"
    path.write_text(format_transcript(state), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    state = create_run_state(args.goal)
    run_group(state)

    print(format_transcript(state), end="")
    log_path = write_log(state)
    print(f"\nSaved transcript: {log_path}")


if __name__ == "__main__":
    main()
