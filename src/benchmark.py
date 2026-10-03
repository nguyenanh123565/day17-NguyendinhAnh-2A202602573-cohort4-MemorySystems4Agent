from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Return the dataset exactly as stored, preserving order and all fields."""
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def recall_points(answer: str, expected: list[str]) -> float:
    """Full credit for all facts, half for some, zero for none."""
    if not expected:
        return 1.0
    found = sum(fact.casefold() in answer.casefold() for fact in expected)
    return 1.0 if found == len(expected) else 0.5 if found else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline score: half for a nonempty answer, half for fact coverage."""
    if not answer.strip():
        return 0.0
    coverage = sum(fact.casefold() in answer.casefold() for fact in expected) / len(expected) if expected else 1.0
    return 0.5 + 0.5 * coverage


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Run a dataset in order, using fresh thread IDs for every recall query.

    Token counters on agents are cumulative per thread, so sum their per-turn
    deltas. Profile growth is final minus initial bytes for benchmark users.
    """
    user_ids = {conversation["user_id"] for conversation in conversations}
    has_memory = callable(getattr(agent, "memory_file_size", None))
    initial_bytes = sum(agent.memory_file_size(user_id) for user_id in user_ids) if has_memory else 0
    agent_tokens = 0
    prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    used_threads: set[str] = set()

    def send(user_id: str, thread_id: str, message: str) -> str:
        nonlocal agent_tokens, prompt_tokens
        before_agent = agent.token_usage(thread_id)
        before_prompt = agent.prompt_token_usage(thread_id)
        result = agent.reply(user_id, thread_id, message)
        agent_tokens += agent.token_usage(thread_id) - before_agent
        prompt_tokens += agent.prompt_token_usage(thread_id) - before_prompt
        used_threads.add(thread_id)
        return result["answer"]

    for conversation_index, conversation in enumerate(conversations):
        user_id = conversation["user_id"]
        conversation_thread = f"conversation:{conversation_index}:{conversation['id']}"
        for turn in conversation["turns"]:
            send(user_id, conversation_thread, turn)
        for question_index, question in enumerate(conversation["recall_questions"]):
            recall_thread = f"recall:{conversation_index}:{question_index}"
            answer = send(user_id, recall_thread, question["question"])
            expected = question["expected_contains"]
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    final_bytes = sum(agent.memory_file_size(user_id) for user_id in user_ids) if has_memory else 0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=final_bytes - initial_bytes,
        compactions=sum(agent.compaction_count(thread_id) for thread_id in used_threads),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format both agents with the same six benchmark metrics."""
    headers = (
        "Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall",
        "Response quality", "Memory growth (bytes)", "Compactions",
    )
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = (
            row.agent_name, str(row.agent_tokens_only), str(row.prompt_tokens_processed),
            f"{row.recall_score:.2f}", f"{row.response_quality:.2f}",
            str(row.memory_growth_bytes), str(row.compactions),
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    """Run both offline suites against independent, initially empty state."""
    config = load_config(Path(__file__).resolve().parent.parent)
    suites = (
        ("Standard Benchmark", "conversations.json"),
        ("Long-Context Stress Benchmark", "advanced_long_context.json"),
    )
    for title, filename in suites:
        conversations = load_conversations(config.data_dir / filename)
        with TemporaryDirectory(prefix="memory-benchmark-") as temporary:
            rows = []
            for agent_name, agent_class in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)):
                isolated_config = replace(config, state_dir=Path(temporary) / agent_name.lower())
                agent = agent_class(isolated_config, force_offline=True)
                rows.append(run_agent_benchmark(agent_name, agent, conversations, isolated_config))
        print(f"{title}\n{format_rows(rows)}\n")


if __name__ == "__main__":
    main()
