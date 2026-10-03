from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore


def make_config(tmp_path: Path):
    """Use an isolated profile directory and a short compaction threshold."""
    return replace(
        load_config(tmp_path),
        state_dir=tmp_path / "state",
        compact_threshold_tokens=700,
        compact_keep_messages=4,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "state" / "profiles")
    path = store.write_text("person", "Nơi ở: Đà Nẵng\n")
    assert path == store.path_for("person")
    assert path.is_file()
    assert store.read_text("person") == "Nơi ở: Đà Nẵng\n"
    assert store.edit_text("person", "Đà Nẵng", "Huế") is True
    content = store.read_text("person")
    assert "Huế" in content
    assert "Đà Nẵng" not in content
    assert store.file_size("person") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(threshold_tokens=80, keep_messages=3)
    thread_id = "long-thread"
    for index in range(12):
        manager.append(thread_id, "user", f"Message {index}: " + "x" * 180)
    context = manager.context(thread_id)
    assert manager.compaction_count(thread_id) > 0
    assert len(context["messages"]) <= manager.keep_messages
    assert context["messages"][-1]["content"].startswith("Message 11:")
    assert context["summary"]


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    user_id = "person"
    statement = "Mình tên là Linh."
    question = "Mình tên gì?"
    for agent in (baseline, advanced):
        agent.reply(user_id, "thread-1", statement)
    baseline_answer = baseline.reply(user_id, "thread-2", question)["answer"]
    advanced_answer = advanced.reply(user_id, "thread-2", question)["answer"]
    assert "Linh" in advanced_answer
    assert "Linh" not in baseline_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    thread_id = "long-thread"
    for index in range(40):
        message = f"Turn {index}: " + "x" * 500
        baseline.reply("person", thread_id, message)
        advanced.reply("person", thread_id, message)
    assert advanced.compaction_count(thread_id) > 0
    assert advanced.prompt_token_usage(thread_id) < baseline.prompt_token_usage(thread_id)
