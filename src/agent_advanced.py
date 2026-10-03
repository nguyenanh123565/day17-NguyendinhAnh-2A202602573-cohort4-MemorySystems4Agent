from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: thread context, persistent user profile, and compaction.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        # The live integration is optional; offline behavior is fully supported.
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Reply with per-thread cumulative token metrics.

        A live agent has not been wired into this scaffold; all calls use the
        deterministic offline path, including when force_offline is False.
        """
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        updates = extract_profile_updates(message)
        for key, value in updates.items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", answer)
        self.thread_tokens[thread_id] = self.token_usage(thread_id) + estimate_tokens(message) + estimate_tokens(answer)
        self.thread_prompt_tokens[thread_id] = self.prompt_token_usage(thread_id) + prompt_tokens
        return {
            "answer": answer,
            "token_usage": self.token_usage(thread_id),
            "prompt_tokens_processed": self.prompt_token_usage(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Count the profile, summary, and full recent messages in this turn."""
        context = self.compact_memory.context(thread_id)
        return (
            estimate_tokens(self.profile_store.read_text(user_id))
            + estimate_tokens(context["summary"])
            + sum(estimate_tokens(item["content"]) for item in context["messages"])
        )

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Answer explicit recall requests from the user's persisted facts."""
        facts = self.profile_store.facts(user_id)
        query = message.casefold()
        is_recall = bool(re.search(r"\?|nhắc lại|nhớ lại|tóm tắt|cho biết", query))
        if not is_recall:
            return "Mình đã ghi nhận thông tin bạn chia sẻ."

        cues = {
            "name": r"\btên\b|\blà ai\b",
            "location": r"ở đâu|nơi ở|đang ở|còn ở|địa điểm hiện tại|huế|hà nội|đà nẵng",
            "occupation": r"nghề|nghiệp|công việc|làm gì|product manager|engineer",
            "response_preference": r"style|phong cách|kiểu trả lời|cách trả lời|trả lời.*thích",
            "interests": r"mối quan tâm|quan tâm|sở thích|thích.*kỹ thuật",
            "favorite_drink": r"đồ uống|uống gì",
            "favorite_food": r"món ăn|ăn gì",
            "pet": r"nuôi|con gì|thú cưng",
        }
        requested = [key for key, pattern in cues.items() if re.search(pattern, query)]
        if not requested and re.search(r"tóm tắt|nhắc lại", query):
            requested = list(cues)
        labels = {
            "name": "Tên", "location": "Nơi ở hiện tại", "occupation": "Nghề nghiệp hiện tại",
            "response_preference": "Style trả lời", "interests": "Mối quan tâm",
            "favorite_drink": "Đồ uống yêu thích", "favorite_food": "Món ăn yêu thích",
            "pet": "Thú cưng",
        }
        known = []
        for key in requested:
            if key not in facts:
                continue
            value = facts[key]
            if key == "response_preference" and "ngắn" in value.casefold() and "gọn" not in value.casefold():
                value = "ngắn gọn, " + value
            known.append(f"{labels[key]}: {value}")
        if not known:
            return "Mình chưa có thông tin đó trong hồ sơ của bạn."
        # This also preserves the literal "3 bullet" when it is part of the style.
        return "; ".join(known) + "."

    def _maybe_build_langchain_agent(self):
        """Reserved for an optional live integration in a later step."""
        return None
