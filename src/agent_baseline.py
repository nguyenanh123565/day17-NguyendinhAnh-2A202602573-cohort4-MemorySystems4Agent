from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Full conversation history in RAM, isolated strictly by thread_id."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None if force_offline else self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # user_id is intentionally never a memory key.
        if self.langchain_agent is None:
            return self._reply_offline(thread_id, message)
        result = self.langchain_agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        last = result["messages"][-1]
        content = last["content"] if isinstance(last, dict) else last.content
        if isinstance(content, list):
            answer = "\n".join(
                block if isinstance(block, str) else block.get("text", "")
                for block in content
                if isinstance(block, (str, dict))
            )
        else:
            answer = str(content)
        return self._record_turn(thread_id, message, answer)

    def token_usage(self, thread_id: str) -> int:
        state = self.sessions.get(thread_id)
        return state.token_usage if state else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        state = self.sessions.get(thread_id)
        return state.prompt_tokens_processed if state else 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _record_turn(self, thread_id: str, message: str, answer: str) -> dict[str, Any]:
        """Heuristic metrics: new user + assistant tokens, and full input history.

        Returned metrics are cumulative for this thread, matching the accessors.
        Prompt accounting includes the current user turn, before assistant output.
        """
        state = self.sessions.setdefault(thread_id, SessionState())
        state.messages.append({"role": "user", "content": message})
        state.prompt_tokens_processed += sum(estimate_tokens(m["content"]) for m in state.messages)
        state.token_usage += estimate_tokens(message) + estimate_tokens(answer)
        state.messages.append({"role": "assistant", "content": answer})
        return {
            "answer": answer,
            "token_usage": state.token_usage,
            "prompt_tokens_processed": state.prompt_tokens_processed,
        }

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        state = self.sessions.setdefault(thread_id, SessionState())
        # Reconstruct facts only from this thread's user messages; no profile store.
        facts: dict[str, str] = {}
        for turn in state.messages + [{"role": "user", "content": message}]:
            if turn["role"] == "user":
                facts.update(extract_profile_updates(turn["content"]))
        question = message.lower()
        requested = []
        cues = {
            "name": r"tên|là ai",
            "location": r"ở đâu|nơi ở|đang ở|còn ở",
            "occupation": r"nghề|công việc hiện tại",
            "response_preference": r"style|kiểu trả lời|phong cách|trả lời.*thích",
            "interests": r"mối quan tâm|sở thích|quan tâm.*gì",
            "favorite_drink": r"đồ uống",
            "favorite_food": r"món ăn",
            "pet": r"nuôi|con gì",
        }
        is_recall = bool(re.search(r"\?|nhắc lại|nhớ lại|tóm tắt|tên gì|ở đâu|nghề gì", question))
        if is_recall:
            requested = [key for key, pattern in cues.items() if re.search(pattern, question)]
            if not requested and re.search(r"tóm tắt|nhắc lại", question):
                requested = list(cues)
        if requested:
            known = [facts[key] for key in requested if key in facts]
            missing = [key for key in requested if key not in facts]
            answer = "; ".join(known) + "." if known else "Mình chưa có thông tin đó trong cuộc hội thoại này."
            if known and missing:
                answer += " Các thông tin còn lại chưa được cung cấp trong cuộc hội thoại này."
        else:
            answer = "Mình đã ghi nhận thông tin trong cuộc hội thoại này."
        return self._record_turn(thread_id, message, answer)

    def _maybe_build_langchain_agent(self):
        """Optional live agent with an in-memory thread checkpointer only."""
        if self.force_offline:
            return None
        model = build_chat_model(self.config.model)
        if model is None:
            return None
        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None
        return create_agent(model=model, tools=[], checkpointer=InMemorySaver())
