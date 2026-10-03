from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Deterministic ceiling of character count / four (including whitespace)."""
    return (len(text) + 3) // 4


@dataclass
class UserProfileStore:
    """Markdown profiles below root_dir (normally state/profiles)."""
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        if not user_id.strip():
            raise ValueError("user_id must not be empty")
        slug = re.sub(r"[^\w-]", "_", user_id, flags=re.UNICODE).strip("_")[:80] or "user"
        # Preserve ordinary ids; disambiguate transformed ids and Windows device names.
        reserved = re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])", slug)
        if slug != user_id or reserved:
            slug += "-" + hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
        root = self.root_dir.resolve()
        path = (root / slug / "User.md").resolve()
        if not path.is_relative_to(root):
            raise ValueError("Profile path escapes root_dir")
        return path

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        if not search_text or search_text == replacement:
            return False
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Read structured fact lines without interpreting arbitrary markdown."""
        result = {}
        for line in self.read_text(user_id).splitlines():
            match = re.fullmatch(r"- ([a-z_]+): (.*)", line)
            if match:
                try:
                    value = json.loads(match[2])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, str):
                    result[match[1]] = value
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Replace a fact in place, removing duplicate lines for that key."""
        if not re.fullmatch(r"[a-z_]+", key):
            raise ValueError("Invalid fact key")
        lines = self.read_text(user_id).splitlines() or ["# User profile", ""]
        entry = f"- {key}: {json.dumps(value, ensure_ascii=False)}"
        prefix = f"- {key}: "
        updated = []
        found = False
        for line in lines:
            if line.startswith(prefix):
                if not found:
                    updated.append(entry)
                    found = True
            else:
                updated.append(line)
        if not found:
            updated.append(entry)
        return self.write_text(user_id, "\n".join(updated) + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Conservative Vietnamese self-declarations; last assertion wins per key.

    Questions, hypothetical/temporary clauses and jokes are not profile evidence.
    Keys are name, location, occupation, response_preference, interests,
    favorite_food, favorite_drink and pet.
    """
    patterns = {
        "name": r"(?:\b(?:mình|tôi)\s+tên(?:\s+là)?|\btên\s+(?:mình|tôi)\s+là)\s+(.+)",
        "location": r"(?:\b(?:mình|tôi)\s+(?:vẫn\s+|hiện\s+|giờ\s+)?(?:đang\s+)?(?:sống\s+|sinh sống\s+)?ở|\bhiện\s+ở|\bnơi ở hiện tại(?:\s+của\s+(?:mình|tôi))?\s+là)\s+(.+)",
        "occupation": r"(?:\b(?:mình|tôi)\s+(?:vẫn\s+|hiện\s+|giờ\s+)?(?:đang\s+)?làm(?:\s+nghề)?|\bđang\s+làm|\bgiờ\s+chuyển\s+sang|\bnghề nghiệp(?:\s+hiện tại)?\s+(?:thì\s+)?(?:vẫn\s+)?là)\s+(.+)",
        "response_preference": r"(?:\b(?:mình|tôi)\s+(?:vẫn\s+)?muốn\s+(?:bạn\s+)?(?:style\s+)?(?:câu\s+)?trả lời|\bhãy\s+trả lời|\bstyle trả lời(?:\s+cũng)?\s+vẫn giữ nguyên\s*:)\s+(.+)",
        "interests": r"\b(?:mình|tôi)\s+(?:vẫn\s+|còn\s+|đang\s+)?(?:thích|quan tâm(?:\s+nhiều)?(?:\s+đến)?|hứng thú với)\s+(.+)",
        "favorite_drink": r"\bđồ uống yêu thích(?:\s+của\s+(?:mình|tôi))?\s+là\s+(.+)",
        "favorite_food": r"\bmón ăn yêu thích(?:\s+của\s+(?:mình|tôi))?\s+là\s+(.+)",
        "pet": r"\b(?:mình|tôi)\s+nuôi\s+(.+)",
    }
    updates = {}
    for sentence in re.findall(r"[^.!?;\n]+[.!?;\n]?", message):
        if sentence.rstrip().endswith("?"):
            continue
        sentence = sentence.strip().rstrip(".!;\n")
        if not sentence or re.search(r"\b(nếu|giả sử|hay là|đùa|tạm thời|hỏi|nhắc lại|ở đâu|tên gì|nghề gì|là gì)\b", sentence, re.I):
            continue
        # Strip negated old facts before looking for a positive correction.
        sentence = re.sub(r"(?:mình|tôi)\s+không còn.+?(?:nữa\s*,|giờ)", "giờ ", sentence, flags=re.I)
        for key, pattern in patterns.items():
            for match in re.finditer(pattern, sentence, re.I):
                value = re.split(r"\s+(?:và đang|và mình|chứ|dù|nhưng|cho|để|vì|trong giai đoạn|mỗi ngày|chưa|không đổi)\b", match[1], maxsplit=1, flags=re.I)[0]
                if key in {"name", "occupation"}:
                    value = value.split(",", 1)[0]
                if key == "location":
                    value = re.split(r"\s+và\b|,", value, maxsplit=1, flags=re.I)[0]
                    if re.search(r"\b(họp|quán|hôm nay|hai ngày)\b", value, re.I):
                        continue
                if key == "occupation" and re.search(r"\b(việc|code|sạch|benchmark|test)\b", value, re.I):
                    continue
                if key == "response_preference" and not re.search(r"ngắn|gọn|bullet|ví dụ|cấu trúc|trade-off|chi tiết|rõ|súc tích", value, re.I):
                    continue
                if key == "interests" and re.match(r"(?:tin|kiểu|cách|điều|nó|thứ|hơn|không)\b", value, re.I):
                    continue
                value = value.strip(" ,:")
                if value:
                    updates[key] = value
        # A multi-month relocation is stronger evidence than a short meeting trip.
        move = re.search(r"(?:mình|tôi)\s+đang làm việc ở\s+(.+?)\s+(?:vài|nhiều|\d+)\s+tháng\b", sentence, re.I)
        if move:
            updates["location"] = move[1].strip()
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Bounded offline excerpts plus latest explicitly declared profile facts."""
    if max_items <= 0 or not messages:
        return ""
    facts = {}
    excerpts = []
    for message in messages:
        content = " ".join(message["content"].split())
        if message["role"] == "summary":
            # Carry prior structured facts forward; later user updates replace them.
            for line in message["content"].splitlines():
                match = re.fullmatch(r"([a-z_]+): (.*)", line)
                if match and match[1] not in {"user", "assistant", "summary"}:
                    facts[match[1]] = match[2]
        if message["role"] == "user":
            facts.update(extract_profile_updates(content))
        excerpt = content if len(content) <= 240 else content[:160] + " … " + content[-77:]
        excerpts.append(f'{message["role"]}: {excerpt}')
    fact_lines = [f"{key}: {value[:240]}" for key, value in sorted(facts.items())]
    return "\n".join(fact_lines + excerpts[-max_items:])


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens <= 0 or self.keep_messages < 0:
            raise ValueError("Threshold must be positive; keep_messages must be nonnegative")

    def append(self, thread_id: str, role: str, content: str) -> None:
        context = self.context(thread_id)
        messages = context["messages"]
        messages.append({"role": role, "content": content})
        tokens = estimate_tokens(context["summary"]) + sum(estimate_tokens(m["content"]) for m in messages)
        old_count = len(messages) - self.keep_messages
        if tokens > self.threshold_tokens and old_count > 0:
            older = messages[:old_count]
            if context["summary"]:
                older = [{"role": "summary", "content": context["summary"]}] + older
            context["summary"] = summarize_messages(older)
            context["messages"] = messages[old_count:]
            context["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return self.context(thread_id)["compactions"]
