from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
STOPWORDS = {
    "a", "an", "the", "my", "i", "me", "in", "on", "at", "to", "of", "for",
    "and", "or", "from", "with", "that", "this", "it", "is", "are", "be", "as",
}


def tokens(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOPWORDS]


@dataclass(frozen=True)
class Skill:
    id: str
    title: str
    pattern: str
    tip: str
    # Historical policies may retain provenance metadata.  H5 never injects or
    # otherwise consumes it at runtime.
    source_task_ids: list[str] | None = None


class H5SkillGuidance:
    """Retrieve a small procedural hint once, before interaction begins.

    H5 does not select APIs, modify tool docs, issue actions, or receive runtime
    state from any other layer.
    """

    def __init__(
        self,
        skills_path: str | Path,
        top_k: int = 1,
        min_overlap_tokens: int = 2,
    ):
        rows = json.loads(Path(skills_path).read_text(encoding="utf-8"))
        self.skills = [Skill(**row) for row in rows]
        self.top_k = max(0, top_k)
        self.min_overlap_tokens = max(1, min_overlap_tokens)

    def retrieve(self, instruction: str) -> list[Skill]:
        query = Counter(tokens(instruction))
        scored: list[tuple[float, Skill]] = []
        for skill in self.skills:
            document = Counter(tokens(skill.pattern + " " + skill.title))
            overlap = set(query) & set(document)
            if len(overlap) < self.min_overlap_tokens:
                continue
            score = sum(
                (1 + math.log1p(query[word])) * (1 + math.log1p(document[word]))
                for word in overlap
            )
            if score > 0:
                scored.append((score, skill))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [skill for _, skill in scored[: self.top_k]]

    @staticmethod
    def format(skills: list[Skill]) -> str:
        if not skills:
            return ""
        lines = ["[H5 procedural guidance]"]
        lines.extend(f"- {skill.title}: {skill.tip}" for skill in skills)
        return "\n".join(lines)
