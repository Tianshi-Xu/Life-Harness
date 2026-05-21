"""Policy RAG: lightweight retrieval-augmented policy injection.

Two usage modes:

Mode 1 – H5 build-time injection (``get_policy_chunks``)
    Retrieve the top-k most relevant policy sections given a task description
    and append them to the system prompt at agent-build time.

Mode 2 – Retrieval-as-a-Tool (``make_policy_retrieval_tool``)
    Expose a ``retrieve_policy(query)`` tool so the agent can look up policy
    sections on demand during the conversation.  In this mode the full policy
    is NOT injected into the system prompt — only the structural "Domain Basic"
    section is kept (data-model facts, not procedural rules).

No external ML dependencies — uses pure-Python BM25-lite scoring.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from tau2.environment.toolkit import ToolKitBase, ToolType

# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can of in on at to for "
    "with by from up about into through during before after above below "
    "between each few more most other some such no nor not only own "
    "same so than too very just but and or if its it this that these "
    "those i you he she we they what which who when where why how all "
    "both any each few more most other some such no nor not".split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, no stop words."""
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Structural prefix vs procedural rules (for prompt + retrieval split)
# ---------------------------------------------------------------------------

_POLICY_TOOL_INSTRUCTION = """
## Policy lookup (retrieve_policy)

Detailed operational rules (booking, modification, cancellation, refunds,
compensation, etc.) are intentionally omitted from this system prompt.

When the user request involves policy-sensitive actions (book/modify/cancel,
refund, compensation, baggage, insurance, cabin, passenger changes), you MUST
call **retrieve_policy** before any write/action tool call.

If you have not called **retrieve_policy** for the current policy-sensitive
topic yet, do not execute write/action tools. Ask a clarification question or
call **retrieve_policy** first.

Use short, focused queries, for example:
- "cancel flight eligibility and insurance"
- "basic economy flight change rules"
- "compensation eligibility delayed flight"
- "baggage allowance gold member"
""".strip()


def split_structural_prefix_and_rules(policy_text: str) -> tuple[str, str]:
    """Split domain policy into (structural_prefix, rules_corpus).

    *Structural prefix* — kept at the start of the agent system prompt:
    title, role / conduct instructions, and the **Domain Basic** section
    (data models: users, products, flights, orders, etc.).

    *Rules corpus* — procedural sections (everything after Domain Basic under
    other ``##`` headings).  Only this part is indexed for BM25 retrieval and
    returned by ``retrieve_policy``.

    Domains without ``##`` headings (e.g. mock) fall back to a minimal prefix
    and use the full document as the rules corpus.
    """
    text = policy_text.strip()
    parts = re.split(r"(?m)^(?=## [^#])", text)
    preamble = parts[0].strip() if parts else ""

    structural_blocks: list[str] = []
    rule_blocks: list[str] = []
    for block in parts[1:]:
        block = block.strip()
        if not block:
            continue
        first_line = block.split("\n", 1)[0].strip().lower()
        title = first_line.lstrip("#").strip()
        if title.startswith("domain basic"):
            structural_blocks.append(block)
        else:
            rule_blocks.append(block)

    # Keep only minimal non-procedural context from preamble.
    # A long preamble can make the model think it already knows all rules.
    preamble_lines = [ln.strip() for ln in preamble.splitlines() if ln.strip()]
    minimal_preamble: list[str] = []
    if preamble_lines:
        # Keep title if present
        if preamble_lines[0].startswith("#"):
            minimal_preamble.append(preamble_lines[0])
        # Keep "current time" line if present (useful for time-based policy checks)
        for ln in preamble_lines[1:]:
            if "current time" in ln.lower():
                minimal_preamble.append(ln)
                break

    prefix_parts = ["\n\n".join(minimal_preamble).strip() if minimal_preamble else ""]
    prefix_parts.extend(structural_blocks)
    prefix = "\n\n".join(p for p in prefix_parts if p).strip()

    rules = "\n\n".join(rule_blocks).strip()
    if not rules:
        # e.g. mock policy: no ## sections — index everything for retrieval
        rules = text
        if preamble:
            prefix = preamble.split("\n", 1)[0].strip() if preamble.startswith("#") else preamble
        else:
            prefix = ""

    return prefix, rules


# ---------------------------------------------------------------------------
# Core retriever
# ---------------------------------------------------------------------------

class PolicyRetriever:
    """Split a policy document into chunks and rank them by query relevance.

    Parameters
    ----------
    policy_text:
        Full policy document text (Markdown).
    min_chunk_tokens:
        Chunks with fewer tokens than this are merged with the previous chunk
        to avoid tiny, uninformative sections.
    """

    def __init__(self, policy_text: str, min_chunk_tokens: int = 20) -> None:
        self._raw_chunks = self._split(policy_text, min_chunk_tokens)
        self._token_lists = [_tokenize(c) for c in self._raw_chunks]
        self._idf = self._build_idf()

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split(text: str, min_tokens: int) -> list[str]:
        """Split by smallest policy section (prefer ``###`` over ``##``).

        Strategy:
        1) Split into ``##`` sections.
        2) For each ``##`` section, if there are ``###`` subsections, each
           subsection becomes its own chunk with the parent ``##`` title kept
           as context.
        3) Merge tiny chunks upward to avoid uninformative fragments.
        """
        sec2_sections = re.findall(
            r"(?ms)^## [^\n]+\n.*?(?=^## |\Z)",
            text.strip(),
        )
        if not sec2_sections:
            sec2_sections = [text.strip()]

        chunks: list[str] = []
        for sec2 in sec2_sections:
            sec2 = sec2.strip()
            if not sec2:
                continue

            sec2_title_match = re.match(r"(?m)^(## [^\n]+)", sec2)
            sec2_title = sec2_title_match.group(1) if sec2_title_match else "## Section"

            sec3_sections = re.findall(
                r"(?ms)^### [^\n]+\n.*?(?=^### |\Z)",
                sec2,
            )
            if sec3_sections:
                for sec3 in sec3_sections:
                    sec3 = sec3.strip()
                    if not sec3:
                        continue
                    chunks.append(f"{sec2_title}\n\n{sec3}")
            else:
                chunks.append(sec2)

        merged: list[str] = []
        for chunk in chunks:
            tokens = _tokenize(chunk)
            if merged and len(tokens) < min_tokens:
                merged[-1] = merged[-1] + "\n\n" + chunk
            else:
                merged.append(chunk)
        return merged

    # ------------------------------------------------------------------
    # IDF
    # ------------------------------------------------------------------

    def _build_idf(self) -> dict[str, float]:
        N = len(self._token_lists)
        df: Counter[str] = Counter()
        for tlist in self._token_lists:
            df.update(set(tlist))
        return {
            term: math.log((N + 1) / (count + 1)) + 1.0
            for term, count in df.items()
        }

    # ------------------------------------------------------------------
    # Scoring (BM25-lite)
    # ------------------------------------------------------------------

    _K1 = 1.5
    _B = 0.75

    def _score(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        if not doc_tokens:
            return 0.0
        avgdl = sum(len(t) for t in self._token_lists) / max(len(self._token_lists), 1)
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for qt in query_tokens:
            if qt not in tf:
                continue
            idf = self._idf.get(qt, 1.0)
            freq = tf[qt]
            numerator = freq * (self._K1 + 1)
            denominator = freq + self._K1 * (1 - self._B + self._B * dl / max(avgdl, 1))
            score += idf * (numerator / denominator)
        return score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Return the ``top_k`` most relevant policy chunks for *query*."""
        q_tokens = _tokenize(query)
        if not q_tokens:
            return self._raw_chunks[:top_k]
        scored = [
            (self._score(q_tokens, doc_toks), chunk)
            for doc_toks, chunk in zip(self._token_lists, self._raw_chunks)
        ]
        scored.sort(key=lambda x: -x[0])
        # Return only chunks with non-zero score
        return [chunk for score, chunk in scored[:top_k] if score > 0]


# ---------------------------------------------------------------------------
# Domain-level cache — build a retriever once per rules corpus
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _get_retriever_for_rules_corpus(rules_corpus: str) -> PolicyRetriever:
    if not rules_corpus.strip():
        return PolicyRetriever("## (empty)\n\nNo procedural rules available.")
    return PolicyRetriever(rules_corpus)


def get_policy_chunks(
    policy_text: str,
    query: str,
    top_k: int = 3,
) -> list[str]:
    """Return the top-k most relevant **procedural** policy chunks.

    Only the rules corpus (excludes preamble + Domain Basic) is searched.

    Args:
        policy_text: Full policy document (loaded once per domain).
        query: Free-text description of the current task or user request.
        top_k: Maximum number of chunks to return.

    Returns:
        List of relevant policy text chunks (may be shorter than top_k if
        the policy has few sections or the query matches nothing).
    """
    _, rules = split_structural_prefix_and_rules(policy_text)
    retriever = _get_retriever_for_rules_corpus(rules)
    return retriever.retrieve(query, top_k=top_k)


# ---------------------------------------------------------------------------
# Toolkit wrapper — register retrieve_policy on the environment for replay
# ---------------------------------------------------------------------------


class ToolKitWithPolicyRetrieval(ToolKitBase):
    """Wraps a domain ``ToolKitBase`` and adds ``retrieve_policy``.

    ``Environment.set_state`` replay checks ``tools.has_tool(name)`` on the
    assistant toolkit.  Policy retrieval must live on that toolkit (not only
    on the agent's tool list) so trajectories containing ``retrieve_policy``
    replay without error.  The tool is non-mutating: replay skips re-execution.
    """

    def __init__(self, inner: ToolKitBase, policy_text: str, top_k: int = 3) -> None:
        super().__init__(db=inner.db)
        self._inner = inner
        self._policy_text = policy_text
        self._top_k = top_k

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    @property
    def tools(self) -> dict:
        out = dict(self._inner.tools)

        def _retrieve_policy(query: str) -> str:
            chunks = get_policy_chunks(self._policy_text, query, top_k=self._top_k)
            if not chunks:
                return "No relevant policy sections found for this query."
            return "\n\n---\n\n".join(chunks)

        out["retrieve_policy"] = _retrieve_policy
        return out

    def use_tool(self, tool_name: str, **kwargs) -> str:
        if tool_name == "retrieve_policy":
            q = kwargs.get("query", "")
            return self.tools["retrieve_policy"](q)
        return self._inner.use_tool(tool_name, **kwargs)

    def has_tool(self, tool_name: str) -> bool:
        if tool_name == "retrieve_policy":
            return True
        return self._inner.has_tool(tool_name)

    def tool_mutates_state(self, tool_name: str) -> bool:
        if tool_name == "retrieve_policy":
            return False
        return self._inner.tool_mutates_state(tool_name)

    def tool_type(self, tool_name: str) -> ToolType:
        if tool_name == "retrieve_policy":
            return ToolType.READ
        return self._inner.tool_type(tool_name)

    def get_tools(self, include=None):
        if include is not None:
            inner_include = [n for n in include if n != "retrieve_policy"]
            d: dict = {}
            if inner_include:
                d.update(self._inner.get_tools(include=inner_include))
            if "retrieve_policy" in include:
                d["retrieve_policy"] = make_policy_retrieval_tool(
                    self._policy_text, self._top_k
                )
            return d
        d = dict(self._inner.get_tools(include=None))
        d["retrieve_policy"] = make_policy_retrieval_tool(self._policy_text, self._top_k)
        return d

    def update_db(self, update_data=None):
        self._inner.update_db(update_data)
        self.db = self._inner.db

    def get_db_hash(self) -> str:
        return self._inner.get_db_hash()

    def is_discoverable(self, tool_name: str) -> bool:
        if tool_name == "retrieve_policy":
            return False
        return self._inner.is_discoverable(tool_name)

    def get_discoverable_tools(self):
        return self._inner.get_discoverable_tools()

    def has_discoverable_tool(self, tool_name: str) -> bool:
        if tool_name == "retrieve_policy":
            return False
        return self._inner.has_discoverable_tool(tool_name)

    def get_statistics(self) -> dict:
        s = dict(self._inner.get_statistics())
        s["num_tools"] = s.get("num_tools", 0) + 1
        s["num_read_tools"] = s.get("num_read_tools", 0) + 1
        return s


def wrap_toolkit_with_policy_retrieval(
    inner: ToolKitBase,
    policy_text: str,
    top_k: int = 3,
) -> ToolKitWithPolicyRetrieval:
    """Return ``inner`` wrapped so ``retrieve_policy`` exists on the toolkit."""
    return ToolKitWithPolicyRetrieval(inner, policy_text, top_k)


# ---------------------------------------------------------------------------
# Retrieval-as-a-Tool
# ---------------------------------------------------------------------------

def extract_domain_basic(policy_text: str) -> str:
    """Return the structural prefix (preamble + Domain Basic).

    Deprecated name kept for compatibility; prefer ``split_structural_prefix_and_rules``.
    """
    prefix, _ = split_structural_prefix_and_rules(policy_text)
    return prefix


def make_policy_retrieval_tool(policy_text: str, top_k: int = 3):
    """Create a ``retrieve_policy`` Tool the agent can call on demand.

    Indexes **procedural rules only** (excludes the structural prefix).  Pair
    with ``split_structural_prefix_and_rules`` to build a short system prompt.

    Args:
        policy_text: Full policy document text for the domain.
        top_k: Number of policy sections to return per query (default 3).

    Returns:
        A ``tau2.environment.tool.Tool`` instance named ``retrieve_policy``.
    """
    from tau2.environment.tool import Tool

    _, rules_corpus = split_structural_prefix_and_rules(policy_text)
    retriever = _get_retriever_for_rules_corpus(rules_corpus)
    _top_k = top_k

    def retrieve_policy(query: str) -> str:
        """Look up relevant policy sections for the current task.

        Call this tool whenever you need to check the rules before taking an
        action (booking, modifying, cancelling, or compensating a customer).
        Provide a short description of what you want to check.

        Args:
            query: A short description of the policy topic to look up,
                e.g. 'cancel flight eligibility', 'basic economy rules',
                'baggage allowance silver member', 'compensation certificate'.

        Returns:
            The most relevant policy sections as plain text.
        """
        chunks = retriever.retrieve(query, top_k=_top_k)
        if not chunks:
            return "No relevant policy sections found for this query."
        return "\n\n---\n\n".join(chunks)

    return Tool(func=retrieve_policy)


def policy_tool_instruction() -> str:
    """Short system-prompt appendix explaining ``retrieve_policy``."""
    return _POLICY_TOOL_INSTRUCTION


def dump_policy_chunks_for_debug(
    policy_text: str,
    output_path: str | Path,
    *,
    min_chunk_tokens: int = 20,
) -> Path:
    """Write current policy chunking result to a human-readable file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    retriever = PolicyRetriever(policy_text, min_chunk_tokens=min_chunk_tokens)

    lines: list[str] = []
    lines.append("# Policy Chunk Debug")
    lines.append("")
    lines.append(f"- total_chunks: {len(retriever._raw_chunks)}")
    lines.append(f"- min_chunk_tokens: {min_chunk_tokens}")
    lines.append("")
    for i, chunk in enumerate(retriever._raw_chunks, start=1):
        token_count = len(_tokenize(chunk))
        lines.append(f"## Chunk {i} (tokens={token_count})")
        lines.append("")
        lines.append(chunk)
        lines.append("")
        lines.append("---")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def dump_actual_policy_layout_for_rag_tool(
    policy_text: str,
    output_path: str | Path,
    *,
    min_chunk_tokens: int = 20,
) -> Path:
    """Dump the real tool-mode layout: prompt prefix vs retrievable chunks.

    This matches the actual retrieval-as-a-tool path:
    - structural_prefix: kept in initial system prompt
    - rules_corpus chunks: searchable by retrieve_policy
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    structural_prefix, rules_corpus = split_structural_prefix_and_rules(policy_text)
    retriever = PolicyRetriever(rules_corpus, min_chunk_tokens=min_chunk_tokens)

    lines: list[str] = []
    lines.append("# Actual Policy Layout (RAG Tool Mode)")
    lines.append("")
    lines.append("## Part A: In Initial System Prompt (structural_prefix)")
    lines.append("")
    if structural_prefix.strip():
        lines.append(structural_prefix)
    else:
        lines.append("(empty)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Part B: Retrieved Later via `retrieve_policy` (rules_corpus chunks)")
    lines.append("")
    lines.append(f"- total_chunks: {len(retriever._raw_chunks)}")
    lines.append(f"- min_chunk_tokens: {min_chunk_tokens}")
    lines.append("")
    for i, chunk in enumerate(retriever._raw_chunks, start=1):
        token_count = len(_tokenize(chunk))
        lines.append(f"### Retrievable Chunk {i} (tokens={token_count})")
        lines.append("")
        lines.append(chunk)
        lines.append("")
        lines.append("---")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
