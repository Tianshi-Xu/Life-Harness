"""Banking Knowledge domain harness: H2 rules + H4 post-call annotations.

H2 rules (pre-execution, deterministic DB/state checks)
--------------------------------------------------------
LogVerificationUserIdRule
    Before log_verification: user_id must exist in db.users.  Catches every
    placeholder, hallucinated, or bypass-code value before it pollutes the DB.

LogVerificationTimeRule (not wired — kept for reference)
    Before log_verification: time_verified must match get_current_time().
    Replaced by auto-fill in use_tool() to avoid correction-loop regressions.

UnlockDiscoverableExistsRule
    Before unlock_discoverable_agent_tool: tool_id must exist in the registry.
    Catches hallucinated tool names at the unlock step before wasting a call.

CallDiscoverableUnlockedRule
    Before call_discoverable_agent_tool: tool must (a) exist and (b) have been
    unlocked (present in toolkit._agent_discoverable_tools_state).  Together
    with the auto-unlock on KB_search hit this enforces the full workflow
    without adding context length.

H4 annotators (post-execution, reactive)
-----------------------------------------
KBSearchDiscoverableToolAnnotator
    After KB_search: detects tool_ids in results and notes them.  Auto-unlock
    (handled in use_tool override) runs in parallel so the agent can skip
    directly to call_discoverable_agent_tool.

UnlockDiscoverableErrorAnnotator
    After unlock_discoverable_agent_tool fails: redirect to KB_search.

CallDiscoverableErrorAnnotator
    After call_discoverable_agent_tool fails: redirect, remind to unlock first.

CallDiscoverableSuccessAnnotator
    After call_discoverable_agent_tool succeeds: remind about
    give_discoverable_user_tool if a user-facing step is required.

GiveDiscoverableErrorAnnotator
    After give_discoverable_user_tool fails: redirect to KB_search.

EmailNotFoundAnnotator
    After get_user_information_by_email returns no records: try name lookup.

NameLookupEmailHintAnnotator
    After get_user_information_by_name succeeds: surface DB email so agent can
    detect a mismatch → account_ownership_dispute reason.

UserTransferRequestAnnotator
    After request_human_agent_transfer: prompt agent to call
    transfer_to_human_agents immediately.

ReferralsFoundAnnotator
    After get_referrals_by_user finds records: remind to submit_referral.
"""

from __future__ import annotations

import re
from typing import Any

from tau2.harness.base import HarnessAnnotator, HarnessedToolKitMixin

# ---------------------------------------------------------------------------
# Shared guidance text
# ---------------------------------------------------------------------------

_DISCOVERABLE_WORKFLOW = """\
⚠ Special operations require discovery first:
  1. KB_search → find the procedure document (it contains the tool_id).
  2. unlock_discoverable_agent_tool(agent_tool_name='<tool_id_from_doc>')
  3. call_discoverable_agent_tool(agent_tool_name='<tool_id_from_doc>', arguments='{{...}}')
Never guess tool IDs — they must come verbatim from a KB document."""

# Matches discoverable tool IDs: name ending with _NNNN (4+ digits).
# Document IDs end with _NNN (3 digits) and are excluded by requiring \d{4,}.
_TOOL_ID_RE = re.compile(r"\b([a-z][a-z0-9_]+_\d{4,})\b")


# ---------------------------------------------------------------------------
# H2 Rules (pre-execution, deterministic)
# ---------------------------------------------------------------------------


class LogVerificationUserIdRule:
    """H2: user_id passed to log_verification must exist in the users table.

    This is the single most impactful gate: it blocks every hallucinated,
    placeholder, or social-engineering bypass value (e.g. 'usr_12345',
    '9K2X7M4P1N8Q3R5T6A') before it writes to verification_history.
    """

    tool_name = "log_verification"

    def check(self, db: Any, user_id: str = "", **kwargs: Any) -> None:
        if not user_id or user_id not in db.users.data:
            raise ValueError(
                f"user_id '{user_id}' does not exist in the database. "
                "Retrieve the correct user_id first with "
                "get_user_information_by_email or get_user_information_by_name, "
                "then use the user_id from that result."
            )


class LogVerificationTimeRule:
    """H2: time_verified in log_verification must match the simulation clock.

    The agent is instructed to call get_current_time() for the timestamp.
    This rule enforces that: if the provided time_verified does not match
    the value returned by get_current_time(), the call is blocked and the
    agent is forced to call get_current_time() first.

    Requires toolkit access to call get_current_time().
    """

    _TIME_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+")

    tool_name = "log_verification"

    def check(
        self, db: Any, time_verified: str = "", toolkit: Any = None, **kwargs: Any
    ) -> None:
        if toolkit is None:
            return
        raw = toolkit.get_current_time()  # "The current time is 2025-11-14 03:40:00 EST."
        m = self._TIME_RE.search(raw)
        if not m:
            return  # can't parse — don't block
        expected = m.group(0)
        if time_verified != expected:
            raise ValueError(
                f"time_verified '{time_verified}' does not match the simulation clock. "
                f"Call get_current_time() first and use the exact timestamp it returns: "
                f"'{expected}'"
            )


class UnlockDiscoverableExistsRule:
    """H2: unlock_discoverable_agent_tool requires a valid, known tool_id.

    Catches hallucinated or guessed tool names at the unlock step — before
    the agent wastes a round-trip on an unknown tool.  The agent must get
    the tool_id from a KB document, not invent it.
    """

    tool_name = "unlock_discoverable_agent_tool"

    def check(
        self, db: Any, agent_tool_name: str = "", toolkit: Any = None, **kwargs: Any
    ) -> None:
        if toolkit is None:
            return
        if not toolkit.has_discoverable_tool(agent_tool_name):
            raise ValueError(
                f"Unknown tool '{agent_tool_name}'. "
                "Tool IDs must come verbatim from a KB document — never guess them. "
                "Use KB_search to find the procedure and its tool_id first."
            )


class CallDiscoverableUnlockedRule:
    """H2: call_discoverable_agent_tool requires a valid, unlocked tool_id.

    Two checks:
    1. tool_id must be a real discoverable tool (not guessed).
    2. tool_id must have been unlocked in this session (present in
       toolkit._agent_discoverable_tools_state).

    With auto-unlock on KB_search, condition 2 is usually satisfied
    automatically — this rule mainly catches pure hallucinations.
    Requires toolkit access (passed as toolkit=self by HarnessedToolKitMixin).
    """

    tool_name = "call_discoverable_agent_tool"

    def check(
        self, db: Any, agent_tool_name: str = "", toolkit: Any = None, **kwargs: Any
    ) -> None:
        if toolkit is None:
            return
        if not toolkit.has_discoverable_tool(agent_tool_name):
            raise ValueError(
                f"Unknown tool '{agent_tool_name}'. "
                "Tool IDs must come verbatim from a KB document — never guess them. "
                "Use KB_search to find the procedure and its tool_id first."
            )
        if agent_tool_name not in toolkit._agent_discoverable_tools_state:
            raise ValueError(
                f"Tool '{agent_tool_name}' exists but has not been unlocked in "
                "this session. Call "
                f"unlock_discoverable_agent_tool(agent_tool_name='{agent_tool_name}') "
                "before calling it."
            )


# ---------------------------------------------------------------------------
# H4 Annotators
# ---------------------------------------------------------------------------


class KBSearchDiscoverableToolAnnotator(HarnessAnnotator):
    """H4: Surface discoverable tool_ids found in KB_search results.

    When a KB document describes a special operation, its tool_id follows the
    pattern ``<snake_case_name>_<4+ digits>`` (e.g. ``transfer_funds_7291``).
    The auto-unlock in use_tool() immediately unlocks these tools; this
    annotator tells the agent to proceed directly to call_discoverable_agent_tool.
    """

    def annotate(self, db: Any, result: Any, **_: Any) -> str | None:
        result_str = str(result)
        matches = _TOOL_ID_RE.findall(result_str)
        if not matches:
            return None
        seen: set[str] = set()
        tool_ids = [m for m in matches if not (m in seen or seen.add(m))]
        ids_str = ", ".join(f"'{t}'" for t in tool_ids)
        return (
            f"⚠ Procedure document found. Tool ID(s): {ids_str}\n"
            f"These tools are auto-unlocked. You MUST call:\n"
            f"  call_discoverable_agent_tool(agent_tool_name='<id_from_above>', arguments='{{...}}')\n"
            f"Do NOT tell the user the operation is complete until you have made this call "
            f"and received a success response. No separate unlock step is needed."
        )


class UnlockDiscoverableErrorAnnotator(HarnessAnnotator):
    """H4: When unlock_discoverable_agent_tool returns 'Unknown agent tool', redirect."""

    def annotate(self, db: Any, result: Any, **_: Any) -> str | None:
        if not str(result).startswith("Error: Unknown agent tool"):
            return None
        return (
            "This tool_id is not recognised. It must come from a KB document — "
            "never guess it.\n" + _DISCOVERABLE_WORKFLOW
        )


class CallDiscoverableErrorAnnotator(HarnessAnnotator):
    """H4: When call_discoverable_agent_tool fails due to unknown/unlocked tool, redirect."""

    def annotate(self, db: Any, result: Any, **_: Any) -> str | None:
        result_str = str(result)
        if not (
            result_str.startswith("Error: Unknown agent tool")
            or "has not been unlocked" in result_str
        ):
            return None
        return (
            "This tool_id is not recognised or was not unlocked first.\n"
            + _DISCOVERABLE_WORKFLOW
        )


class CallDiscoverableSuccessAnnotator(HarnessAnnotator):
    """H4: After call_discoverable_agent_tool succeeds, remind about user-side tools.

    Some operations have a user-facing component: the KB document will say
    something like 'give the user tool X'.  This reminder fires after every
    successful agent-side call so the agent doesn't forget the give step.
    """

    def annotate(self, db: Any, result: Any, **_: Any) -> str | None:
        if str(result).startswith("Error:"):
            return None
        return (
            "Agent-side operation complete. "
            "If the KB document for this operation mentions a user-facing tool, "
            "call give_discoverable_user_tool(discoverable_tool_name='<name_from_KB>') now."
        )


class GiveDiscoverableErrorAnnotator(HarnessAnnotator):
    """H4: When give_discoverable_user_tool fails, redirect to KB_search."""

    def annotate(self, db: Any, result: Any, **_: Any) -> str | None:
        if not str(result).startswith("Error: Unknown discoverable tool"):
            return None
        return (
            "This user tool name is not recognised — it must come from a KB document.\n"
            "Call KB_search to find the procedure document that describes this "
            "user-facing operation. The document will specify the exact "
            "discoverable_tool_name to use with give_discoverable_user_tool."
        )


class EmailNotFoundAnnotator(HarnessAnnotator):
    """H4: When email lookup finds no records, suggest name-based lookup."""

    def annotate(self, db: Any, result: Any, **_: Any) -> str | None:
        if "No records found" not in str(result):
            return None
        return (
            "Email not found in database. "
            "Try get_user_information_by_name with the customer's full name. "
            "There is no phone-number lookup tool."
        )


class NameLookupEmailHintAnnotator(HarnessAnnotator):
    """H4: After get_user_information_by_name succeeds, surface the DB email.

    The agent must compare the returned email against what the customer stated.
    A mismatch signals a potential identity verification failure — the correct
    transfer reason in that case is ``account_ownership_dispute``.
    """

    _EMAIL_RE = re.compile(r"email:\s*(\S+)", re.IGNORECASE)

    def annotate(self, db: Any, result: Any, **_: Any) -> str | None:
        result_str = str(result)
        if "No records found" in result_str:
            return None
        m = self._EMAIL_RE.search(result_str)
        if not m:
            return None
        db_email = m.group(1).rstrip(",;")
        return (
            f"DB email on file: {db_email}. "
            "Compare this against the email address the customer stated. "
            "If they do not match, this is an identity verification discrepancy — "
            "use reason='account_ownership_dispute' if you must transfer."
        )


class UserTransferRequestAnnotator(HarnessAnnotator):
    """H4: When user submits a transfer request, prompt agent to complete it."""

    def annotate(self, db: Any, result: Any, **_: Any) -> str | None:
        result_str = str(result)
        if "Transfer request" not in result_str or "submitted" not in result_str:
            return None
        return (
            "The user has submitted a human transfer request. "
            "Call transfer_to_human_agents now to complete the transfer, "
            "using the most specific applicable reason code."
        )


class ReferralsFoundAnnotator(HarnessAnnotator):
    """H4: After get_referrals_by_user finds records, remind of next steps."""

    def annotate(self, db: Any, result: Any, **_: Any) -> str | None:
        result_str = str(result)
        if "No records found" in result_str or "Found 0" in result_str:
            return None
        if "record" not in result_str.lower():
            return None
        return (
            "Referral records retrieved. "
            "Use submit_referral to submit a new referral if the user requests it."
        )


# ---------------------------------------------------------------------------
# H4 Mixin  (annotations + H2 rules + auto-unlock)
# ---------------------------------------------------------------------------


class H4BankingKnowledgeAnnotationMixin:
    """Mixin that wires H2 rules, H4 annotators, and auto-unlock into
    banking_knowledge tools.

    Place before HarnessedToolKitMixin and the concrete toolkit class in MRO::

        class HarnessedKBSearch(H4BankingKnowledgeAnnotationMixin,
                                 HarnessedToolKitMixin,
                                 KnowledgeToolsWithKBSearch): ...

    Auto-unlock
    -----------
    After every KB_search call this mixin inspects the returned text for
    discoverable tool IDs (pattern ``<name>_\\d{4+}``).  Any valid tool found
    is immediately written into ``_agent_discoverable_tools_state``, so the
    agent can skip directly to ``call_discoverable_agent_tool`` without a
    separate unlock step.  This removes one reasoning hop that weak models
    consistently fail to take.
    """

    harness_rules: dict = {
        # LogVerificationTimeRule is intentionally NOT here: wrong timestamps are
        # silently corrected by the auto-fill in use_tool() below, avoiding the
        # correction-loop that disrupts weak model reasoning flow.
        #
        # CallDiscoverableUnlockedRule is intentionally NOT here: skipped unlocks
        # are silently corrected by the auto-unlock in use_tool() below (same
        # pattern as time_verified auto-fill). Only truly unknown tool names are
        # blocked via the existence check inside use_tool.
        "log_verification": [LogVerificationUserIdRule()],
        "unlock_discoverable_agent_tool": [UnlockDiscoverableExistsRule()],
    }
    harness_annotators: dict = {
        "KB_search": [KBSearchDiscoverableToolAnnotator()],
        "unlock_discoverable_agent_tool": [UnlockDiscoverableErrorAnnotator()],
        "call_discoverable_agent_tool": [
            CallDiscoverableErrorAnnotator(),
            CallDiscoverableSuccessAnnotator(),
        ],
        "give_discoverable_user_tool": [GiveDiscoverableErrorAnnotator()],
        "get_user_information_by_email": [EmailNotFoundAnnotator()],
        "get_user_information_by_name": [NameLookupEmailHintAnnotator()],
        "request_human_agent_transfer": [UserTransferRequestAnnotator()],
        "get_referrals_by_user": [ReferralsFoundAnnotator()],
    }

    def use_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Extend use_tool with auto-fill and auto-unlock.

        log_verification: auto-fill time_verified
        -----------------------------------------
        If the agent provides a wrong timestamp, silently overwrite it with the
        value from get_current_time() before the call reaches H2 / the tool.
        This avoids the correction-loop (block → get_current_time → retry) that
        disrupts weak-model reasoning flow, while still ensuring the correct
        simulation clock is used.

        call_discoverable_agent_tool: auto-unlock if skipped
        -----------------------------------------------------
        If the tool_id is valid but the agent forgot (or skipped) the unlock
        step, silently unlock it before executing — same pattern as the
        time_verified auto-fill.  This collapses the 3-step workflow
        (KB_search → unlock → call) down to 2 steps for the model.
        If the tool_id is completely unknown (hallucinated), return an error
        instead so the agent is redirected to KB_search.

        KB_search: auto-unlock discovered tool IDs
        ------------------------------------------
        After the parent pipeline (H2 checks → tool execution → H4 annotations)
        completes, inspect KB_search results for discoverable tool IDs and
        immediately unlock any that are valid and not yet unlocked.  The agent
        receives a note that it can proceed directly to call_discoverable_agent_tool.
        """
        # Auto-fill time_verified for log_verification.
        if tool_name == "log_verification":
            raw_time = self.get_current_time()  # type: ignore[attr-defined]
            m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+", raw_time)
            if m:
                kwargs["time_verified"] = m.group(0)

        # Auto-unlock for call_discoverable_agent_tool when the unlock was skipped.
        if tool_name == "call_discoverable_agent_tool":
            agent_tool_name = kwargs.get("agent_tool_name", "")
            state: dict = self._agent_discoverable_tools_state  # type: ignore[attr-defined]
            if agent_tool_name and agent_tool_name not in state:
                if self.has_discoverable_tool(agent_tool_name):  # type: ignore[attr-defined]
                    # Valid tool, just not unlocked yet — silently unlock.
                    from tau2.domains.banking_knowledge.tools import (
                        parse_discoverable_tool_docstring,
                    )
                    method = self.get_discoverable_tools()[agent_tool_name]  # type: ignore[attr-defined]
                    tool_info = parse_discoverable_tool_docstring(method)
                    state[agent_tool_name] = {
                        "unlocked_at": "auto-unlocked-by-harness",
                        "tool_info": tool_info,
                    }
                else:
                    # Unknown tool_id — return an error directly so the agent
                    # is redirected to KB_search rather than hitting a confusing
                    # downstream error.
                    return (
                        f"Error: Unknown tool '{agent_tool_name}'. "
                        "Tool IDs must come verbatim from a KB document — never guess them. "
                        "Use KB_search to find the procedure and its exact tool_id."
                    )

        result = super().use_tool(tool_name, **kwargs)  # type: ignore[misc]

        if tool_name != "KB_search":
            return result

        result_str = str(result)
        tool_ids = list(dict.fromkeys(_TOOL_ID_RE.findall(result_str)))
        if not tool_ids:
            return result

        auto_unlocked: list[str] = []
        for tid in tool_ids:
            if not self.has_discoverable_tool(tid):  # type: ignore[attr-defined]
                continue
            state: dict = self._agent_discoverable_tools_state  # type: ignore[attr-defined]
            if tid in state:
                continue  # already unlocked
            # Lazy import to avoid circular dependency at module load time.
            from tau2.domains.banking_knowledge.tools import (
                parse_discoverable_tool_docstring,
            )

            method = self.get_discoverable_tools()[tid]  # type: ignore[attr-defined]
            tool_info = parse_discoverable_tool_docstring(method)
            state[tid] = {"unlocked_at": "auto-unlocked-by-harness", "tool_info": tool_info}
            auto_unlocked.append(tid)

        if auto_unlocked:
            ids_str = ", ".join(f"'{t}'" for t in auto_unlocked)
            result = (
                result_str
                + f"\n\n[Harness: auto-unlocked {ids_str}. "
                f"You MUST now call: "
                f"call_discoverable_agent_tool(agent_tool_name='<one of the ids above>', "
                f"arguments='{{...}}'). "
                f"Do NOT claim the operation is complete until you have made this call "
                f"and received a success response. No separate unlock step needed.]"
            )

        return result
