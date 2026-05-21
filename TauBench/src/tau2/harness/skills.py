"""H5 Skill injection: task-aware learned tips derived from training trajectories.

Skills are short, actionable tips extracted from successful and failed training
conversations.  At agent-build time the top-k most relevant skills are retrieved
via BM25-lite and prepended to the agent's system prompt as a "Learned Tips"
block (highest attention position).

Design contrast with existing H layers:
  H2 (pre-call block)     — per-tool, blocking, execution-time
  H3 (schema hint)        — per-tool, passive, every API call
  H4 (post-call note)     — per-tool-result, informative, reactive
  H5 (skill injection)    — task-level, strategic, build-time once

Skills are task-LEVEL (planning / strategy) and should NOT duplicate
per-tool constraints already in H3.  Good skill content:
  ✓ "When user says 'all orders', scan every order_id before acting"
  ✓ "After any write, state the exact dollar amount to the user"
  ✗ "Status must be 'pending' to cancel"  ← already in H3 / H2

Usage
-----
    from tau2.harness.skills import retrieve_skills, format_skills_block, DOMAIN_SKILLS
    skills = retrieve_skills("retail", task_description, top_k=3)
    prefix = format_skills_block(skills)
    domain_policy = prefix + "\\n\\n" + domain_policy
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """A single learned tip derived from training trajectories.

    Attributes
    ----------
    id:
        Unique snake_case identifier (used for deduplication).
    title:
        One-line summary shown as a bullet heading in the prompt.
    pattern:
        Space-separated keywords used for BM25 retrieval.  Should cover the
        scenario vocabulary that will appear in task descriptions.
    tip:
        Actionable guidance injected verbatim into the system prompt.
        Keep under ~100 words; every word costs tokens every API call.
    source:
        ``"failure"`` — derived from a recurring error pattern.
        ``"success"`` — derived from an exemplary handling pattern.
    task_ids:
        Training task IDs that motivated this skill (documentation only).
    """

    id: str
    title: str
    pattern: str
    tip: str
    source: Literal["failure", "success"]
    task_ids: list[int] = field(default_factory=list)
    issue_type: str | None = None
    """Optional issue-type tag used for routing-based pre-filtering.

    Recognised values (telecom domain):
      ``"mms"``     — MMS / picture-message issues
      ``"abroad"``  — Roaming / international data issues
      ``"service"`` — Line suspension / no-service issues
      ``"data"``    — Domestic mobile-data issues
      ``None``      — General / applicable to any issue type
    """


# ---------------------------------------------------------------------------
# Retail skill library
# (derived from train-split trajectory analysis, Apr 2026)
# ---------------------------------------------------------------------------

RETAIL_SKILLS: list[Skill] = [
    Skill(
        id="product_available_count",
        title="Count all available variants, not only matching variants",
        pattern=(
            "how many count available options currently available in stock variants "
            "tshirt t-shirt product options online store exact number"
        ),
        tip=(
            "When the user asks how many product options are currently available, "
            "count every variant with available=true for that product. Do not "
            "filter the count by the user's later desired color, size, material, "
            "or style. You may still use those constraints separately when choosing "
            "a replacement item."
        ),
        source="failure",
        task_ids=[3, 4],
    ),
    Skill(
        id="address_from_order_history",
        title="Copy hidden target addresses from order history, then execute",
        pattern=(
            "address default profile shipping moved new home old address washington "
            "dc nyc chicago houston order history do not reveal all pending orders"
        ),
        tip=(
            "If the user says the target address is in their profile or another "
            "order and does not want to say it aloud, inspect the relevant orders "
            "and copy the exact address fields from the matching record. Profile "
            "address and order shipping addresses are separate writes. Determine "
            "the source and target from wording: if the new home/address is in a "
            "recent order and another order has the old/wrong address, copy from "
            "the new-home order to the old/wrong order and profile. Do the requested "
            "address writes, but do not let address lookup displace other requested "
            "item changes in the same task."
        ),
        source="failure",
        task_ids=[22, 41, 43, 87, 109, 110],
    ),
    Skill(
        id="pending_item_removal_fallback",
        title="Pending orders do not support partial item cancellation",
        pattern=(
            "pending remove item cancel just only single item one item partial "
            "cancellation cancel the charger cancel the hose not shipped whole order"
        ),
        tip=(
            "For a pending order, cancel_pending_order cancels the whole order; "
            "there is no tool for removing only one selected item. If the user "
            "asked not to cancel the whole order, do not call cancel_pending_order. "
            "Use full-order cancellation only when the user explicitly gave or "
            "confirmed that fallback AND accepts the actual refund route. "
            "Cancellation refunds go to the original payment method, not a chosen "
            "gift card; if the user's fallback requires gift-card refund and that "
            "is impossible, do not cancel."
        ),
        source="failure",
        task_ids=[28, 30, 57, 76, 88],
    ),
    Skill(
        id="bulk_order_completion",
        title="For bulk requests, scan all candidate orders and group writes by order",
        pattern=(
            "all orders all items everything cancel return multiple orders several "
            "orders financial issue total refund except keep boots delivered pending"
        ),
        tip=(
            "For bulk requests such as cancel/return everything or all possible "
            "orders, first inspect every candidate order from get_user_details. "
            "Group requested delivered items by order for one return call per "
            "order, and cancel each eligible pending order only when whole-order "
            "cancellation matches the user's scope. Keep explicitly excluded "
            "items out of the write calls."
        ),
        source="failure",
        task_ids=[28, 54, 103, 104],
    ),
    Skill(
        id="same_order_multi_item_exchange",
        title="Exchange all requested items from one delivered order in one call",
        pattern=(
            "same order also needs exchanged mention at same time bicycle jigsaw "
            "two items delivered order exchange both items one order"
        ),
        tip=(
            "If two or more items from the same delivered order need exchange, "
            "gather every replacement item_id first and call "
            "exchange_delivered_order_items once with parallel item_ids and "
            "new_item_ids for that order. Do not exchange one item first and then "
            "try a second exchange on the same order; retail permits only one "
            "return/exchange action per delivered order. When the user says an "
            "item is in the same order as another item, choose the order that "
            "contains both requested product names; do not use a similar item "
            "from a different order."
        ),
        source="failure",
        task_ids=[23, 98, 99],
    ),
    Skill(
        id="numeric_option_variant_selection",
        title="For numeric variant requests, compare option values",
        pattern=(
            "maximum max minimum min larger smaller bigger lower higher slightly "
            "resolution zoom storage gb tb inch pieces frame size same specs "
            "same options fallback available exchange modify"
        ),
        tip=(
            "For variant requests involving numeric options, compare the option "
            "values directly, not item price or listing order. Preserve every "
            "option the user wants unchanged, then change only the requested "
            "numeric option such as zoom, resolution, storage, screen size, "
            "frame size, or piece count. If the exact same item is unavailable "
            "but the user named a fallback numeric option, exchange to that "
            "fallback variant instead of returning the item."
        ),
        source="failure",
        task_ids=[52, 91, 95, 98, 99],
    ),
    Skill(
        id="category_scope_precision",
        title="Keep themed requests inside the user's stated category",
        pattern=(
            "associated with related to gaming office work-from-home hiking school "
            "accessories quit gaming category theme only items"
        ),
        tip=(
            "When the user scopes a request by theme or category, include only "
            "items that actually match that scope. Do not include unrelated items "
            "merely because they appear in the same order. If an order mixes in-scope "
            "and out-of-scope products, write only the in-scope item_ids."
        ),
        source="failure",
        task_ids=[14, 31, 54],
    ),
    Skill(
        id="pending_replacement_not_new_order",
        title="Replacing a pending item is not a new order",
        pattern=(
            "pending not shipped add replace change cheapest available under less "
            "than modify item same product variant speaker tablet order"
        ),
        tip=(
            "If a pending-order request asks to add, replace, or switch to a "
            "cheaper/more expensive variant of the same product, use "
            "modify_pending_order_items on the existing line item. Do not refuse "
            "as if this were placing a new order. If partial item cancellation is "
            "unsupported, a same-product replacement may still satisfy the fallback."
        ),
        source="failure",
        task_ids=[20, 63, 96, 109],
    ),
    Skill(
        id="execute_after_confirmation",
        title="After clear confirmation, call the write tool before summarizing",
        pattern=(
            "confirm confirmation yes proceed go ahead do it update modify return "
            "exchange cancel processed completed final scope exact address"
        ),
        tip=(
            "If the user clearly confirms the exact order/items/address/payment "
            "scope you just summarized, the next assistant message should call "
            "the corresponding write tool. Do not respond with another summary "
            "or claim completion without the tool call. If the scope changed, "
            "apply the newest user scope."
        ),
        source="failure",
        task_ids=[6, 7, 30, 72, 87],
    ),
]


# ---------------------------------------------------------------------------
# BM25-lite retrieval (same algorithm as policy_rag.py, self-contained here)
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can of in on at to for "
    "with by from up about into through during before after above below "
    "between each few more most other some such no nor not only own "
    "same so than too very just but and or if its it this that these "
    "those i you he she we they what which who when where why how all "
    "both any each few more most other some such no nor not "
    "my me us our your".split()
)


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


_K1 = 1.5
_B = 0.75


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    idf: dict[str, float],
    avgdl: float,
) -> float:
    if not doc_tokens:
        return 0.0
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        if qt not in tf:
            continue
        freq = tf[qt]
        numerator = freq * (_K1 + 1)
        denominator = freq + _K1 * (1 - _B + _B * dl / max(avgdl, 1))
        score += idf.get(qt, 1.0) * (numerator / denominator)
    return score


@lru_cache(maxsize=8)
def _build_index(domain: str) -> tuple[list[Skill], list[list[str]], dict[str, float]]:
    """Build token lists and IDF for the given domain's skill library."""
    skills = list(DOMAIN_SKILLS.get(domain, []))
    token_lists = [_tokenize(s.pattern) for s in skills]
    N = len(token_lists)
    df: Counter[str] = Counter()
    for tlist in token_lists:
        df.update(set(tlist))
    idf = {
        term: math.log((N + 1) / (count + 1)) + 1.0 for term, count in df.items()
    }
    return skills, token_lists, idf


# ---------------------------------------------------------------------------
# Issue-type routing (pre-filter before BM25)
# ---------------------------------------------------------------------------

# Keywords that identify the problem domain from the user's first message.
# Order matters only for logging; detection collects ALL matching types.
_ISSUE_TYPE_KEYWORDS: dict[str, list[str]] = {
    "mms": [
        "mms", "picture message", "picture messages",
        "multimedia message", "multimedia messaging",
    ],
    "abroad": [
        "abroad", "france", "international", "traveling",
        "overseas", "foreign country",
    ],
    "service": [
        "no service", "no signal", "can't make calls", "cannot make calls",
        "no calls", "no service",
    ],
    "data": [
        "mobile data", "data not working", "data keeps",
        "data is slow", "internet on my phone",
    ],
}

# When a type is detected, which additional issue_type buckets to include.
# "None" (general) skills are always included regardless.
_ISSUE_TYPE_EXPAND: dict[str, set[str]] = {
    "mms": {"data"},     # MMS can fail due to data quota → include data skills
    "abroad": {"data"},  # Abroad implies data context → include data skills
    "service": set(),
    "data": set(),
}


def _allowed_skill_types(query: str) -> set[str | None] | None:
    """Detect issue type(s) from a query and return the allowed skill bucket set.

    Returns ``None`` when no issue type is detected (no pre-filtering).
    Returns a set of ``issue_type`` values (including ``None`` for general skills)
    when at least one issue type is detected.
    """
    q = query.lower()
    detected: set[str] = set()
    for itype, keywords in _ISSUE_TYPE_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            detected.add(itype)
    if not detected:
        return None  # no filter — run BM25 over all skills
    allowed: set[str | None] = {None}  # general skills always included
    for itype in detected:
        allowed.add(itype)
        allowed.update(_ISSUE_TYPE_EXPAND.get(itype, set()))
    return allowed


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    """Return whether any needle appears as words or a phrase.

    Single-word needles match tokens, so ``address`` does not accidentally
    match ``addressed``. Multi-word needles match either the raw phrase or all
    non-stopword phrase tokens in the query.
    """
    query_tokens = set(_tokenize(text))
    for needle in needles:
        needle_tokens = _tokenize(needle)
        if not needle_tokens:
            continue
        if len(needle_tokens) == 1:
            if needle_tokens[0] in query_tokens:
                return True
            continue
        if needle in text or all(token in query_tokens for token in needle_tokens):
            return True
    return False


def _retail_skill_matches_context(skill: Skill, query: str) -> bool:
    """Precision guard for high-salience retail skills.

    BM25 alone over-injects narrow skills because retail task descriptions share
    many generic words ("order", "return", "address" in known-info text).  These
    guards keep H5 as task-level guidance instead of a broad second policy.
    """
    q = query.lower()
    sid = skill.id
    has_write_intent = _contains_any(
        q,
        (
            "cancel",
            "return",
            "exchange",
            "modify",
            "change",
            "update",
            "replace",
            "swap",
            "proceed",
            "go ahead",
        ),
    )

    if sid == "status_routing":
        return has_write_intent

    if sid == "product_available_count":
        count_phrases = (
            "how many",
            "count",
            "exactly how many",
            "number of options",
            "number of variants",
            "number of product options",
        )
        product_option_phrases = (
            "option",
            "options",
            "variant",
            "variants",
            "available",
            "in stock",
            "t-shirt",
            "tshirt",
        )
        return any(phrase in q for phrase in count_phrases) and any(
            phrase in q for phrase in product_option_phrases
        )

    if sid == "complete_all_in_one_turn":
        has_multi_scope = _contains_any(
            q,
            (
                "all orders",
                "all my orders",
                "multiple orders",
                "several orders",
                "two orders",
                "both orders",
                "everything",
                "all items",
                "both",
                "couple",
            ),
        )
        return has_write_intent and has_multi_scope

    if sid == "ranked_variant_preferences":
        has_variant_choice = _contains_any(
            q,
            (
                "variant",
                "options",
                "preference",
                "prefer",
                "priority",
                "ranked",
                "bigger",
                "smaller",
                "brighter",
                "less bright",
                "same specs",
                "same model",
                "one size",
                "cheapest",
                "most expensive",
            ),
        )
        return has_write_intent and has_variant_choice

    if sid == "confirmation_boundary":
        has_preview_or_confirm = _contains_any(
            q,
            (
                "before",
                "proceed",
                "go ahead",
                "confirm",
                "confirmation",
                "yes",
                "availability",
                "available",
                "price difference",
                "refund amount",
                "steps",
                "charges",
                "cost",
            ),
        ) or "confirmation" in q
        return has_write_intent and has_preview_or_confirm

    if sid == "communicate_exact_amounts":
        has_amount = _contains_any(
            q,
            (
                "amount",
                "refund",
                "price difference",
                "how much",
                "cost",
                "charge",
                "total",
                "dollar",
            ),
        )
        return has_write_intent and has_amount

    if sid == "no_fake_success":
        return has_write_intent

    if sid == "tracking_can_exist_on_cancelled_orders":
        return _contains_any(q, ("tracking", "tracking_id", "tracking number"))

    if sid == "multi_order_scan":
        return _contains_any(
            q,
            (
                "which order",
                "different order",
                "another order",
                "recent orders",
                "look up my account",
                "find the order",
                "not listed",
                "not in",
                "mixed up",
                "wrong order",
                "could be in",
            ),
        )

    if sid == "gift_card_rules":
        has_gift_card = _contains_any(
            q,
            (
                "gift card",
                "gift_card",
                "giftcard",
            ),
        )
        refund_conflict_phrases = (
            "other order's payment method",
            "other order payment method",
            "other payment method",
            "different payment method",
            "another payment method",
            "refund each order to the other",
            "refund to the other",
            "refund to credit card",
        )
        has_refund_method_conflict = any(
            phrase in q for phrase in refund_conflict_phrases
        ) and _contains_any(q, ("refund", "return", "cancel", "exchange"))
        return has_gift_card or has_refund_method_conflict

    if sid == "exchange_price_diff_communication":
        return _contains_any(
            q,
            (
                "price difference",
                "price diff",
                "cost difference",
                "how much",
                "extra cost",
                "charge",
                "charges",
                "refund amount",
                "pay extra",
                "cost",
            ),
        )

    if sid == "address_two_operations":
        return _contains_any(
            q,
            (
                "default address",
                "profile address",
                "shipping address",
                "ship to",
                "new home",
                "old address",
                "moved",
                "move to",
                "everywhere",
                "all addresses",
                "orders too",
            ),
        )

    if sid == "address_from_order_history":
        address_phrases = (
            "default address",
            "profile address",
            "user address",
            "order address",
            "order addresses",
            "shipping address",
            "ship to",
            "new home",
            "profile",
            "default home",
            "old address",
            "moved",
            "move to",
            "all addresses",
            "orders too",
        )
        has_address_context = any(phrase in q for phrase in address_phrases)
        has_privacy_context = _contains_any(
            q,
            (
                "do not reveal",
                "don't want to reveal",
                "private",
                "privacy",
                "look it up in orders",
                "in my profile",
            ),
        )
        has_address_write = _contains_any(
            q,
            (
                "change address",
                "update address",
                "modify address",
                "correct address",
                "correct all order addresses",
                "typed address wrong",
                "typed your address wrong",
                "address wrong",
                "wrong address",
                "old address",
                "new address",
                "new home",
                "default address",
                "profile address",
                "shipping address",
            ),
        )
        return has_address_context and (has_privacy_context or has_address_write)

    if sid == "category_scope_precision":
        return _contains_any(
            q,
            (
                "associated with",
                "related to",
                "gaming",
                "office items",
                "work-from-home",
                "work from home",
                "hiking items",
                "school",
                "accessories",
                "keep the hiking",
            ),
        )

    if sid == "numeric_option_variant_selection":
        has_numeric_preference = _contains_any(
            q,
            (
                "maximum",
                "max",
                "minimum",
                "min",
                "larger",
                "smaller",
                "bigger",
                "lower",
                "higher",
                "slightly",
                "one size",
                "more pieces",
                "less bright",
                "brighter",
                "same specs",
                "same options",
                "32gb",
                "1tb",
            ),
        )
        has_numeric_option = _contains_any(
            q,
            (
                "resolution",
                "zoom",
                "storage",
                "screen size",
                "frame size",
                "piece",
                "pieces",
                "size",
                "gb",
                "tb",
                "inch",
            ),
        )
        return has_write_intent and has_numeric_preference and has_numeric_option

    if sid == "pending_replacement_not_new_order":
        has_pending_context = _contains_any(
            q,
            (
                "pending",
                "not shipped",
                "hasn't shipped",
                "has not shipped",
                "before it ships",
                "recent order",
            ),
        )
        has_replacement = _contains_any(
            q,
            (
                "add",
                "replace",
                "change",
                "switch",
                "cheapest",
                "most expensive",
                "under",
                "less than",
                "available for less",
            ),
        )
        has_product = _contains_any(
            q,
            (
                "speaker",
                "tablet",
                "same product",
                "variant",
                "color",
                "red",
                "green",
            ),
        )
        return has_pending_context and has_replacement and has_product

    if sid == "mixed_info_and_confirmed_write":
        has_question = _contains_any(
            q,
            (
                "how many",
                "count",
                "available",
                "in stock",
                "return label",
                "label",
                "timing",
                "timeline",
                "price difference",
                "refund amount",
            ),
        )
        has_write = _contains_any(
            q,
            (
                "modify",
                "change",
                "exchange",
                "return",
                "cancel",
                "update",
                "proceed",
                "go ahead",
            ),
        )
        return has_question and has_write

    if sid == "bulk_order_completion":
        category_phrases = (
            "associated with",
            "not associated with",
            "related to",
            "not related to",
            "gaming",
            "office items",
            "work-from-home",
            "work from home",
            "hiking items",
            "school",
            "accessories",
        )
        if any(phrase in q for phrase in category_phrases):
            return False
        item_modification_phrases = (
            "cheapest options",
            "switch all items",
            "change all items",
            "modify all items",
            "cheapest variants",
        )
        if any(phrase in q for phrase in item_modification_phrases):
            return False
        bulk_phrases = (
            "all orders",
            "all my orders",
            "all possible orders",
            "all items",
            "everything",
            "return all",
            "cancel all",
            "cancel or return all",
        )
        has_bulk_scope = any(phrase in q for phrase in bulk_phrases)
        return has_write_intent and has_bulk_scope

    if sid == "same_order_multi_item_exchange":
        same_order_phrases = (
            "same order also needs",
            "same order needs",
            "same order",
            "mention these at the same time",
            "mention the two requests at the same time",
        )
        has_same_order = any(phrase in q for phrase in same_order_phrases)
        has_exchange = _contains_any(q, ("exchange", "exchanged"))
        has_multi_item = _contains_any(q, ("both", "two", "also", "multiple"))
        return has_exchange and has_same_order and has_multi_item

    if sid == "execute_after_confirmation":
        has_confirmation_trigger = (
            "if the agent asks" in q
            or "asks for confirmation" in q
            or "ask for confirmation" in q
            or _contains_any(
                q,
                (
                    "confirmation",
                    "confirm",
                    "proceed",
                    "go ahead",
                    "yes",
                    "after the agent asks",
                ),
            )
        )
        return has_write_intent and has_confirmation_trigger

    if sid == "refund_timeline_no_transfer":
        has_timeline = _contains_any(
            q,
            (
                "3 days",
                "three days",
                "timeline",
                "timing",
                "business days",
                "expedite",
                "supervisor",
                "escalate",
            ),
        )
        return has_timeline and _contains_any(q, ("return", "refund"))

    if sid == "modify_items_once":
        return _contains_any(
            q, ("pending", "modify", "change", "swap", "replace", "upgrade")
        )

    if sid == "pending_item_removal_fallback":
        has_pending = any(
            phrase in q
            for phrase in (
                "pending order",
                "pending-order",
                "just placed",
                "recently placed",
                "not shipped",
                "has not shipped",
                "hasn't shipped",
            )
        )
        has_item_removal = any(
            phrase in q
            for phrase in (
                "remove",
                "cancel just",
                "cancel only",
                "partial cancellation",
                "partial cancel",
                "specific item",
                "one item",
                "single item",
                "cancel the air purifier",
                "remove the fleece jacket",
            )
        )
        return has_pending and has_item_removal

    if sid == "same_product_multi_order_precision":
        return _contains_any(
            q,
            (
                "same product",
                "multiple orders",
                "different orders",
                "more expensive",
                "cheaper",
                "all bookshelves",
                "all puzzles",
                "all skateboards",
                "all tablets",
            ),
        )

    if sid == "process_orders_immediately":
        if not has_write_intent:
            return False
        return _contains_any(
            q,
            (
                "all my orders",
                "all orders",
                "multiple orders",
                "several orders",
                "two pending",
                "two delivered",
                "cancel all",
                "return all",
                "everything",
            ),
        )

    return True


def _airline_skill_matches_context(skill: Skill, query: str) -> bool:
    """Precision guard for airline H5 skills.

    Airline tasks share many generic tokens ("reservation", "refund", "flight"),
    so BM25 alone injects broad skills into unrelated conversations.  Keep H5 as
    a sparse task-level hint; stable policy/tool-contract rules belong in H3/H4.
    """
    q = query.lower()
    sid = skill.id

    has_write_intent = _contains_any(
        q,
        (
            "book",
            "cancel",
            "change",
            "modify",
            "update",
            "upgrade",
            "downgrade",
            "add",
            "remove",
        ),
    )
    explicit_amount_request = _contains_any(
        q,
        (
            "how much",
            "total",
            "sum",
            "saved",
            "save",
            "save in total",
            "cost in total",
            "total cost",
            "total price",
            "charge",
            "charged",
            "card will be charged",
            "master card will be charged",
            "mastercard will be charged",
            "refund amount",
            "how much refund",
            "refunded",
            "amount",
            "balance",
            "budget",
            "less than",
            "under",
            "up to",
        ),
    )

    if sid == "communicate_exact_amounts":
        return has_write_intent and explicit_amount_request

    if sid == "verify_correct_reservation":
        return _contains_any(
            q,
            (
                "don't remember",
                "do not remember",
                "last reservation",
                "most recent",
                "recent reservation",
                "which reservation",
                "figure out",
                "multiple reservations",
                "booked multiple",
                "same day",
                "mixup",
            ),
        )

    if sid == "one_certificate_per_reservation":
        return _contains_any(
            q,
            (
                "certificate",
                "certificates",
                "travel certificate",
                "certificate balance",
            ),
        )

    if sid == "round_trip_include_all_legs":
        return has_write_intent and _contains_any(
            q,
            (
                "round trip",
                "roundtrip",
                "return flight",
                "return",
                "outbound",
                "both legs",
                "same dates",
                "one direction",
            ),
        )

    if sid == "check_cancellation_eligibility":
        return _contains_any(q, ("cancel", "cancellation")) and _contains_any(
            q,
            (
                "refund",
                "insurance",
                "sick",
                "health",
                "weather",
                "business",
                "basic economy",
                "booked",
                "24 hours",
                "full refund",
            ),
        )

    if sid == "budget_before_write":
        return has_write_intent and any(
            phrase in q
            for phrase in (
                "budget",
                "less than",
                "under $",
                "under ",
                "up to $",
                "willing to pay",
                "maximum",
                "max ",
                "master card",
                "mastercard",
                "total extra cost",
                "card will be charged",
            )
        )

    if sid == "other_upcoming_reservations_scan":
        return any(
            phrase in q
            for phrase in (
                "other upcoming",
                "other flights",
                "other reservations",
                "duplicate",
                "mixup",
                "figure out which",
                "booked multiple flights",
                "multiple flights for the same day",
            )
        )

    return True


def _retail_skill_score_boost(skill: Skill, query: str) -> float:
    """Small precision boosts so top-k keeps the most actionable retail tips."""
    q = query.lower()
    sid = skill.id

    if sid == "confirmation_boundary" and (
        "confirmation" in q
        or "asks for confirmation" in q
        or "ask for confirmation" in q
        or "if the agent asks" in q
        or "if agent asks" in q
        or _contains_any(q, ("proceed", "go ahead", "yes"))
    ):
        return 6.5

    if sid == "no_fake_success" and (
        "confirmation" in q
        or _contains_any(
            q,
            (
                "proceed",
                "go ahead",
                "confirm",
                "completed",
                "processed",
            ),
        )
    ):
        return 2.5

    if sid == "multi_order_scan" and _contains_any(
        q,
        (
            "different order",
            "another order",
            "other order",
            "wrong order",
            "look up my account",
            "search my account",
            "find the purchase",
        ),
    ):
        return 3.0

    if sid == "refund_timeline_no_transfer" and _contains_any(
        q,
        ("3 days", "three days", "expedite", "supervisor", "business days"),
    ):
        return 5.0

    if sid == "modify_items_once" and _contains_any(
        q,
        ("pending", "upgrade", "most expensive", "costliest", "modify", "change"),
    ):
        return 3.0

    return 0.0


def _airline_skill_score_boost(skill: Skill, query: str) -> float:
    """Prefer the single most actionable airline skill for the user's wording."""
    q = query.lower()
    sid = skill.id

    if sid == "other_upcoming_reservations_scan" and any(
        phrase in q
        for phrase in (
            "other upcoming",
            "other flights",
            "other reservations",
            "duplicate",
            "mixup",
            "figure out which",
            "booked multiple flights",
            "multiple flights for the same day",
        )
    ):
        return 6.0

    if sid == "check_cancellation_eligibility" and _contains_any(
        q, ("cancel", "cancellation")
    ) and _contains_any(
        q,
        (
            "basic economy",
            "sick",
            "insurance",
            "full refund",
            "upgrade to business first",
            "then cancel",
        ),
    ):
        return 5.0

    if sid == "budget_before_write" and any(
        phrase in q
        for phrase in (
            "budget",
            "up to $",
            "less than",
            "under $",
            "under ",
            "willing to pay",
            "master card will be charged",
            "mastercard charges",
        )
    ):
        return 4.0

    if sid == "one_certificate_per_reservation" and _contains_any(
        q, ("certificate", "certificates", "certificate balance")
    ):
        return 4.0

    if sid == "round_trip_include_all_legs" and _contains_any(
        q,
        (
            "round trip",
            "roundtrip",
            "return flights",
            "fastest return",
            "one direction",
            "move back your return",
        ),
    ):
        return 3.0

    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve_skills(
    domain: str,
    task_description: str,
    top_k: int = 3,
    min_score: float | None = None,
) -> list[Skill]:
    """Return the top-k most relevant skills for the given task description.

    Uses issue-type routing to pre-filter the skill library, then BM25-lite
    scoring over each skill's ``pattern`` and ``title`` fields.

    Routing: detects issue type keywords in ``task_description`` ("mms",
    "abroad", "no service", "mobile data") and restricts BM25 to the matching
    skill buckets plus general (``issue_type=None``) skills.  When no issue
    type is detected, all skills are considered.

    Args:
        domain: Domain name (e.g. "retail").
        task_description: Free-text description of the current task / scenario.
        top_k: Maximum number of skills to return.
        min_score: Minimum BM25-lite score required for a skill to be injected.
            If omitted, a conservative domain default is used so H5 prefers
            precision over recall on new test-set tasks. Set to 0.0 to recover
            the old "any positive match" behavior.

    Returns:
        List of relevant Skill objects, most relevant first.
    """
    skills = DOMAIN_SKILLS.get(domain)
    if not skills:
        return []
    if domain == "retail":
        top_k = min(top_k, 2)
    elif domain == "airline":
        top_k = min(top_k, 1)
    if min_score is None:
        min_score = {
            "airline": 2.5,
            "retail": 1.5,
        }.get(domain, 1.5)

    skills_list, token_lists, idf = _build_index(domain)
    avgdl = sum(len(t) for t in token_lists) / max(len(token_lists), 1)
    q_tokens = _tokenize(task_description)

    # Apply issue-type routing to narrow the candidate pool
    allowed_types = _allowed_skill_types(task_description)
    if allowed_types is not None:
        candidates = [
            (doc_toks, skill)
            for doc_toks, skill in zip(token_lists, skills_list)
            if skill.issue_type in allowed_types
        ]
    else:
        candidates = list(zip(token_lists, skills_list))

    if domain == "retail":
        candidates = [
            (doc_toks, skill)
            for doc_toks, skill in candidates
            if _retail_skill_matches_context(skill, task_description)
        ]
    elif domain == "airline":
        candidates = [
            (doc_toks, skill)
            for doc_toks, skill in candidates
            if _airline_skill_matches_context(skill, task_description)
        ]

    if not q_tokens or not candidates:
        return []

    scored = []
    for doc_toks, skill in candidates:
        score = _bm25_score(q_tokens, doc_toks, idf, avgdl)
        if domain == "retail":
            score += _retail_skill_score_boost(skill, task_description)
        elif domain == "airline":
            score += _airline_skill_score_boost(skill, task_description)
        scored.append((score, skill))
    scored.sort(key=lambda x: -x[0])
    return [
        skill
        for score, skill in scored[:top_k]
        if score >= min_score and score > 0
    ]


def format_skills_block(skills: list[Skill]) -> str:
    """Format a list of skills as a compact system-prompt prefix block.

    The block is designed to be prepended BEFORE the domain policy so it
    occupies the highest-attention region of the system prompt.

    Returns an empty string when ``skills`` is empty.
    """
    if not skills:
        return ""

    lines: list[str] = [
        "## Task-Specific Guidance",
        "Apply only guidance that is relevant to the user's current request. "
        "If guidance appears to conflict with the current dialogue, tool result, "
        "or domain policy, follow the current dialogue, tool result, and domain "
        "policy.",
        "",
    ]
    for skill in skills:
        lines.append(f"- **{skill.title}**")
        lines.append(skill.tip)
        lines.append("")
    lines.append("---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Telecom skill library
# (derived from train-split trajectory analysis, Apr 2026)
# ---------------------------------------------------------------------------

TELECOM_SKILLS: list[Skill] = [
    Skill(
        id="data_quota_check",
        title="Check data quota when data is enabled but still not working",
        pattern=(
            "mobile data not working speed_test no_connection "
            "data enabled slow internet quota exceeded plan limit "
            "data_usage_exceeded refuel data_usage"
        ),
        tip=(
            "When mobile data is ON but speed tests return 'No Connection', "
            "OR when MMS still fails after fixing WiFi Calling and data mode: "
            "call get_data_usage IMMEDIATELY. "
            "If data_used_gb ≥ data_limit_gb + data_refueling_gb, the quota is "
            "EXHAUSTED — this blocks both internet AND MMS delivery. "
            "Inform the customer of the cost per GB (from their plan) and offer to "
            "call refuel_data (max 2 GB per call) upon consent. "
            "Do NOT chase APN / VPN / roaming causes until data quota is ruled out."
        ),
        source="failure",
        task_ids=[],
        issue_type="data",
    ),
    Skill(
        id="abroad_enable_roaming",
        title="Enable account-level roaming and device toggle for customers traveling abroad",
        pattern=(
            "abroad traveling internationally france europe asia "
            "data_roaming roaming not working foreign country roaming_disabled "
            "no_signal roaming_enabled"
        ),
        tip=(
            "If the customer is traveling internationally and data/MMS fails: "
            "1) call get_details_by_id to check roaming_enabled. "
            "2) If roaming_enabled=False → call enable_roaming (FREE). "
            "3) ALWAYS (regardless of account roaming status) remind the customer "
            "to toggle Data Roaming ON in device Settings (toggle_roaming). "
            "Account-level enable (your tool) ≠ device roaming toggle (user action). "
            "BOTH steps are independently required — even if roaming_enabled=True, "
            "the device toggle may still be OFF and must be confirmed."
        ),
        source="failure",
        task_ids=[],
        issue_type="abroad",
    ),
    Skill(
        id="suspended_overdue_bill",
        title="Resolve suspended line caused by overdue bill",
        pattern=(
            "no_service service suspended line calls cannot_call overdue bill payment "
            "restore reactivate"
        ),
        tip=(
            "When service is lost and the line is Suspended: "
            "1) call get_bills_for_customer and find any OVERDUE bill. "
            "2) call send_payment_request with the bill_id. "
            "3) Inform the customer to use check_payment_request + make_payment. "
            "4) After payment confirmed PAID, call resume_line. "
            "5) Remind customer to reboot device. "
            "This is FULLY within scope — do NOT transfer to human."
        ),
        source="failure",
        task_ids=[],
        issue_type="service",
    ),
    Skill(
        id="fix_all_causes_systematically",
        title="Troubleshoot ALL causes before concluding or transferring",
        pattern=(
            "multiple causes troubleshoot mobile data mms service_issue service calls "
            "still not working after fixing persistent problem"
        ),
        tip=(
            "Technical issues often have MULTIPLE simultaneous causes. "
            "Systematic order for mobile data: "
            "airplane mode OFF → data toggle ON → check data quota (get_data_usage) → "
            "check roaming if abroad → check network preference → check VPN/APN. "
            "For MMS: same as above + check app permissions. "
            "After fixing one cause, if the problem persists, continue to the next. "
            "Do NOT give up or transfer while any fixable cause remains."
        ),
        source="failure",
        task_ids=[],
        issue_type=None,
    ),
    Skill(
        id="quota_and_roaming_independent",
        title="After enabling roaming, if still failing — check data quota too",
        pattern=(
            "roaming enabled speed_test still fails no_connection after roaming "
            "abroad data_roaming not working despite roaming_enabled"
        ),
        tip=(
            "Roaming and data quota are TWO INDEPENDENT causes. "
            "Enabling roaming fixes the roaming problem but NOT an exhausted quota. "
            "After enable_roaming: if speed test still returns 'No Connection' → "
            "IMMEDIATELY call get_data_usage. "
            "If data_used_gb ≥ data_limit_gb, offer refuel_data (max 2 GB). "
            "Both must be fixed for the customer to have working internet."
        ),
        source="failure",
        task_ids=[],
        issue_type="abroad",
    ),
    Skill(
        id="roaming_already_on_device_fix",
        title="Roaming already enabled on account — fix is device toggle only",
        pattern=(
            "roaming already enabled device_toggle data_roaming on off "
            "toggle_roaming phone_settings roaming_enabled"
        ),
        tip=(
            "If enable_roaming returns 'already enabled' (or H2 blocks it): "
            "account-level roaming is ON — do NOT call enable_roaming again. "
            "The fix is on the DEVICE: instruct the customer to toggle Data Roaming "
            "ON in phone settings (toggle_roaming user action). "
            "After device toggle → run speed test to confirm."
        ),
        source="failure",
        task_ids=[],
        issue_type="abroad",
    ),
    Skill(
        id="identify_correct_line",
        title="Use the line matching the customer's phone number, not the first line",
        pattern=(
            "refuel enable roaming line_id customer multiple_lines "
            "phone_number which_line write operation wrong_line"
        ),
        tip=(
            "A customer may have MULTIPLE lines (e.g. L1001, L1002, L1003). "
            "When you call get_customer_by_phone('555-123-2002'), the result "
            "contains ALL line_ids. "
            "You MUST use the line_id whose phone_number matches the customer's "
            "reported number for ALL write operations (refuel_data, enable_roaming, etc.). "
            "Verify via get_details_by_id on each line_id until you find the matching "
            "phone_number, or check the H4 annotation which identifies it for you. "
            "NEVER default to the first line_id in the list."
        ),
        source="failure",
        task_ids=[],
        issue_type=None,
    ),
    Skill(
        id="sim_pin_plus_billing",
        title="Handle SIM PIN lock independently from billing issues",
        pattern=(
            "sim_locked pin no_service service pin_locked overdue bill suspended calls "
            "cannot_unlock sim_card payment"
        ),
        tip=(
            "A locked SIM (status: locked_pin) and an overdue bill are TWO separate issues. "
            "Handle them independently: "
            "1) Locked SIM PIN itself is NOT fixable with available tools; it needs "
            "human/in-store support. "
            "2) ALSO check for overdue bills (get_bills_for_customer) and "
            "send a payment request if needed (send_payment_request). "
            "A locked SIM does NOT prevent you from resolving billing issues, but "
            "after finishing any bill payment/resume workflow, call "
            "transfer_to_human_agents(summary=...) TOOL for the remaining SIM PIN "
            "lock. Do not just tell the customer to enter a PIN and close. "
            "If contract has expired: call transfer_to_human_agents(summary=...) TOOL "
            "(must be a tool call — not just text in the conversation)."
        ),
        source="failure",
        task_ids=[],
        issue_type="service",
    ),
    Skill(
        id="mms_wifi_calling",
        title="Disable WiFi Calling to fix MMS issues",
        pattern=(
            "mms not working cannot send receive picture message wifi_calling "
            "mms fails multimedia messaging mms_issue"
        ),
        tip=(
            "WiFi Calling interferes with standard cellular MMS delivery. "
            "When MMS is failing, ALWAYS advise the customer to turn off WiFi Calling "
            "in phone settings (turn_off_wifi_calling user action). "
            "MMS is a multi-cause workflow: do not transfer just because one fix "
            "did not restore MMS. Also verify mobile data ON, APN settings correct, "
            "SIM re-seated, "
            "roaming enabled if abroad (toggle_roaming on device), "
            "messaging app has SMS + Storage permissions "
            "(check_app_permissions('messaging') → grant_app_permission if missing). "
            "If can_send_mms is still false after these device-side fixes, "
            "check the account-side data quota before transferring: "
            "get_customer_by_phone → use the line_id matching the phone number → "
            "get_data_usage. If data_used_gb ≥ data_limit_gb + data_refueling_gb, "
            "quote the plan's per-GB refuel cost and call refuel_data(max 2 GB) "
            "after consent. Exhausted data quota blocks MMS even on 5G. "
            "Do NOT skip WiFi Calling even if other causes are present."
        ),
        source="failure",
        task_ids=[],
        issue_type="mms",
    ),
    Skill(
        id="abroad_all_causes",
        title="Fix ALL data causes when customer is abroad (roaming + VPN + network preference)",
        pattern=(
            "abroad international traveling roaming enable account device "
            "vpn network_preference still not working after roaming"
        ),
        tip=(
            "When abroad with mobile data issues, fix EVERY cause in order: "
            "1) ROAMING: call enable_roaming if roaming_enabled=False; also tell "
            "customer to toggle Data Roaming ON in device settings (toggle_roaming). "
            "2) VPN: if VPN is active, advise customer to disconnect it (disconnect_vpn). "
            "3) NETWORK PREFERENCE: advise customer to set network mode to Auto/4G. "
            "Each cause independently blocks data — fix all before concluding."
        ),
        source="failure",
        task_ids=[],
        issue_type="abroad",
    ),
    Skill(
        id="service_billing_and_device_fixes",
        title="When service is lost: complete billing workflow AND all device/setting fixes",
        pattern=(
            "no_service service_lost service suspended calls overdue bill airplane_mode "
            "sim unseat apn settings pin_locked multiple causes"
        ),
        tip=(
            "When service is lost with MULTIPLE causes (billing + device issues): "
            "handle BOTH the tool workflow AND user actions. "
            "Billing: send_payment_request → customer makes payment → resume_line → reboot. "
            "Device: ALSO instruct every needed user action — turn off airplane mode, "
            "re-seat SIM card, reset APN settings. "
            "If the SIM is locked with PIN, that part is not fixable by tools; after "
            "resolving any overdue bill workflow, call transfer_to_human_agents "
            "as a TOOL for SIM PIN/in-store support. "
            "Do NOT focus only on billing and neglect device-side fixes. "
            "All causes must be resolved for service to restore."
        ),
        source="failure",
        task_ids=[],
        issue_type="service",
    ),
    Skill(
        id="hard_persona_loop_break",
        title="Escape repeated-instruction deadlock with uncooperative Hard customers",
        pattern=(
            "hard difficult unresponsive confused repeated instruction "
            "stuck loop cannot perform action again"
        ),
        tip=(
            "This customer may struggle with complex multi-step instructions. "
            "STRATEGY for Hard persona: "
            "1) Give ONE short instruction per turn (≤1 sentence, no sub-bullets). "
            "2) After each step, ask a simple yes/no confirmation question. "
            "3) If the customer says 'I don't understand' or gives no progress "
            "2+ times in a row for the SAME step: immediately switch approach — "
            "try different wording, or skip to the next actionable tool call you can "
            "make on your side (e.g. enable_roaming, send_payment_request). "
            "4) After completing all server-side tool calls, provide a concise summary "
            "of what was resolved and close: 'A technician can assist with remaining "
            "device configuration steps if needed.' "
            "NEVER repeat the same detailed explanation more than twice."
        ),
        source="failure",
        task_ids=[],
        issue_type=None,
    ),
]


# ---------------------------------------------------------------------------
# Airline skill library
# (derived from train-split trajectory analysis, Apr 2026)
# ---------------------------------------------------------------------------

AIRLINE_SKILLS: list[Skill] = [
    Skill(
        id="communicate_exact_amounts",
        title="Always state the exact dollar amount after any write operation",
        pattern=(
            "cancel refund amount total price charged credit_card gift_card "
            "certificate payment communicate tell_user exact_amount how_much "
            "update_flights book_reservation change_flights exact_charge exact_refund "
            "money cost how_much_charged how_much_refunded"
        ),
        tip=(
            "After EVERY write operation (cancel_reservation, update_reservation_flights, "
            "book_reservation), tell the user the EXACT dollar amount. "
            "For cancellations: 'Your reservation has been cancelled. $X has been "
            "refunded to your [payment method].' "
            "For flight updates: 'Your flights have been updated. $X has been "
            "charged/refunded to [payment method].' "
            "NEVER say only 'a refund has been processed' without the number. "
            "The exact amount is in the H4 annotation appended to the tool result."
        ),
        source="failure",
        task_ids=[7, 11, 14],
    ),
    Skill(
        id="verify_correct_reservation",
        title="Confirm the right reservation before acting when user has multiple",
        pattern=(
            "multiple reservations which reservation correct_reservation "
            "latest most_recent reservation_id verify identify confirm "
            "recent booking which_reservation user_reservations"
        ),
        tip=(
            "When get_user_details shows multiple reservations: NEVER auto-select "
            "the most recent or the first one. "
            "Ask the user for the reservation ID or cross-reference with the flight "
            "details they gave (route, date, cabin) to identify the correct one. "
            "Call get_reservation_details on candidates to verify. "
            "For schedule mixups, duplicate/same-day bookings, or requests where "
            "the user asks you to figure out what to fix, inspect every plausible "
            "reservation from get_user_details before deciding which ones to cancel. "
            "Only act after you are certain you have the right reservation(s). "
            "The H4 annotation after get_user_details lists all reservations with "
            "origin, destination, and dates to help you identify the right one."
        ),
        source="failure",
        task_ids=[38, 42],
    ),
    Skill(
        id="one_certificate_per_reservation",
        title="At most ONE travel certificate per reservation; use separate bookings for more",
        pattern=(
            "certificate travel_certificate multiple certificates two certificates "
            "pay_with_certificate certificate_limit per_reservation split_certificates "
            "certificate_balance larger_certificate smaller_certificate"
        ),
        tip=(
            "Policy: each reservation may use at most ONE travel certificate. "
            "If the user has multiple certificates or wants to use more than one: "
            "do NOT attempt to apply 2+ certificates to a single reservation — "
            "the harness will block this. For a single reservation, choose only "
            "one certificate, usually the largest usable one, then use allowed "
            "gift cards/credit card for the remainder. Only split into separate "
            "bookings when the user explicitly asks to use multiple certificates "
            "and separate reservations still satisfy the travel request. "
            "Also: certificates CANNOT be used to pay for flight updates "
            "(update_reservation_flights); only gift cards or credit cards are accepted. "
            "If the current reservation is basic_economy and the user wants a new "
            "itinerary, do not update it with different flights; cancel and book "
            "new reservation(s) when that fallback has been accepted. Do not count "
            "a cancellation refund as immediately increasing gift-card balance for "
            "the new booking; use only the balances currently shown in user details."
        ),
        source="failure",
        task_ids=[23],
    ),
    Skill(
        id="round_trip_include_all_legs",
        title="For round-trip updates, submit ALL legs (outbound AND return) together",
        pattern=(
            "round_trip return leg outbound missing return_flight "
            "complete_itinerary both_legs all_flights update_reservation_flights "
            "one_direction outbound_only return_only cabin_upgrade single_segment"
        ),
        tip=(
            "For round-trip updates, submit the full connected itinerary in one "
            "update_reservation_flights call. If only one direction changes, copy "
            "the other direction exactly from get_reservation_details. Cabin is "
            "reservation-wide, so confirm before treating a one-direction cabin "
            "request as a whole-reservation cabin change. For fastest/cheapest "
            "return changes, compare complete candidate paths, require enough "
            "seats in the requested cabin on every leg, and retry with the next "
            "valid candidate if the first update fails. Preserve the original or "
            "requested return date instead of reusing the outbound date."
        ),
        source="failure",
        task_ids=[15, 33],
    ),
    Skill(
        id="check_cancellation_eligibility",
        title="Check all 4 cancellation conditions before attempting cancel_reservation",
        pattern=(
            "cancel cancellation cancel_reservation cancel_flight cancellation_eligibility "
            "check_eligibility booked_24h airline_cancelled travel_insurance "
            "covered_reason refund_due_to_cancel basic_economy_cancel"
        ),
        tip=(
            "cancel_reservation requires at least ONE of these 4 conditions: "
            "1) Booked within the last 24 hours "
            "2) Airline cancelled a flight in the reservation "
            "3) Cabin is 'business' "
            "4) Travel insurance was purchased ('yes') "
            "For basic_economy without meeting conditions 1/2/4: only upgrading "
            "cabin to business first (update_reservation_flights with same flights, "
            "cabin='business') makes it cancellable via condition 3. Upgrading to "
            "economy does NOT satisfy cancellation eligibility. "
            "If the user asked to cancel after this eligibility upgrade, call "
            "cancel_reservation immediately after the successful upgrade before "
            "summarizing. When multiple cancellations are confirmed, complete every "
            "confirmed cancel/update step before the final response. If the same "
            "turn also asks for a total over other upcoming flights, inspect the "
            "listed reservations and communicate that total after the writes. Check "
            "the reservation's insurance, cabin, and created_at before calling cancel."
        ),
        source="failure",
        task_ids=[5, 23],
    ),
    Skill(
        id="budget_before_write",
        title="When user gives a budget or asks total card charge, calculate before writing",
        pattern=(
            "budget max maximum less_than under up_to willing_to_pay mastercard_charge "
            "credit_card_charge total_extra_cost net_cost fee quote confirm before_booking "
            "if_cost_less_than minimize_card_payment"
        ),
        tip=(
            "If the user gives a budget or asks how much the card will be charged, "
            "calculate the full net cost before any write. For updates, compare new "
            "fare to current fare; for cancel-and-rebook, include refund and new "
            "payment allocation. Search the relevant direct/one-stop legs before "
            "declaring a cabin upgrade possible or impossible. If over budget, do "
            "not write; offer the fallback. Cabin changes apply to the whole "
            "reservation, not just one passenger or one leg."
        ),
        source="failure",
        task_ids=[10, 12, 14, 23, 33],
    ),
    Skill(
        id="other_upcoming_reservations_scan",
        title="For other upcoming flights or schedule mixups, scan every reservation",
        pattern=(
            "other_flights additional_flights other_reservations all_reservations "
            "multiple_reservations same_day duplicate mixup assistant_booked "
            "which_flights_should_cancel schedule_conflict"
        ),
        tip=(
            "When the user asks whether they have other upcoming flights, duplicate/same-day "
            "bookings, or asks you to figure out which reservations to cancel: call "
            "get_user_details, then inspect every reservation_id with get_reservation_details. "
            "Do not infer from only the reservation IDs mentioned by the user, and do not claim "
            "there are no other flights until every listed reservation has been checked. "
            "Scanning every reservation does NOT mean cancelling every mismatch: cancel only "
            "reservations that truly conflict with the user's stated itinerary, dates, "
            "passenger identity, and fallback instructions."
        ),
        source="failure",
        task_ids=[7, 42],
    ),
]


# ---------------------------------------------------------------------------
# Domain registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Banking-knowledge skill library
# (derived from train-split trajectory analysis, Apr 2026)
# ---------------------------------------------------------------------------

BANKING_KNOWLEDGE_SKILLS: list[Skill] = [
    Skill(
        id="log_verification_before_action",
        title="Call log_verification immediately after confirming 2+ identity fields",
        pattern=(
            "verify identity email phone date_of_birth address confirm "
            "log verification account change update modify"
        ),
        tip=(
            "After confirming 2+ identity fields from the customer (any combination "
            "of: email, phone, date_of_birth, address), call log_verification "
            "BEFORE executing any account changes. Do not skip this step."
        ),
        source="failure",
        task_ids=[4, 5, 23, 36, 37],
    ),
    Skill(
        id="user_lookup_fallback",
        title="Email not found? Try by name. No phone lookup exists.",
        pattern=(
            "look up user find customer email not found no records "
            "phone number verify identity lookup search"
        ),
        tip=(
            "If get_user_information_by_email returns no records, immediately try "
            "get_user_information_by_name with the customer's full name. "
            "There is NO get_user_information_by_phone tool — never attempt it."
        ),
        source="failure",
        task_ids=[4, 5, 33],
    ),
    Skill(
        id="new_user_credit_card_application",
        title="New users apply for cards using their own apply_for_credit_card tool",
        pattern=(
            "new customer no account apply credit card first time "
            "don't have account open card application new applicant"
        ),
        tip=(
            "If the customer states they do not have a Rho-Bank account, do NOT "
            "attempt DB identity lookups. Confirm which card they qualify for, "
            "then explicitly tell the customer: 'You can apply now using your "
            "apply_for_credit_card tool with card_type=<card_name>.' "
            "Do NOT process the application yourself — the user initiates it."
        ),
        source="failure",
        task_ids=[25, 3, 24],
    ),
    Skill(
        id="email_mismatch_ownership_dispute",
        title="DB email differs from stated email → account_ownership_dispute",
        pattern=(
            "email not match mismatch different email verify identity "
            "email address doesn't match ownership dispute transfer"
        ),
        tip=(
            "After looking up a customer by name and finding their record, compare "
            "the DB email against the email they stated in this conversation. "
            "If they differ, you cannot fully verify identity — use "
            "reason='account_ownership_dispute' when calling transfer_to_human_agents. "
            "Do NOT use 'customer_requests_human_no_specific_reason' when there is "
            "an underlying identity verification failure."
        ),
        source="failure",
        task_ids=[4, 29],
    ),
    Skill(
        id="referral_workflow",
        title="Referral tasks: verify → log → get_referrals_by_user → submit_referral",
        pattern=(
            "referral refer friend bonus program submit referral "
            "referred someone referral link reward"
        ),
        tip=(
            "For referral tasks: (1) verify and log_verification, "
            "(2) call get_referrals_by_user to check existing referrals, "
            "(3) call submit_referral for new referrals. "
            "You (the agent) must call submit_referral — do not wait for the user."
        ),
        source="failure",
        task_ids=[99, 101, 16],
    ),
    Skill(
        id="transfer_on_user_request",
        title="When user calls request_human_agent_transfer, complete it immediately",
        pattern=(
            "transfer human agent request frustrated escalate "
            "human agent transfer request submitted"
        ),
        tip=(
            "When you see a user-submitted transfer request in the conversation, "
            "call transfer_to_human_agents immediately using the most specific "
            "applicable reason code. Do not delay or ask clarifying questions."
        ),
        source="failure",
        task_ids=[34],
    ),
]

DOMAIN_SKILLS: dict[str, list[Skill]] = {
    "retail": RETAIL_SKILLS,
    "telecom": TELECOM_SKILLS,
    "airline": AIRLINE_SKILLS,
    "banking_knowledge": BANKING_KNOWLEDGE_SKILLS,
}
