"""Reuse audited Flash accounting without changing the sealed implementation."""
from meta.life_optimizer_v25.context import (
    ContextLimit, ProposalFailure, FlashCounter, CONTEXT_LIMIT, OUTPUT_LIMIT,
    TOKEN_MARGIN, context_fingerprint, verify_usage, flash_counter,
)
