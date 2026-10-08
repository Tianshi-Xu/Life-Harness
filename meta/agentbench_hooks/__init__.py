"""Versioned, policy-free AgentBench H2–H5 interfaces for future Life runs."""

from .runtime import CompositeHarness, install, interface_fingerprint

__all__ = ["CompositeHarness", "install", "interface_fingerprint"]
