"""
Shared state for the Support Agent LangGraph workflow.
"""

from typing import Any, Optional
from typing_extensions import TypedDict


class SupportState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    user_input: str                  # raw message from the user

    # ── Extracted ─────────────────────────────────────────────────────────────
    user_name: Optional[str]         # parsed customer name (may be None)
    product_id: Optional[str]        # parsed product UUID / code (may be None)

    # ── DB result ─────────────────────────────────────────────────────────────
    product_status: Optional[str]    # pending | in_progress | done | stopped
    product_note: Optional[str]      # free-text note from the products table
    raw_product: Optional[dict[str, Any]]  # full row returned from Supabase

    # ── Output ────────────────────────────────────────────────────────────────
    response: Optional[str]          # final natural-language reply to the user
    error: Optional[str]             # human-readable error description (or None)