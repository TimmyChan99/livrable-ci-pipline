"""
Supabase PostgreSQL tool — checks a product record by ID.

Expected table schema:
    CREATE TABLE products (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name        TEXT,
        status      TEXT CHECK (status IN ('pending', 'in_progress', 'done', 'stopped')),
        note        TEXT,
        created_at  TIMESTAMPTZ DEFAULT now(),
        updated_at  TIMESTAMPTZ DEFAULT now()
    );
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

_VALID_STATUSES = {"pending", "in_progress", "done", "stopped"}


def _get_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_ANON_KEY) must be set."
        )
    return create_client(url, key)


def check_product_status(product_id: str) -> dict[str, Any]:
    """
    Fetch a single product row from Supabase by its ID.

    Returns a dict with keys: id, name, status, note  — or {"error": "..."}.
    """
    if not product_id or not isinstance(product_id, str):
        return {"error": "Invalid product_id provided."}

    product_id = product_id.strip()
    if not product_id:
        return {"error": "product_id cannot be blank."}

    try:
        client = _get_client()
        response = (
            client.table("products")
            .select("id, name, status, note")
            .eq("id", product_id)
            .limit(1)
            .execute()
        )

        if not response.data:
            return {"error": f"No product found with ID '{product_id}'."}

        row: dict[str, Any] = response.data[0]

        # Validate status value in case DB has unexpected data
        if row.get("status") not in _VALID_STATUSES:
            row["status"] = "unknown"

        return row

    except EnvironmentError as env_err:
        return {"error": str(env_err)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Database error: {exc}"}