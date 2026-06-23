"""
Support Agent Workflow — LangGraph + Supabase PostgreSQL
"""

from __future__ import annotations

import json
import os
import re
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from state import SupportState
from tools import check_product_status

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    print('-------> ', api_key)
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set.")
    return ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key)


# ── Node: extract ─────────────────────────────────────────────────────────────

def extract_node(state: SupportState) -> SupportState:
    """
    Use the LLM to extract user_name and product_id from the user message.
    Returns JSON: {"user_name": "...", "product_id": "..."}
    Missing fields come back as null.
    """
    llm = _get_llm()
    user_input = state["user_input"]

    prompt = f"""You are a data extraction assistant.
Extract the following fields from the user message below.
Return ONLY valid JSON with keys "user_name" and "product_id".
If a field is not present, use null.

User message: "{user_input}"

Rules:
- product_id looks like a UUID (e.g. 550e8400-e29b-41d4-a716-446655440000) or a short alphanumeric code.
- user_name is the person's name if mentioned.
- Do not add any explanation, just JSON."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        # Strip possible markdown fences
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        return {
            **state,
            "user_name": data.get("user_name"),
            "product_id": data.get("product_id"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **state,
            "error": f"Extraction failed: {exc}",
        }


# ── Node: validate ────────────────────────────────────────────────────────────

def validate_node(state: SupportState) -> SupportState:
    """Check that a product_id was actually extracted."""
    if state.get("error"):
        return state
    if not state.get("product_id"):
        return {
            **state,
            "error": "No product ID found in your request. Please provide a valid product ID.",
        }
    return state


# ── Node: fetch ───────────────────────────────────────────────────────────────

def fetch_node(state: SupportState) -> SupportState:
    """Query Supabase for the product record."""
    if state.get("error"):
        return state

    product_id: str = state["product_id"]  # type: ignore[assignment]
    result = check_product_status(product_id)

    if "error" in result:
        return {**state, "error": result["error"]}

    return {
        **state,
        "product_status": result.get("status"),
        "product_note": result.get("note"),
        "raw_product": result,
    }


# ── Node: respond ─────────────────────────────────────────────────────────────

def respond_node(state: SupportState) -> SupportState:
    """Generate a natural-language reply for the user."""
    llm = _get_llm()

    # ── error path ──
    if state.get("error"):
        name_greeting = f"Hi {state['user_name']}, " if state.get("user_name") else "Hello, "
        reply = (
            f"{name_greeting}I encountered an issue: {state['error']} "
            "Please check your request or contact our support team if the problem persists."
        )
        return {**state, "response": reply}

    # ── happy path ──
    name_greeting = state.get("user_name") or "there"
    status = state.get("product_status", "unknown")
    note = state.get("product_note") or "No additional details available."
    product_id = state.get("product_id", "")

    status_descriptions: dict[str, str] = {
        "pending": "is currently pending and has not been started yet",
        "in_progress": "is currently in progress and being worked on",
        "done": "has been completed successfully",
        "stopped": "has been stopped and is no longer active",
    }
    status_text = status_descriptions.get(status, f"has an unknown status: {status}")

    prompt = f"""You are a friendly and professional customer support agent.
Write a clear, helpful reply to the customer based on the data below.
Keep the tone warm but concise (2-4 sentences max).

Customer name: {name_greeting}
Product ID: {product_id}
Status: {status} — {status_text}
Note from team: {note}

End with an offer to help further if needed."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        reply = response.content.strip()
    except Exception as exc:  # noqa: BLE001
        reply = (
            f"Hi {name_greeting}, your product ({product_id}) {status_text}. "
            f"Details: {note}. Feel free to reach out if you need more help!"
        )

    return {**state, "response": reply}


# ── Router ────────────────────────────────────────────────────────────────────

def route_after_validate(
    state: SupportState,
) -> Literal["fetch_product", "respond"]:
    if state.get("error"):
        return "respond"
    return "fetch_product"


def route_after_fetch(
    state: SupportState,
) -> Literal["respond"]:
    return "respond"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(SupportState)

    graph.add_node("extract", extract_node)
    graph.add_node("validate", validate_node)
    graph.add_node("fetch_product", fetch_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"fetch_product": "fetch_product", "respond": "respond"},
    )
    graph.add_edge("fetch_product", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


# ── Public helper ─────────────────────────────────────────────────────────────

def run_support_agent(user_input: str) -> dict:
    """
    Run the support workflow for a given user message.

    Returns the final state dict, always including a 'response' key.
    """
    app = build_graph()
    initial_state: SupportState = {
        "user_input": user_input,
        "user_name": None,
        "product_id": None,
        "product_status": None,
        "product_note": None,
        "raw_product": None,
        "response": None,
        "error": None,
    }
    return app.invoke(initial_state)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Your request: ")
    result = run_support_agent(msg)
    print("\n── Agent Response ──────────────────────────────────")
    print(result["response"])
    if result.get("error"):
        print(f"\n[DEBUG] Error: {result['error']}")