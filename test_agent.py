"""
Tests for the Support Agent LangGraph workflow.

Strategy: mock the LLM (extract / respond nodes) and the DB tool (fetch node)
so tests run without real credentials and are fast + deterministic.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from state import SupportState

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.content = content
    return mock


def _llm_side_effect(extract_json: dict, respond_text: str = "Here is your update."):
    """
    Returns a side_effect callable for ChatGoogleGenerativeAI.invoke:
    - First call  → extraction JSON
    - Second call → natural-language response
    """
    calls: list[int] = [0]

    def _invoke(_messages):
        calls[0] += 1
        if calls[0] == 1:
            return _make_llm_response(json.dumps(extract_json))
        return _make_llm_response(respond_text)

    return _invoke


# ── Tests: extract_node ───────────────────────────────────────────────────────

class TestExtractNode:
    def test_extracts_name_and_product_id(self):
        from agent import extract_node

        extracted = {"user_name": "Alice", "product_id": "abc-123"}
        with patch("agent._get_llm") as mock_llm_ctor:
            mock_llm_ctor.return_value.invoke = MagicMock(
                return_value=_make_llm_response(json.dumps(extracted))
            )
            state: SupportState = {
                "user_input": "Hi, I'm Alice and my product ID is abc-123",
                "user_name": None, "product_id": None,
                "product_status": None, "product_note": None,
                "raw_product": None, "response": None, "error": None,
            }
            result = extract_node(state)

        assert result["user_name"] == "Alice"
        assert result["product_id"] == "abc-123"
        assert result["error"] is None

    def test_no_product_id_returns_none(self):
        from agent import extract_node

        extracted = {"user_name": "Bob", "product_id": None}
        with patch("agent._get_llm") as mock_llm_ctor:
            mock_llm_ctor.return_value.invoke = MagicMock(
                return_value=_make_llm_response(json.dumps(extracted))
            )
            state: SupportState = {
                "user_input": "Hi Bob here, any news?",
                "user_name": None, "product_id": None,
                "product_status": None, "product_note": None,
                "raw_product": None, "response": None, "error": None,
            }
            result = extract_node(state)

        assert result["user_name"] == "Bob"
        assert result["product_id"] is None

    def test_llm_failure_sets_error(self):
        from agent import extract_node

        with patch("agent._get_llm") as mock_llm_ctor:
            mock_llm_ctor.return_value.invoke = MagicMock(side_effect=RuntimeError("LLM down"))
            state: SupportState = {
                "user_input": "Check product 999",
                "user_name": None, "product_id": None,
                "product_status": None, "product_note": None,
                "raw_product": None, "response": None, "error": None,
            }
            result = extract_node(state)

        assert result["error"] is not None
        assert "Extraction failed" in result["error"]

    def test_strips_markdown_fences_from_llm(self):
        from agent import extract_node

        raw = "```json\n{\"user_name\": \"Charlie\", \"product_id\": \"xyz-9\"}\n```"
        with patch("agent._get_llm") as mock_llm_ctor:
            mock_llm_ctor.return_value.invoke = MagicMock(
                return_value=_make_llm_response(raw)
            )
            state: SupportState = {
                "user_input": "Charlie here, product xyz-9",
                "user_name": None, "product_id": None,
                "product_status": None, "product_note": None,
                "raw_product": None, "response": None, "error": None,
            }
            result = extract_node(state)

        assert result["product_id"] == "xyz-9"


# ── Tests: validate_node ──────────────────────────────────────────────────────

class TestValidateNode:
    def _base_state(self, **overrides) -> SupportState:
        base: SupportState = {
            "user_input": "test", "user_name": "Alice", "product_id": "p-1",
            "product_status": None, "product_note": None,
            "raw_product": None, "response": None, "error": None,
        }
        return {**base, **overrides}  # type: ignore[return-value]

    def test_passes_when_product_id_present(self):
        from agent import validate_node
        result = validate_node(self._base_state())
        assert result["error"] is None

    def test_error_when_product_id_missing(self):
        from agent import validate_node
        result = validate_node(self._base_state(product_id=None))
        assert result["error"] is not None
        assert "product ID" in result["error"]

    def test_propagates_existing_error(self):
        from agent import validate_node
        result = validate_node(self._base_state(error="upstream error", product_id=None))
        # should still have the error, not overwrite with a new one
        assert result["error"] == "upstream error"


# ── Tests: fetch_node ─────────────────────────────────────────────────────────

class TestFetchNode:
    def _base_state(self, **overrides) -> SupportState:
        base: SupportState = {
            "user_input": "test", "user_name": "Alice", "product_id": "p-1",
            "product_status": None, "product_note": None,
            "raw_product": None, "response": None, "error": None,
        }
        return {**base, **overrides}  # type: ignore[return-value]

    def test_fetches_product_status(self):
        from agent import fetch_node

        db_row = {"id": "p-1", "name": "Widget", "status": "done", "note": "Shipped!"}
        with patch("agent.check_product_status", return_value=db_row):
            result = fetch_node(self._base_state())

        assert result["product_status"] == "done"
        assert result["product_note"] == "Shipped!"
        assert result["error"] is None

    def test_db_error_sets_error(self):
        from agent import fetch_node

        with patch("agent.check_product_status", return_value={"error": "Not found"}):
            result = fetch_node(self._base_state())

        assert result["error"] == "Not found"

    def test_skips_when_error_already_set(self):
        from agent import fetch_node

        with patch("agent.check_product_status") as mock_db:
            result = fetch_node(self._base_state(error="prior error"))
            mock_db.assert_not_called()

        assert result["error"] == "prior error"


# ── Tests: respond_node ───────────────────────────────────────────────────────

class TestRespondNode:
    def _base_state(self, **overrides) -> SupportState:
        base: SupportState = {
            "user_input": "test", "user_name": "Alice", "product_id": "p-1",
            "product_status": "done", "product_note": "All good.",
            "raw_product": {"id": "p-1", "status": "done", "note": "All good."},
            "response": None, "error": None,
        }
        return {**base, **overrides}  # type: ignore[return-value]

    def test_generates_reply_on_happy_path(self):
        from agent import respond_node

        with patch("agent._get_llm") as mock_llm_ctor:
            mock_llm_ctor.return_value.invoke = MagicMock(
                return_value=_make_llm_response("Great news Alice, product p-1 is done!")
            )
            result = respond_node(self._base_state())

        assert result["response"] == "Great news Alice, product p-1 is done!"

    def test_error_reply_without_llm(self):
        from agent import respond_node

        result = respond_node(self._base_state(error="No product found"))
        assert result["response"] is not None
        assert "No product found" in result["response"]

    def test_fallback_reply_when_llm_fails(self):
        from agent import respond_node

        with patch("agent._get_llm") as mock_llm_ctor:
            mock_llm_ctor.return_value.invoke = MagicMock(side_effect=RuntimeError("LLM down"))
            result = respond_node(self._base_state())

        assert result["response"] is not None
        assert "p-1" in result["response"]


# ── Tests: full workflow ──────────────────────────────────────────────────────

class TestFullWorkflow:
    def test_happy_path_end_to_end(self):
        from agent import run_support_agent

        db_row = {"id": "p-42", "name": "Widget", "status": "in_progress", "note": "ETA tomorrow."}
        extract_json = {"user_name": "Dana", "product_id": "p-42"}

        call_count = [0]

        def llm_invoke(_msgs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_llm_response(json.dumps(extract_json))
            return _make_llm_response("Hi Dana, your product p-42 is in progress.")

        with (
            patch("agent._get_llm") as mock_llm_ctor,
            patch("agent.check_product_status", return_value=db_row),
        ):
            mock_llm_ctor.return_value.invoke = llm_invoke
            result = run_support_agent("Hi I'm Dana, product p-42 please")

        assert result["user_name"] == "Dana"
        assert result["product_id"] == "p-42"
        assert result["product_status"] == "in_progress"
        assert result["response"] is not None
        assert result["error"] is None

    def test_missing_product_id_path(self):
        from agent import run_support_agent

        extract_json = {"user_name": "Eve", "product_id": None}

        with patch("agent._get_llm") as mock_llm_ctor:
            mock_llm_ctor.return_value.invoke = MagicMock(
                return_value=_make_llm_response(json.dumps(extract_json))
            )
            result = run_support_agent("Hi I'm Eve, just checking in")

        assert result["error"] is not None
        assert "product ID" in result["error"]
        assert result["response"] is not None

    def test_product_not_found_in_db(self):
        from agent import run_support_agent

        extract_json = {"user_name": "Frank", "product_id": "ghost-99"}

        call_count = [0]

        def llm_invoke(_msgs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_llm_response(json.dumps(extract_json))
            return _make_llm_response("Sorry Frank, not found.")

        with (
            patch("agent._get_llm") as mock_llm_ctor,
            patch("agent.check_product_status", return_value={"error": "No product found with ID 'ghost-99'."}),
        ):
            mock_llm_ctor.return_value.invoke = llm_invoke
            result = run_support_agent("Frank here, product ghost-99")

        assert result["error"] is not None
        assert result["response"] is not None

    @pytest.mark.parametrize("status", ["pending", "in_progress", "done", "stopped"])
    def test_all_valid_statuses(self, status: str):
        from agent import run_support_agent

        extract_json = {"user_name": "Gina", "product_id": "p-status"}
        db_row = {"id": "p-status", "name": "Item", "status": status, "note": "Note here."}

        call_count = [0]

        def llm_invoke(_msgs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_llm_response(json.dumps(extract_json))
            return _make_llm_response(f"Status is {status}.")

        with (
            patch("agent._get_llm") as mock_llm_ctor,
            patch("agent.check_product_status", return_value=db_row),
        ):
            mock_llm_ctor.return_value.invoke = llm_invoke
            result = run_support_agent("Gina here, product p-status")

        assert result["product_status"] == status
        assert result["error"] is None


# ── Tests: tools ──────────────────────────────────────────────────────────────

class TestCheckProductStatus:
    def test_returns_product_row(self):
        from tools import check_product_status

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
            {"id": "abc", "name": "Widget", "status": "done", "note": "Shipped"}
        ]
        with patch("tools._get_client", return_value=mock_client):
            result = check_product_status("abc")

        assert result["status"] == "done"
        assert result["note"] == "Shipped"

    def test_empty_result_returns_error(self):
        from tools import check_product_status

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = []
        with patch("tools._get_client", return_value=mock_client):
            result = check_product_status("missing-id")

        assert "error" in result
        assert "No product found" in result["error"]

    def test_blank_id_returns_error(self):
        from tools import check_product_status
        result = check_product_status("   ")
        assert "error" in result

    def test_invalid_status_normalized(self):
        from tools import check_product_status

        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
            {"id": "abc", "name": "Widget", "status": "weird_status", "note": "Hmm"}
        ]
        with patch("tools._get_client", return_value=mock_client):
            result = check_product_status("abc")

        assert result["status"] == "unknown"

    def test_missing_env_returns_error(self):
        from tools import check_product_status
        with patch("tools._get_client", side_effect=EnvironmentError("Missing env")):
            result = check_product_status("any-id")
        assert "error" in result