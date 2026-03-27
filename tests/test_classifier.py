"""
test_classifier.py — Classifier service unit tests.

All tests patch `_call_anthropic` so no real API calls are made.
"""
import json
import pytest
from unittest.mock import MagicMock, patch

from app.services.classifier import (
    FALLBACK_RESPONSE,
    _extract_json,
    _extract_text,
    classify_message,
)
from app.schemas.response import ClassifyResponse


VALID_PAYLOAD = {
    "category": "bug",
    "priority": "high",
    "summary": "Login button is unresponsive.",
    "suggested_reply": "We are investigating the issue and will update you shortly.",
    "needs_human_review": False,
}


def _raw(payload: dict) -> str:
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# _extract_text — content block handling
# ---------------------------------------------------------------------------

class TestExtractText:
    def _make_response(self, *texts: str):
        blocks = []
        for t in texts:
            b = MagicMock()
            b.text = t
            blocks.append(b)
        resp = MagicMock()
        resp.content = blocks
        resp.stop_reason = "end_turn"
        return resp

    def test_single_text_block(self) -> None:
        resp = self._make_response('{"category":"question"}')
        assert _extract_text(resp) == '{"category":"question"}'

    def test_multiple_text_blocks_are_joined(self) -> None:
        """All text blocks must be concatenated, not just the first."""
        resp = self._make_response("Hello ", "World")
        assert _extract_text(resp) == "Hello \nWorld"

    def test_non_text_blocks_are_skipped(self) -> None:
        non_text = MagicMock(spec=[])  # no .text attribute
        text_block = MagicMock()
        text_block.text = "good"
        resp = MagicMock()
        resp.content = [non_text, text_block]
        assert _extract_text(resp) == "good"

    def test_raises_when_no_text_block(self) -> None:
        non_text = MagicMock(spec=[])
        resp = MagicMock()
        resp.content = [non_text]
        resp.stop_reason = "end_turn"
        with pytest.raises(ValueError, match="No text block found"):
            _extract_text(resp)


# ---------------------------------------------------------------------------
# _extract_json — JSON extraction from various wrapping formats
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_bare_json(self) -> None:
        raw = _raw(VALID_PAYLOAD)
        assert json.loads(_extract_json(raw)) == VALID_PAYLOAD

    def test_fenced_json(self) -> None:
        fenced = f"```json\n{_raw(VALID_PAYLOAD)}\n```"
        assert json.loads(_extract_json(fenced)) == VALID_PAYLOAD

    def test_fenced_json_no_language(self) -> None:
        fenced = f"```\n{_raw(VALID_PAYLOAD)}\n```"
        assert json.loads(_extract_json(fenced)) == VALID_PAYLOAD

    def test_prose_before_and_after(self) -> None:
        prose = f"Sure! Here you go:\n{_raw(VALID_PAYLOAD)}\nLet me know if you need more."
        assert json.loads(_extract_json(prose)) == VALID_PAYLOAD

    def test_nested_json_is_handled(self) -> None:
        """Balanced-brace scanner must handle nested objects without truncating."""
        nested = {"outer": {"inner": "value"}, "category": "bug"}
        raw = json.dumps(nested)
        assert json.loads(_extract_json(raw)) == nested


# ---------------------------------------------------------------------------
# classify_message — end-to-end service tests
# ---------------------------------------------------------------------------

class TestClassifyMessage:
    def test_returns_valid_response_on_good_output(self) -> None:
        with patch(
            "app.services.classifier._call_anthropic", return_value=_raw(VALID_PAYLOAD)
        ):
            result = classify_message("Login button broken.")

        assert isinstance(result, ClassifyResponse)
        assert result.category == "bug"
        assert result.priority == "high"
        assert result.needs_human_review is False
        assert result.is_fallback is False

    def test_fallback_on_unparseable_output(self) -> None:
        with patch(
            "app.services.classifier._call_anthropic", return_value="no json here"
        ):
            result = classify_message("Something happened.")

        assert result == FALLBACK_RESPONSE
        assert result.is_fallback is True

    def test_fallback_on_missing_fields(self) -> None:
        with patch(
            "app.services.classifier._call_anthropic",
            return_value=_raw({"category": "bug"}),
        ):
            result = classify_message("Something happened.")

        assert result == FALLBACK_RESPONSE

    def test_fallback_on_invalid_enum_value(self) -> None:
        bad = {**VALID_PAYLOAD, "category": "complaint"}
        with patch(
            "app.services.classifier._call_anthropic", return_value=_raw(bad)
        ):
            result = classify_message("I have a complaint.")

        assert result == FALLBACK_RESPONSE

    def test_fallback_on_anthropic_network_error(self) -> None:
        from anthropic import APIConnectionError

        with patch(
            "app.services.classifier._call_anthropic",
            side_effect=APIConnectionError(request=None),  # type: ignore[arg-type]
        ):
            result = classify_message("Server is down.")

        assert result == FALLBACK_RESPONSE
        assert result.is_fallback is True

    def test_parses_fenced_json(self) -> None:
        fenced = f"```json\n{_raw(VALID_PAYLOAD)}\n```"
        with patch("app.services.classifier._call_anthropic", return_value=fenced):
            result = classify_message("Login button broken.")

        assert result.category == "bug"
        assert result.is_fallback is False

    def test_parses_prose_wrapped_json(self) -> None:
        prose = f"Sure:\n{_raw(VALID_PAYLOAD)}\nLet me know."
        with patch("app.services.classifier._call_anthropic", return_value=prose):
            result = classify_message("Login button broken.")

        assert result.category == "bug"
        assert result.is_fallback is False
