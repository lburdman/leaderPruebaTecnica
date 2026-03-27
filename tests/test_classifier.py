"""
test_classifier.py — Classifier service unit tests.

All tests patch `_call_anthropic` so no real API calls are made.
"""
import json
from unittest.mock import MagicMock, patch

from app.services.classifier import FALLBACK_RESPONSE, classify_message
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

    def test_fallback_on_invalid_json(self) -> None:
        with patch(
            "app.services.classifier._call_anthropic", return_value="not json at all"
        ):
            result = classify_message("Something happened.")

        assert result == FALLBACK_RESPONSE
        assert result.needs_human_review is True
        assert result.is_fallback is True

    def test_fallback_on_missing_fields(self) -> None:
        incomplete = {"category": "bug"}  # missing required fields
        with patch(
            "app.services.classifier._call_anthropic", return_value=_raw(incomplete)
        ):
            result = classify_message("Something happened.")

        assert result == FALLBACK_RESPONSE

    def test_fallback_on_invalid_enum_value(self) -> None:
        bad_payload = {**VALID_PAYLOAD, "category": "complaint"}
        with patch(
            "app.services.classifier._call_anthropic", return_value=_raw(bad_payload)
        ):
            result = classify_message("I have a complaint.")

        assert result == FALLBACK_RESPONSE

    def test_fallback_on_anthropic_error(self) -> None:
        from anthropic import APIConnectionError

        with patch(
            "app.services.classifier._call_anthropic",
            side_effect=APIConnectionError(request=None),  # type: ignore[arg-type]
        ):
            result = classify_message("Server is down.")

        assert result == FALLBACK_RESPONSE
        assert result.needs_human_review is True
        assert result.is_fallback is True

    def test_parses_fenced_json(self) -> None:
        """Model output wrapped in ```json fences must be parsed correctly."""
        fenced = f"```json\n{_raw(VALID_PAYLOAD)}\n```"
        with patch(
            "app.services.classifier._call_anthropic", return_value=fenced
        ):
            result = classify_message("Login button broken.")

        assert result.category == "bug"
        assert result.is_fallback is False

    def test_parses_prose_wrapped_json(self) -> None:
        """JSON embedded in surrounding prose must be extracted correctly."""
        prose = f"Sure, here is the classification:\n{_raw(VALID_PAYLOAD)}\nLet me know if you need anything else."
        with patch(
            "app.services.classifier._call_anthropic", return_value=prose
        ):
            result = classify_message("Login button broken.")

        assert result.category == "bug"
        assert result.is_fallback is False

    def test_extract_text_from_response_blocks(self) -> None:
        """_extract_text must iterate blocks and return the first text block."""
        from app.services.classifier import _extract_text

        block = MagicMock()
        block.text = '{"category":"question"}'
        mock_response = MagicMock()
        mock_response.content = [block]

        assert _extract_text(mock_response) == '{"category":"question"}'

    def test_extract_text_raises_on_no_text_block(self) -> None:
        """_extract_text must raise ValueError if no text block is present."""
        import pytest
        from app.services.classifier import _extract_text

        non_text_block = MagicMock(spec=[])  # no .text attribute
        mock_response = MagicMock()
        mock_response.content = [non_text_block]
        mock_response.stop_reason = "end_turn"

        with pytest.raises(ValueError, match="No text block found"):
            _extract_text(mock_response)
