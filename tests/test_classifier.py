"""
test_classifier.py — Classifier service unit tests.

All tests patch `_call_anthropic` so no real API calls are made.
"""
import json
from unittest.mock import patch

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

    def test_fallback_on_invalid_json(self) -> None:
        with patch(
            "app.services.classifier._call_anthropic", return_value="not json at all"
        ):
            result = classify_message("Something happened.")

        assert result == FALLBACK_RESPONSE
        assert result.needs_human_review is True

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
