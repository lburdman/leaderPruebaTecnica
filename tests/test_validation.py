"""
test_validation.py — Pydantic request/response schema tests.
"""
import pytest
from pydantic import ValidationError

from app.schemas.request import ClassifyRequest
from app.schemas.response import ClassifyResponse


class TestClassifyRequest:
    def test_valid_message(self) -> None:
        req = ClassifyRequest(message="My account is locked.")
        assert req.message == "My account is locked."

    def test_message_is_stripped(self) -> None:
        req = ClassifyRequest(message="  Hello there  ")
        assert req.message == "Hello there"

    def test_empty_message_raises(self) -> None:
        with pytest.raises(ValidationError):
            ClassifyRequest(message="")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValidationError):
            ClassifyRequest(message="   ")

    def test_missing_message_raises(self) -> None:
        with pytest.raises(ValidationError):
            ClassifyRequest()  # type: ignore[call-arg]


class TestClassifyResponse:
    def test_valid_response(self) -> None:
        resp = ClassifyResponse(
            category="bug",
            priority="high",
            summary="App crashes on login.",
            suggested_reply="We're looking into this.",
            needs_human_review=False,
        )
        assert resp.category == "bug"
        assert resp.priority == "high"

    def test_invalid_category_raises(self) -> None:
        with pytest.raises(ValidationError):
            ClassifyResponse(
                category="unknown",  # type: ignore[arg-type]
                priority="low",
                summary="x",
                suggested_reply="x",
                needs_human_review=False,
            )

    def test_invalid_priority_raises(self) -> None:
        with pytest.raises(ValidationError):
            ClassifyResponse(
                category="bug",
                priority="urgent",  # type: ignore[arg-type]
                summary="x",
                suggested_reply="x",
                needs_human_review=False,
            )

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValidationError):
            ClassifyResponse(  # type: ignore[call-arg]
                category="question",
                priority="low",
                # summary missing
                suggested_reply="x",
                needs_human_review=False,
            )
