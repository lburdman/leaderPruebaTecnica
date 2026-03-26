"""
test_api.py — FastAPI endpoint integration tests using httpx TestClient.

OpenAI is mocked at the service layer so tests run offline.
"""
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.classifier import FALLBACK_RESPONSE

client = TestClient(app)

VALID_PAYLOAD = {
    "category": "question",
    "priority": "low",
    "summary": "User asking about pricing.",
    "suggested_reply": "Happy to help! Here are our pricing details…",
    "needs_human_review": False,
}


def _raw(payload: dict) -> str:
    return json.dumps(payload)


class TestClassifyEndpoint:
    def test_success_returns_200_and_valid_schema(self) -> None:
        with patch(
            "app.services.classifier._call_openai", return_value=_raw(VALID_PAYLOAD)
        ):
            resp = client.post("/api/classify", json={"message": "What is the price?"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["category"] == "question"
        assert data["priority"] == "low"
        assert data["needs_human_review"] is False
        assert "summary" in data
        assert "suggested_reply" in data

    def test_empty_message_returns_422(self) -> None:
        resp = client.post("/api/classify", json={"message": ""})
        assert resp.status_code == 422

    def test_whitespace_only_message_returns_422(self) -> None:
        resp = client.post("/api/classify", json={"message": "   "})
        assert resp.status_code == 422

    def test_missing_message_field_returns_422(self) -> None:
        resp = client.post("/api/classify", json={})
        assert resp.status_code == 422

    def test_fallback_returned_on_openai_error(self) -> None:
        from openai import APIConnectionError

        with patch(
            "app.services.classifier._call_openai",
            side_effect=APIConnectionError(request=None),  # type: ignore[arg-type]
        ):
            resp = client.post("/api/classify", json={"message": "My server is down."})

        assert resp.status_code == 200
        data = resp.json()
        assert data["needs_human_review"] is True
        assert data["category"] == FALLBACK_RESPONSE.category

    def test_ui_home_page_returns_200(self) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
