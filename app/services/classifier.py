"""
classifier.py — Core service that calls Anthropic and parses the response.

All provider-specific code and fallback logic live here.
Routes and UI layers call `classify_message()` only.
"""
import json
import logging

from anthropic import Anthropic, APIError

from app.core.config import settings
from app.core.prompts import SYSTEM_PROMPT, build_user_prompt
from app.schemas.response import ClassifyResponse

logger = logging.getLogger("classifier")

# ---------------------------------------------------------------------------
# Deterministic fallback used whenever the model output cannot be trusted.
# ---------------------------------------------------------------------------
FALLBACK_RESPONSE = ClassifyResponse(
    category="other",
    priority="medium",
    summary="Unable to confidently classify the request.",
    suggested_reply=(
        "Thanks for your message. Our team will review it and get back to you soon."
    ),
    needs_human_review=True,
)


def _call_anthropic(message: str) -> str:
    """Send the prompt to Anthropic and return the raw text response.

    Raises ``anthropic.APIError`` on provider or network failures.
    """
    client = Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_user_prompt(message)},
        ],
        temperature=0,
    )
    raw = response.content[0].text
    return raw


def _parse_and_validate(raw: str) -> ClassifyResponse:
    """Parse JSON and validate it against ``ClassifyResponse``.

    Raises ``ValueError`` or ``json.JSONDecodeError`` if anything is wrong.
    """
    data = json.loads(raw)
    return ClassifyResponse.model_validate(data)


def classify_message(message: str) -> ClassifyResponse:
    """Classify a support message.

    Returns a validated ``ClassifyResponse``.  Falls back to
    ``FALLBACK_RESPONSE`` on any provider error or invalid model output.
    """
    try:
        raw = _call_anthropic(message)
        logger.info("Anthropic response received (%d chars)", len(raw))
        return _parse_and_validate(raw)
    except APIError as exc:
        logger.error("Anthropic provider error: %s", exc)
        return FALLBACK_RESPONSE
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Invalid model output — using fallback. Detail: %s", exc)
        return FALLBACK_RESPONSE
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during classification: %s", exc)
        return FALLBACK_RESPONSE
