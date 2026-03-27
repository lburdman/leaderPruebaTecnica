"""
classifier.py — Core service that calls Anthropic and parses the response.

All provider-specific code and fallback logic live here.
Routes and UI layers call `classify_message()` only.
"""
import json
import logging
import re

from anthropic import Anthropic, APIError, APIStatusError

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
    is_fallback=True,
)


def _extract_text(response) -> str:  # type: ignore[no-untyped-def]
    """Safely extract text from an Anthropic message response.

    Iterates content blocks and returns the first text block found.
    Raises ``ValueError`` if no text block is present.
    """
    for block in response.content:
        if hasattr(block, "text") and isinstance(block.text, str):
            return block.text
    raise ValueError(
        f"No text block found in Anthropic response. "
        f"stop_reason={response.stop_reason!r}, "
        f"content_types={[type(b).__name__ for b in response.content]}"
    )


def _strip_fences(text: str) -> str:
    """Strip optional markdown code fences around JSON.

    Handles outputs like:
        ```json
        {...}
        ```
    and returns the inner content.  If no fences are found the original
    string is returned unchanged.
    """
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Also handle a bare JSON object that might be surrounded by prose
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text


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
    return _extract_text(response)


def _parse_and_validate(raw: str) -> ClassifyResponse:
    """Strip fences, parse JSON, and validate against ``ClassifyResponse``.

    Raises ``ValueError`` or ``json.JSONDecodeError`` if anything is wrong.
    """
    cleaned = _strip_fences(raw)
    data = json.loads(cleaned)
    return ClassifyResponse.model_validate(data)


def classify_message(message: str) -> ClassifyResponse:
    """Classify a support message.

    Returns a validated ``ClassifyResponse``.  Falls back to
    ``FALLBACK_RESPONSE`` on any provider error or invalid model output.
    The fallback is always marked ``is_fallback=True`` so callers can
    surface a warning to the user.
    """
    try:
        raw = _call_anthropic(message)
        logger.info("Anthropic response received (%d chars)", len(raw))
        return _parse_and_validate(raw)
    except APIStatusError as exc:
        # Configuration / auth / quota errors — these will not self-heal
        logger.error(
            "Anthropic API error %s (model=%r): %s",
            exc.status_code,
            settings.anthropic_model,
            exc.message,
        )
        return FALLBACK_RESPONSE
    except APIError as exc:
        # Network / transient provider errors
        logger.error("Anthropic provider error: %s", exc)
        return FALLBACK_RESPONSE
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Invalid model output — using fallback. Detail: %s", exc)
        return FALLBACK_RESPONSE
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during classification: %s", exc)
        return FALLBACK_RESPONSE
