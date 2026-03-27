"""
classifier.py — Core service that calls Anthropic and parses the response.

All provider-specific code and fallback logic live here.
Routes and UI layers call `classify_message()` only.
"""
import json
import logging
import re

from pydantic import ValidationError

from anthropic import Anthropic, APIError, APIStatusError

from app.core.config import get_settings
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
    """Extract and join all text blocks from an Anthropic message response.

    Iterates all content blocks, collects every text block, and joins them
    with a newline.  Raises ``ValueError`` if no text block is found.
    """
    parts = [
        block.text
        for block in response.content
        if hasattr(block, "text") and isinstance(block.text, str)
    ]
    if not parts:
        raise ValueError(
            f"No text block found in Anthropic response. "
            f"stop_reason={response.stop_reason!r}, "
            f"content_types={[type(b).__name__ for b in response.content]}"
        )
    return "\n".join(parts)


def _extract_json(text: str) -> str:
    """Extract the first JSON object from *text*.

    Handles three common formats:
    1. Bare JSON: ``{"key": "value"}``
    2. Fenced JSON: ````` ```json\\n{...}\\n``` `````
    3. JSON embedded in surrounding prose

    The fence pattern is tried first (highest confidence), then a braces
    scan that counts depth to find the exact object boundaries.
    """
    # 1. Markdown code fence (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # 2. Find the JSON object by scanning for balanced braces
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return text  # let json.loads raise a meaningful error


def _call_anthropic(message: str) -> str:
    """Send the prompt to Anthropic and return the raw text response.

    Raises ``anthropic.APIError`` on provider or network failures.
    """
    s = get_settings()
    client = Anthropic(api_key=s.anthropic_api_key)
    response = client.messages.create(
        model=s.anthropic_model,
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": build_user_prompt(message)},
        ],
        temperature=0,
    )
    return _extract_text(response)


def _parse_and_validate(raw: str) -> ClassifyResponse:
    """Extract JSON from *raw*, parse it, and validate against ``ClassifyResponse``.

    Raises ``json.JSONDecodeError`` if the extracted text is not valid JSON.
    Raises ``pydantic.ValidationError`` if the JSON does not match the schema.
    """
    cleaned = _extract_json(raw)
    data = json.loads(cleaned)
    return ClassifyResponse.model_validate(data)


def classify_message(message: str) -> ClassifyResponse:
    """Classify a support message.

    Returns a validated ``ClassifyResponse``.  Falls back to
    ``FALLBACK_RESPONSE`` on any provider error or invalid model output.
    ``is_fallback=True`` is always set on the fallback so callers can
    surface a warning to the user.
    """
    try:
        raw = _call_anthropic(message)
        logger.info("Anthropic response received (%d chars)", len(raw))
        return _parse_and_validate(raw)

    except APIStatusError as exc:
        # Permanent failures (bad model id, auth, quota).  Will not self-heal.
        logger.error(
            "Anthropic config/auth error HTTP %s (model=%r): %s",
            exc.status_code,
            get_settings().anthropic_model,
            exc.message,
        )
        return FALLBACK_RESPONSE

    except APIError as exc:
        # Transient network / provider errors.
        logger.error("Anthropic network error: %s", exc)
        return FALLBACK_RESPONSE

    except json.JSONDecodeError as exc:
        logger.warning("Model output is not valid JSON — using fallback. Detail: %s", exc)
        return FALLBACK_RESPONSE

    except ValidationError as exc:
        logger.warning(
            "Model JSON failed schema validation — using fallback. Errors: %s",
            exc.error_count(),
        )
        return FALLBACK_RESPONSE

    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected classification error: %s", exc)
        return FALLBACK_RESPONSE
