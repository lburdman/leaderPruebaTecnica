"""
routes.py — API endpoints.

Route handlers are intentionally thin: they delegate all business logic
to the classifier service and let FastAPI / Pydantic handle validation.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas.request import ClassifyRequest
from app.schemas.response import ClassifyResponse
from app.services.classifier import classify_message

router = APIRouter()


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify a support ticket",
    description=(
        "Receives a user support message and returns a structured classification "
        "including category, priority, summary, a suggested reply, and a flag "
        "indicating whether human review is recommended."
    ),
)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Classify a support message and return a validated response."""
    return classify_message(request.message)
