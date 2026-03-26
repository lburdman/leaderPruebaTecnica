from typing import Literal

from pydantic import BaseModel


Category = Literal["bug", "billing", "feature_request", "question", "other"]
Priority = Literal["low", "medium", "high"]


class ClassifyResponse(BaseModel):
    """Structured classification response returned to callers."""

    category: Category
    priority: Priority
    summary: str
    suggested_reply: str
    needs_human_review: bool
