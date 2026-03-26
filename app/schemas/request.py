from typing import Literal

from pydantic import BaseModel, field_validator


class ClassifyRequest(BaseModel):
    """Incoming classification request."""

    message: str

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("message must be a non-empty string")
        return stripped
