from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    anthropic_api_key: str
    anthropic_model: str = "claude-haiku-4-5"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings.

    Using a cached factory instead of a module-level singleton means:
    - Tests can clear the cache and inject different env vars.
    - Import of this module never fails due to missing env vars.
    """
    return Settings()  # type: ignore[call-arg]
