"""
main.py — Application entry point.

Creates the FastAPI app, registers routes, and mounts the minimal UI.
"""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title="Support Ticket Classifier",
    description="AI-assisted support ticket classification API.",
    version="1.0.0",
)

# --- API routes ---
app.include_router(router, prefix="/api")

# --- Static files (optional JS / CSS) ---
static_dir = Path(__file__).parent / "ui" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# --- Templates ---
templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "ui" / "templates")
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index(request: Request) -> HTMLResponse:
    """Serve the minimal manual-testing UI."""
    return templates.TemplateResponse(request, "index.html")
