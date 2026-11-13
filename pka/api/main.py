"""Web API and interface.

  python -m pka.app                 # recommended: server + automatic folder syncing
  uvicorn pka.api.main:app          # API only (no automatic syncing)

The page itself is pka/web/index.html. Everything it does goes through /api/*.
"""
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class AskRequest(BaseModel):
    question: str
    as_of: Optional[str] = None


def create_app(assistant=None) -> FastAPI:
    from pka.service import Assistant
    assistant = assistant or Assistant()
    app = FastAPI(title="Personal Knowledge Assistant")
    app.state.assistant = assistant

    @app.get("/", response_class=HTMLResponse)
    def home():
        page = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    @app.get("/health")
    def health():
        return {"status": "ok", "phase": 6}

    @app.get("/api/status")
    def status():
        return assistant.status()

    @app.post("/api/ask")
    def ask(req: AskRequest):
        if req.as_of:
            try:
                date.fromisoformat(req.as_of)
            except ValueError:
                raise HTTPException(status_code=422, detail="as_of must be a date like 2026-06-30")
        return assistant.ask(req.question, req.as_of)

    @app.post("/ask")                     # the Phase 1 address still works
    def ask_legacy(req: AskRequest):
        return ask(req)

    @app.get("/api/documents")
    def documents():
        return assistant.documents()

    @app.get("/api/documents/{doc_id}")
    def document(doc_id: str):
        found = assistant.document(doc_id)
        if not found:
            raise HTTPException(status_code=404, detail=f"No document called '{doc_id}'.")
        return found

    @app.get("/api/documents/{doc_id}/chunks")
    def chunks(doc_id: str, version: Optional[int] = None):
        if version is None:
            found = assistant.document(doc_id)
            if not found:
                raise HTTPException(status_code=404, detail=f"No document called '{doc_id}'.")
            version = found["versions"][-1]["version"]
        return assistant.chunks(doc_id, version)

    @app.post("/api/sync")
    def sync():
        return {"started": assistant.start_sync()}

    return app


app = create_app()
