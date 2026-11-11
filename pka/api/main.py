"""API entry point. Run with: uvicorn pka.api.main:app --reload"""
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Personal Knowledge Assistant")


class AskRequest(BaseModel):
    question: str
    as_of: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "phase": 0}


@app.post("/ask")
def ask(req: AskRequest):
    raise HTTPException(status_code=501, detail="Q&A arrives in Phase 1.")
