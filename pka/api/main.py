"""API entry point. Run with: uvicorn pka.api.main:app --reload"""
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Personal Knowledge Assistant")


class AskRequest(BaseModel):
    question: str
    as_of: Optional[str] = None


@lru_cache(maxsize=1)
def get_system():
    from pka.baseline import BaselineSystem
    return BaselineSystem()


@app.get("/health")
def health():
    return {"status": "ok", "phase": 1}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        result = get_system().answer(req.question, as_of=req.as_of)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "answer": result.answer,
        "sources": [{"doc_id": c.doc_id, "version": c.version, "section": c.section,
                     "score": round(c.score, 4), "text": c.text} for c in result.retrieved],
    }
