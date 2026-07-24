"""
API FastAPI de démonstration pour l'agent de triage médical (P14).
Éxpose l'inférence du modèle fine-tuné servi par vLLM, avec tracabilité
des interactions (log JSONL horodaté) pour les audits médicaux.

Usage local (dev): uv run uvicorn api.main:app --reload
"""

import json
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

VLLM_URL = "http://localhost:8000/v1/completions"
MODEL_NAME = "models/merged-triage-model"

PROMPT_TEMPLATE = """### Instruction:
Tu es un agent de triage medical. Evalue le niveau d'urgence et oriente le patient.

### Symptomes rapportes:
{symptoms}

### Reponse:
"""

TRACE_LOG_PATH = Path("logs/triage_interactions.jsonl")
TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Agent de triage medical, CHSA (POC)")


class TriageRequest(BaseModel):
    symptoms: str


class TriageResponse(BaseModel):
    interaction_id: str
    raw_completion: str


def log_interaction(interaction_id: str, symptoms: str, completion: str) -> None:
    """Ecrit une ligne de tracabilité JSONL, horodatée, pour audit médical."""
    record = {
        "interaction_id": interaction_id,
        "timestamp": time.time(),
        "symptoms": symptoms,
        "raw_completion": completion,
    }
    with open(TRACE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest):
    prompt = PROMPT_TEMPLATE.format(symptoms=request.symptoms)
    interaction_id = str(uuid.uuid4())

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            VLLM_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "max_tokens": 300,
                "stop": ["###"],
            },
        )
        response.raise_for_status()
        completion = response.json()["choices"][0]["text"].strip()

    log_interaction(interaction_id, request.symptoms, completion)

    return TriageResponse(interaction_id=interaction_id, raw_completion=completion)


@app.get("/health")
async def health():
    return {"status": "ok"}