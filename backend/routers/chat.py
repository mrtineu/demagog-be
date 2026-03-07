import json
import logging
import os

from fastapi import APIRouter, HTTPException
from openai import OpenAI

from backend.models import ChatRequest, ChatResponse
from backend.qdrant_service import get_qdrant_client, embed
from backend.config import COLLECTION_NAME

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)

LLM_MODEL = "google/gemini-3-flash-preview"
LLM_TEMPERATURE = 0.3
TOP_K = 5
THRESHOLD = 0.5

_VERDICT_MAP = {
    "Pravda": "true",
    "Nepravda": "false",
    "Zavádzajúce": "misleading",
    "Neoveriteľné": "uncheckable",
}

SYSTEM_PROMPT = """\
Si asistent portálu Demagog.sk — overovateľa politických výrokov. Používateľ ti \
pošle správu (otázku alebo politický výrok). Tvoja úloha:

1. Ak správa vyzerá ako politický výrok na overenie, porovnaj ju s dodanými výsledkami \
z databázy overených výrokov. Ak nájdeš zhodu (rovnaké fakticke tvrdenie, rovnaké čísla, \
rovnaký smer, rovnaká polarita), odpovedz jasne s verdiktom a krátkym vysvetlením.

2. Ak správa je všeobecná otázka o factcheckingu, o Demagog.sk, alebo o metodológii, \
odpovedz stručne a informatívne.

3. Ak nemáš dostatočné dáta, povedz to úprimne.

Odpovedaj v slovenčine. Buď stručný ale informatívny. Nepoužívaj JSON formát — \
odpovedaj prirodzeným textom vhodným pre chat.

Ak ti boli dodané výsledky z databázy, zohľadni ich vo svojej odpovedi.\
"""

_openrouter_client: OpenAI | None = None


def _get_openrouter_client() -> OpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        api_key = os.getenv("OPENROUTER_KEY")
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="OPENROUTER_KEY not configured on the server.",
            )
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=api_key
        )
    return _openrouter_client


def _search_context(query: str) -> str:
    """Search Qdrant for relevant statements and format as context."""
    try:
        vector = embed(query)
        client = get_qdrant_client()
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=TOP_K,
            with_payload=True,
        )
        hits = [h for h in response.points if h.score >= THRESHOLD]
    except Exception:
        logger.warning("Failed to search Qdrant for chat context", exc_info=True)
        return ""

    if not hits:
        return ""

    parts = ["Výsledky z databázy overených výrokov:\n"]
    for i, hit in enumerate(hits, 1):
        p = hit.payload or {}
        verdict_sk = p.get("Vyhodnotenie", "")
        parts.append(
            f"--- {i} (skóre: {hit.score:.3f}) ---\n"
            f"Výrok: {p.get('Výrok', '')}\n"
            f"Verdikt: {verdict_sk}\n"
            f"Odôvodnenie: {p.get('Odôvodnenie', '')}\n"
            f"Politik: {p.get('Meno', '')} ({p.get('Politická strana', '')})\n"
            f"Dátum: {p.get('Dátum', '')}\n"
        )
    return "\n".join(parts)


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    """Conversational fact-checking endpoint."""
    context = _search_context(body.message)

    user_content = body.message
    if context:
        user_content = f"{body.message}\n\n{context}"

    client = _get_openrouter_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    reply = response.choices[0].message.content.strip()
    return ChatResponse(reply=reply)
