import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI

from backend.models import ChatRequest, ChatResponse
from backend.qdrant_service import get_qdrant_client, embed
from backend.config import COLLECTION_NAME, LLM_MODEL, OPENROUTER_KEY
from shared.verdicts import VERDICT_MAP as _VERDICT_MAP

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)

LLM_TEMPERATURE = 0.3
TOP_K = 5
THRESHOLD = 0.5
MAX_TOOL_ROUNDS = 5  # safety cap on the agentic loop

SYSTEM_PROMPT = """\
Si asistent portálu Demagog.sk — overovateľa politických výrokov. Používateľ ti \
pošle správu (otázku alebo politický výrok). Tvoja úloha:

1. Ak správa vyzerá ako politický výrok na overenie, použi nástroj search_demagog_database \
na nájdenie relevantných overených výrokov. Ak nájdeš zhodu (rovnaké fakticke tvrdenie, \
rovnaké čísla, rovnaký smer, rovnaká polarita), odpovedz jasne s verdiktom a krátkym vysvetlením.

2. Ak správa je všeobecná otázka o factcheckingu, o Demagog.sk, alebo o metodológii, \
odpovedz stručne a informatívne bez nutnosti vyhľadávať v databáze.

3. Ak nemáš dostatočné dáta, povedz to úprimne.

Odpovedaj v slovenčine. Buď stručný ale informatívny. Nepoužívaj JSON formát — \
odpovedaj prirodzeným textom vhodným pre chat.\
"""

# --- Tool schema ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_demagog_database",
            "description": (
                "Vyhľadá v databáze Demagog.sk overené politické výroky podobné zadanému textu. "
                "Použi tento nástroj vždy, keď potrebuješ overiť faktické tvrdenie, "
                "porovnať výrok s existujúcimi verdiktami alebo nájsť relevantnú fact-checkingovú históriu."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Politický výrok alebo tvrdenie, ktoré chceš overiť v databáze.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# --- Async OpenAI client (lazy singleton) ---

_openrouter_client: AsyncOpenAI | None = None


def _get_openrouter_client() -> AsyncOpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        if not OPENROUTER_KEY:
            raise HTTPException(
                status_code=503,
                detail="OPENROUTER_KEY not configured on the server.",
            )
        _openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY
        )
    return _openrouter_client


# --- Tool implementation ---

async def _search_context(query: str) -> str:
    """Search Qdrant for relevant statements and return a formatted context string."""
    try:
        vector = await asyncio.to_thread(embed, query)
        qdrant = get_qdrant_client()
        response = await asyncio.to_thread(
            qdrant.query_points,
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=TOP_K,
            with_payload=True,
        )
        hits = [h for h in response.points if h.score >= THRESHOLD]
    except Exception:
        logger.warning("Failed to search Qdrant for chat context", exc_info=True)
        return "Vyhľadávanie v databáze zlyhalo."

    if not hits:
        return "V databáze sa nenašli žiadne relevantné výroky."

    parts = ["Výsledky z databázy overených výrokov:\n"]
    for i, hit in enumerate(hits, 1):
        p = hit.payload or {}
        parts.append(
            f"--- {i} (skóre: {hit.score:.3f}) ---\n"
            f"Výrok: {p.get('Výrok', '')}\n"
            f"Verdikt: {p.get('Vyhodnotenie', '')}\n"
            f"Odôvodnenie: {p.get('Odôvodnenie', '')}\n"
            f"Politik: {p.get('Meno', '')} ({p.get('Politická strana', '')})\n"
            f"Dátum: {p.get('Dátum', '')}\n"
        )
    return "\n".join(parts)


# --- Endpoint ---

@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """Conversational fact-checking endpoint with an agentic tool-calling loop."""
    llm = _get_openrouter_client()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": body.message},
    ]

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = await llm.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                tools=TOOLS,
                tool_choice="auto",
                messages=messages,
            )
        except Exception as exc:
            logger.error("LLM API call failed", exc_info=True)
            raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc

        assistant_msg = response.choices[0].message

        # No tool calls → the model produced the final answer
        if not assistant_msg.tool_calls:
            return ChatResponse(reply=(assistant_msg.content or "").strip())

        # Append the assistant turn (including tool_calls) to the conversation history
        messages.append(
            {
                "role": "assistant",
                "content": assistant_msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ],
            }
        )

        # Execute each tool call and feed results back
        for tc in assistant_msg.tool_calls:
            if tc.function.name == "search_demagog_database":
                try:
                    args = json.loads(tc.function.arguments)
                    query = args.get("query", body.message)
                except (json.JSONDecodeError, KeyError):
                    query = body.message
                tool_result = await _search_context(query)
            else:
                tool_result = f"Neznámy nástroj: {tc.function.name}"
                logger.warning("LLM requested unknown tool: %s", tc.function.name)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                }
            )

    # Safety fallback: MAX_TOOL_ROUNDS exhausted — one final call without tools
    try:
        fallback = await llm.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=messages,
        )
        return ChatResponse(reply=(fallback.choices[0].message.content or "").strip())
    except Exception as exc:
        logger.error("Fallback LLM call failed", exc_info=True)
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc
