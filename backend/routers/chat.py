import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from openai import AsyncOpenAI

from backend.models import ChatRequest, ChatResponse
from backend.config import LLM_MODEL, OPENROUTER_KEY
from backend.services.llm_client import get_openrouter_client as get_sync_client
from backend.services.verification_service import verify_statement
from backend.services.research_service import research_statement

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger(__name__)

LLM_TEMPERATURE = 0.3
MAX_TOOL_ROUNDS = 5  # safety cap on the agentic loop

SYSTEM_PROMPT = """\
Si asistent portálu Demagog.sk — overovateľa politických výrokov. Používateľ ti \
pošle správu (otázku alebo politický výrok). Tvoja úloha:

1. Ak správa obsahuje konkrétne faktické tvrdenie alebo politický výrok na overenie, \
použi nástroj handle_statement. Nástroj overí výrok v databáze Demagog.sk a ak nenájde \
zhodu, automaticky vyhľadá informácie na webe.

2. Ak správa je všeobecná otázka, pozdrav, alebo konverzácia (napr. o factcheckingu, \
o Demagog.sk, metodológii), odpovedz priamo bez použitia nástroja.

Keď nástroj vráti výsledky z databázy Demagog.sk, odpovedz s verdiktom a vysvetlením. \
Keď výsledky pochádzajú z webového výskumu (označené ako WEBOVÝ VÝSKUM), jasne uveď, \
že overenie je založené na webových zdrojoch a nie na overenej databáze Demagog.sk.

Odpovedaj v slovenčine. Buď stručný ale informatívny. Odpovedaj prirodzeným textom \
vhodným pre chat.\
"""

# --- Tool schema ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "handle_statement",
            "description": (
                "Overí politický výrok alebo faktické tvrdenie. Najprv hľadá v databáze "
                "Demagog.sk a ak nenájde zhodu, automaticky vyhľadá informácie na webe. "
                "Použi IBA keď správa obsahuje konkrétne faktické tvrdenie na overenie."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "statement": {
                        "type": "string",
                        "description": "Politický výrok alebo faktické tvrdenie na overenie.",
                    }
                },
                "required": ["statement"],
            },
        },
    },
]

# --- Async OpenAI client (lazy singleton, for chat completions) ---

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


def _format_verification_result(result: dict) -> str:
    """Format a verify_statement result dict into a text string for the LLM."""
    verdikt = result.get("verdikt", "Nedostatok dát")
    odovodnenie = result.get("odovodnenie_llm", "")

    parts = ["VÝSLEDOK Z DATABÁZY DEMAGOG.SK:\n"]
    parts.append(f"Verdikt: {verdikt} ({result.get('verdikt_label', '')})")
    parts.append(f"Odôvodnenie: {odovodnenie}")

    zdroj = result.get("zdroj")
    if zdroj and isinstance(zdroj, dict):
        parts.append(f"\nZdrojový výrok: {zdroj.get('vyrok', '')}")
        parts.append(f"Politik: {zdroj.get('meno', '')} ({zdroj.get('politicka_strana', '')})")
        parts.append(f"Dátum: {zdroj.get('datum', '')}")
        parts.append(f"Skóre podobnosti: {zdroj.get('skore_podobnosti', 0):.3f}")

    return "\n".join(parts)


def _format_research_result(result: dict) -> str:
    """Format a research_statement result dict into a text string for the LLM."""
    verdikt = result.get("verdikt", "Neoveriteľné")
    odovodnenie = result.get("odovodnenie_llm", "")
    istota = result.get("istota", "nízka")
    sources = result.get("webove_zdroje", [])

    parts = [
        "⚠️ WEBOVÝ VÝSKUM (nie z databázy Demagog.sk):\n",
        f"Verdikt: {verdikt} ({result.get('verdikt_label', '')})",
        f"Istota: {istota}",
        f"Odôvodnenie: {odovodnenie}",
    ]

    if result.get("safeguard_override"):
        parts.append("Poznámka: Verdikt bol upravený bezpečnostným pravidlom.")

    if sources:
        parts.append(f"\nPoužité webové zdroje ({len(sources)}):")
        for i, src in enumerate(sources, 1):
            if isinstance(src, dict):
                url = src.get("url", "")
                nazov = src.get("nazov", "")
                citat = src.get("relevantny_citat", "")
                parts.append(f"  {i}. {nazov}\n     URL: {url}")
                if citat:
                    parts.append(f"     Citát: {citat}")

    return "\n".join(parts)


def _extract_links(result: dict) -> list[dict]:
    """Extract link objects from a research_statement result."""
    sources = result.get("webove_zdroje", [])
    links = []
    for src in sources:
        if isinstance(src, dict) and src.get("url"):
            links.append({
                "url": src.get("url", ""),
                "title": src.get("nazov", ""),
            })
    return links


async def _handle_statement(query: str) -> tuple[str, list[dict], list[dict]]:
    """Run verify_statement; if no match, fall back to research_statement.

    Returns (tool_result_text, links, all_results) where links is populated
    only when research_statement was used, and all_results contains the raw
    DB search results from verify_statement.
    """
    sync_client = get_sync_client()

    # Step 1: Try database verification
    try:
        result = await asyncio.to_thread(verify_statement, query, sync_client)
    except Exception:
        logger.warning("verify_statement failed", exc_info=True)
        result = {"status": "bez_zhody", "verdikt": "Nedostatok dát", "all_results": []}

    all_results = result.get("all_results", [])

    # If we got a DB match, return it (no links — came from DB)
    if result.get("status") == "zhoda":
        return _format_verification_result(result), [], all_results

    # Step 2: No DB match — fall back to web research
    logger.info("No DB match for '%s', falling back to web research", query[:80])
    try:
        research = await asyncio.to_thread(research_statement, query, sync_client)
        return _format_research_result(research), _extract_links(research), all_results
    except Exception:
        logger.error("research_statement failed", exc_info=True)
        return (
            _format_verification_result(result)
            + "\n\nWebový výskum zlyhal. Výrok sa nepodarilo overiť ani z webových zdrojov."
        ), [], all_results


def _build_reply(message: str, links: list[dict], all_results: list[dict]) -> str:
    """Build the JSON reply string with message, links, and all_results."""
    return json.dumps(
        {"message": message, "links": links, "all_results": all_results},
        ensure_ascii=False,
    )


# --- Endpoint ---

@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    """Conversational fact-checking endpoint with an agentic tool-calling loop."""
    llm = _get_openrouter_client()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": body.message},
    ]

    # Track links and DB results collected across tool rounds
    collected_links: list[dict] = []
    collected_all_results: list[dict] = []

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
            return ChatResponse(
                reply=_build_reply(
                    (assistant_msg.content or "").strip(),
                    collected_links,
                    collected_all_results,
                )
            )

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
            try:
                args = json.loads(tc.function.arguments)
                statement = args.get("statement", body.message)
            except (json.JSONDecodeError, KeyError):
                statement = body.message

            if tc.function.name == "handle_statement":
                tool_result, links, all_results = await _handle_statement(statement)
                collected_links.extend(links)
                collected_all_results.extend(all_results)
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
        return ChatResponse(
            reply=_build_reply(
                (fallback.choices[0].message.content or "").strip(),
                collected_links,
                collected_all_results,
            )
        )
    except Exception as exc:
        logger.error("Fallback LLM call failed", exc_info=True)
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc
