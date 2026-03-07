import json
import logging
import os

from fastapi import APIRouter, HTTPException
from openai import OpenAI

from backend.models import VerifyRequest, VerifyResponse, VerifySource
from backend.qdrant_service import get_qdrant_client, embed
from backend.config import EC2_IP, QDRANT_PORT, COLLECTION_NAME

router = APIRouter(prefix="/api", tags=["verify"])
logger = logging.getLogger(__name__)

# ── LLM configuration ──────────────────────────────────────────────────

LLM_MODEL = "google/gemini-3-flash-preview"
LLM_TEMPERATURE = 0.1

# Incomplete-statement detection
MIN_WORD_COUNT = 4
MIN_LENGTH_RATIO = 0.5

VERDICT_LABEL = {
    "Pravda": "PRAVDA",
    "Nepravda": "NEPRAVDA",
    "Zavádzajúce": "ZAVÁDZAJÚCE",
    "Neoveriteľné": "NEOVERITEĽNÉ",
}

# ── System prompt (copied verbatim from Client/verify.py) ──────────────

SYSTEM_PROMPT = """\
Si analytik portálu Demagog.sk. Tvoja JEDINÁ úloha je porovnať vstupný výrok s výsledkami \
z databázy overených výrokov Demagog.sk a rozhodnúť, či niektorý záznam z databázy hovorí \
o TOM ISTOM faktickom tvrdení.

=== METODOLÓGIA DEMAGOG.SK ===

Demagog.sk overuje výlučne overiteľné tvrdenia postavené na faktoch — číselné údaje, \
minulé udalosti, historické fakty. NEOVERUJÚ sa politické názory, hodnotové súdy ani \
predpovede do budúcnosti.

Verdikty databázy a ich presný význam:

PRAVDA — Výrok používa správne informácie v správnom kontexte.

NEPRAVDA — Výrok sa nezhoduje s verejne dostupnými číslami alebo informáciami. \
Žiadny dostupný zdroj nepodporuje tvrdenie, a to ani pri použití alternatívnych \
metód výpočtu.

ZAVÁDZAJÚCE — Výrok spadá do jednej z troch kategórií:
  a) Nevhodné porovnania bez faktického základu.
  b) Informácia prezentovaná v inom kontexte, než bola pôvodne zamýšľaná — \
vytrhnutie z pôvodného kontextu.
  c) Vytváranie falošnej kauzality.

NEOVERITEĽNÉ — Neexistuje žiadny zdroj, ktorý by tvrdenie potvrdil alebo vyvrátil.

=== PRÍSNE PRAVIDLÁ ===

1. NIKDY nevytváraj vlastné hodnotenie. NIKDY nepoužívaj vlastné znalosti. \
Môžeš IBA preniesť verdikt z databázového záznamu Demagog.sk. Tvoja úloha nie je \
byť factchecker — tým je redakcia Demagog.sk. Ty len porovnávaš, či sa vstupný výrok \
zhoduje s už overeným záznamom.

2. Zhoda znamená, že vstupný výrok a databázový záznam hovoria o PRESNE TOM ISTOM \
faktickom tvrdení. Podobná téma NESTAČÍ. Tvrdenie musí obsahovať ten istý faktický \
nárok — tie isté čísla, ten istý smer trendu, ten istý subjekt a rovnakú polaritu \
(bez zmeny záporu).
   - "HDP rástol o 3 %" a "HDP rástol o 2,8 %" NIE SÚ zhoda (rôzne čísla).
   - "Nezamestnanosť klesla" a "Nezamestnanosť stúpla" NIE SÚ zhoda (opačný trend).
   - "Slovensko má najnižšiu nezamestnanosť v EÚ" a "Nezamestnanosť na Slovensku klesla" \
NIE SÚ zhoda (iné tvrdenie).
   - "Postavili sme 100 km diaľnic" a "Postavili sme 80 km diaľnic" NIE SÚ zhoda (iné číslo).
   - "Zvýšenie minimálnej mzdy z 1000 na 360 eur" a "Zvýšenie minimálnej mzdy z 352 na 380 eur" \
NIE SÚ zhoda (úplne iné čísla — 1000≠352, 360≠380).

2a. NEGÁCIA ÚPLNE MENÍ VÝZNAM VÝROKU. Ak vstupný výrok obsahuje zápor a databázový \
záznam nie, alebo naopak, NIE JE to zhoda — aj keď je téma, subjekt a všetko ostatné \
identické. Zápor v slovenčine má tieto formy:
   - Predpona "ne-" na slovese: "patrí" vs "nepatrí", "je" vs "nie je", "má" vs "nemá", \
"existuje" vs "neexistuje", "môže" vs "nemôže", "súhlasí" vs "nesúhlasí"
   - Samostatné záporné slovo "nie": "je členom" vs "nie je členom"
   - Slová "nikdy", "nikto", "nič", "žiadny/žiadna/žiadne", "bez", "ani"
   Príklady:
   - "Slovensko patrí do NATO" a "Slovensko nepatrí do NATO" NIE SÚ zhoda (zápor mení tvrdenie).
   - "Vláda má mandát" a "Vláda nemá mandát" NIE SÚ zhoda (zápor mení tvrdenie).
   - "Zákon existuje" a "Zákon neexistuje" NIE SÚ zhoda (zápor mení tvrdenie).
   - "Slovensko je suverénna krajina" a "Slovensko nie je suverénna krajina" NIE SÚ zhoda.
   POZOR: Vysoké skóre podobnosti NEZNAMENÁ zhodu! Vety s negáciou majú často veľmi \
vysoké skóre podobnosti (nad 0.90), pretože pojednávajú o tej istej téme. Vždy \
skontroluj prítomnosť záporu PRED rozhodnutím o zhode.

3. NIKDY neopravuj ani neinterpretuj predpokladané preklepy alebo chyby vo vstupnom výroku. \
Ber vstupný výrok DOSLOVNE tak, ako je napísaný. Ak vstup hovorí "z 1000 na 360 eur" a \
databáza hovorí "z 352 na 380 eur", sú to DVA RÔZNE výroky — aj keby sa zdalo, že ide o preklep. \
Nie je tvoja úloha hádať, čo autor myslel.

4. Mnohé fakty sa menia v čase (minimálna mzda, HDP, nezamestnanosť, rozpočet atď.). \
Výrok o minimálnej mzde v roku 2015 NIE JE ten istý výrok ako o minimálnej mzde v roku 2020, \
aj keď sa oba týkajú rovnakej témy. Ak vstupný výrok neuvádza rovnaké časové obdobie alebo \
rovnaké konkrétne hodnoty ako databázový záznam, NIE JE to zhoda.

5. Ak vstupný výrok je názor, hodnotový súd alebo predpoveď do budúcnosti, vráť verdikt \
"Nedostatok dát" s vysvetlením, že Demagog.sk neoveruje názory a predpovede.

6. Ak ŽIADNY výsledok z databázy presne nezodpovedá vstupnému výroku, vráť verdikt \
"Nedostatok dát". Nikdy nehádaj, nikdy neodhaduj — ak si nie si istý, vráť "Nedostatok dát".

7. Ak viacero výsledkov zodpovedá, vyber ten s najlepšou sémantickou zhodou a použi jeho verdikt.

8. Odpovedz VÝHRADNE v nasledujúcom JSON formáte, bez akéhokoľvek ďalšieho textu:

9. NEÚPLNÉ VÝROKY: Ak vstupný výrok je zjavne neúplný, fragmentárny alebo nevyjadruje \
ucelené faktické tvrdenie, vráť verdikt "Nedostatok dát". Neúplný výrok je taký, ktorému \
chýba podmět, prísudok alebo predmet — napríklad iba niekoľko slov z dlhšej vety. \
Aj keď databáza obsahuje podobný ÚPLNÝ výrok, neúplný vstup NIE JE možné klasifikovať, \
pretože neobsahuje celé tvrdenie. \
Príklady: \
   - "Slovensko patrí" — NEÚPLNÝ (chýba predmet: patrí kam? do čoho?) → Nedostatok dát \
   - "Ekonomika rástla" — NEÚPLNÝ (chýba kontext: o koľko? kedy?) → Nedostatok dát \
   - "Minister povedal že" — NEÚPLNÝ (chýba obsah výroku) → Nedostatok dát \
   - "Slovensko patrí do NATO" — ÚPLNÝ (subjekt + prísudok + predmet) → pokračuj v analýze \
POZOR: Vysoké skóre podobnosti NEZNAMENÁ, že vstupný výrok je úplný! Fragment vety môže \
mať vysoké skóre, pretože obsahuje kľúčové slová z úplného výroku.

Ak existuje zhoda:
{
  "zhoda": true,
  "verdikt": "<verdikt z databázy: Pravda|Nepravda|Zavádzajúce|Neoveriteľné>",
  "zdrojovy_vyrok": "<presný text výroku z databázy>",
  "odovodnenie_llm": "<vysvetlenie prečo sa vstupný výrok zhoduje s databázovým záznamom. \
MUSÍ obsahovať: (1) či sa zhodujú čísla, (2) či sa zhoduje smer/trend, (3) či je \
prítomný zápor v jednom ale nie v druhom výroku. Odkazuj na metodológiu Demagog.sk.>",
  "index_zhody": <index zvoleného výsledku, počítaný od 1>
}

Ak neexistuje zhoda:
{
  "zhoda": false,
  "verdikt": "Nedostatok dát",
  "zdrojovy_vyrok": null,
  "odovodnenie_llm": "<vysvetlenie prečo žiadny výsledok nezodpovedá, alebo prečo výrok \
nie je overiteľný podľa metodológie Demagog.sk>",
  "index_zhody": null
}
"""

# ── Incomplete-statement detection ────────────────────────────────────

def _is_incomplete_statement(statement: str, db_results: list[dict]) -> tuple[bool, str]:
    """Return (True, reason) if the statement is obviously fragmentary."""
    words = statement.split()

    if len(words) < MIN_WORD_COUNT:
        return True, (
            f"Vstupný výrok obsahuje iba {len(words)} slov(á). "
            "Úplné faktické tvrdenie vyžaduje aspoň podmět, prísudok a predmět."
        )

    if db_results:
        best_match = db_results[0]["vyrok"]
        if best_match:
            ratio = len(statement) / len(best_match)
            if ratio < MIN_LENGTH_RATIO:
                return True, (
                    f"Vstupný výrok je výrazne kratší než najbližší nájdený záznam "
                    f"({len(statement)} vs {len(best_match)} znakov, pomer {ratio:.0%}). "
                    "Neúplný výrok nie je možné overiť."
                )

    return False, ""


# ── OpenRouter client (lazy singleton) ─────────────────────────────────

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


# ── Helpers ─────────────────────────────────────────────────────────────

def _search_similar(query_text: str, top_k: int) -> list[dict]:
    vector = embed(query_text)
    client = get_qdrant_client()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "score": hit.score,
            "vyrok": (hit.payload or {}).get("Výrok", ""),
            "vyhodnotenie": (hit.payload or {}).get("Vyhodnotenie", ""),
            "odovodnenie": (hit.payload or {}).get("Odôvodnenie", ""),
            "oblast": (hit.payload or {}).get("Oblast", ""),
            "datum": (hit.payload or {}).get("Dátum", ""),
            "meno": (hit.payload or {}).get("Meno", ""),
            "politicka_strana": (hit.payload or {}).get("Politická strana", ""),
        }
        for hit in response.points
    ]


def _build_user_message(input_statement: str, db_results: list[dict]) -> str:
    parts = [f'Vstupný výrok: "{input_statement}"\n\nVýsledky z databázy:\n']
    for i, r in enumerate(db_results, 1):
        parts.append(
            f"--- Výsledok {i} (skóre podobnosti: {r['score']:.4f}) ---\n"
            f"Výrok: {r['vyrok']}\n"
            f"Verdikt: {r['vyhodnotenie']}\n"
            f"Odôvodnenie: {r['odovodnenie']}\n"
            f"Politik: {r['meno']} ({r['politicka_strana']})\n"
            f"Oblasť: {r['oblast']}\n"
            f"Dátum: {r['datum']}\n"
        )
    return "\n".join(parts)


def _call_llm(input_statement: str, db_results: list[dict]) -> dict:
    client = _get_openrouter_client()
    user_message = _build_user_message(input_statement, db_results)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    return json.loads(raw)


def _build_result(
    input_statement: str,
    llm_resp: dict,
    above: list[dict],
    all_results: list[dict],
    threshold: float,
) -> VerifyResponse:
    zdroj = None
    idx = llm_resp.get("index_zhody")
    if llm_resp.get("zhoda") and idx is not None and 1 <= idx <= len(above):
        m = above[idx - 1]
        zdroj = VerifySource(
            vyrok=m["vyrok"],
            vyhodnotenie=m["vyhodnotenie"],
            odovodnenie=m["odovodnenie"],
            oblast=m["oblast"],
            datum=m["datum"],
            meno=m["meno"],
            politicka_strana=m["politicka_strana"],
            skore_podobnosti=m["score"],
        )

    verdikt = llm_resp.get("verdikt", "Nedostatok dát")
    return VerifyResponse(
        vstupny_vyrok=input_statement,
        status="zhoda" if llm_resp.get("zhoda") else "bez_zhody",
        verdikt=verdikt,
        verdikt_label=VERDICT_LABEL.get(verdikt, verdikt),
        odovodnenie_llm=llm_resp.get("odovodnenie_llm", ""),
        zdrojovy_vyrok=llm_resp.get("zdrojovy_vyrok"),
        zdroj=zdroj,
        pouzity_prah=threshold,
        pocet_nad_prahom=len(above),
        pocet_celkom=len(all_results),
    )


def _build_no_data(
    input_statement: str,
    all_results: list[dict],
    threshold: float,
) -> VerifyResponse:
    top_score = all_results[0]["score"] if all_results else 0.0
    return VerifyResponse(
        vstupny_vyrok=input_statement,
        status="bez_zhody",
        verdikt="Nedostatok dát",
        verdikt_label="NEDOSTATOK DÁT",
        odovodnenie_llm=(
            f"Žiadny výsledok z databázy nepresiahol prah podobnosti {threshold}. "
            f"Najvyššie skóre: {top_score:.4f}."
        ),
        zdrojovy_vyrok=None,
        zdroj=None,
        pouzity_prah=threshold,
        pocet_nad_prahom=0,
        pocet_celkom=len(all_results),
    )


# ── Endpoint ────────────────────────────────────────────────────────────

@router.post("/verify", response_model=VerifyResponse)
def verify_statement(body: VerifyRequest):
    """Verify a political statement against the Demagog.sk database."""
    all_results = _search_similar(body.vyrok, body.top_k)

    if not all_results:
        return _build_no_data(body.vyrok, all_results, body.threshold)

    above = [r for r in all_results if r["score"] >= body.threshold]

    if not above:
        return _build_no_data(body.vyrok, all_results, body.threshold)

    # Step 2.5: Reject incomplete / fragmentary input
    incomplete, reason = _is_incomplete_statement(body.vyrok, above or all_results)
    if incomplete:
        return VerifyResponse(
            vstupny_vyrok=body.vyrok,
            status="bez_zhody",
            verdikt="Nedostatok dát",
            verdikt_label="NEDOSTATOK DÁT",
            odovodnenie_llm=reason,
            zdrojovy_vyrok=None,
            zdroj=None,
            pouzity_prah=body.threshold,
            pocet_nad_prahom=len(above),
            pocet_celkom=len(all_results),
        )

    try:
        llm_resp = _call_llm(body.vyrok, above)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {exc}")

    return _build_result(body.vyrok, llm_resp, above, all_results, body.threshold)
