# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests",
#     "qdrant-client",
#     "openai",
#     "python-dotenv",
# ]
# ///

import argparse
import io
import json
import os
import sys

# Ensure stdin/stdout handle Slovak characters correctly regardless of locale
if hasattr(sys.stdin, "buffer"):
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from openai import OpenAI
import requests
from qdrant_client import QdrantClient

load_dotenv()

# Infrastructure
EC2_IP = "13.48.59.38"
QDRANT_PORT = 6333
INFINITY_PORT = 7997
COLLECTION = "politicke_vyroky_final_1"
MODEL = "BAAI/bge-m3"

# Verification defaults
TOP_K = 5
DEFAULT_THRESHOLD = 0.6
LLM_MODEL = "google/gemini-3-flash-preview"
LLM_TEMPERATURE = 0.1

VERDICT_LABEL = {
    "Pravda": "PRAVDA",
    "Nepravda": "NEPRAVDA",
    "Zavádzajúce": "ZAVÁDZAJÚCE",
    "Neoveriteľné": "NEOVERITEĽNÉ",
}


# --- Infrastructure ---

def embed(text: str) -> list[float]:
    """Get embedding vector via the Infinity API."""
    resp = requests.post(
        f"http://{EC2_IP}:{INFINITY_PORT}/embeddings",
        json={"model": MODEL, "input": text},
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def get_openrouter_client() -> OpenAI:
    """Create OpenRouter client using .env key."""
    api_key = os.getenv("OPENROUTER_KEY")
    if not api_key:
        print("Error: OPENROUTER_KEY not found in .env file.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


# --- Search & Filter ---

def search_similar(qdrant: QdrantClient, query_text: str, top_k: int = TOP_K) -> list[dict]:
    """Embed query and search Qdrant. Returns list of dicts with score + payload fields."""
    vector = embed(query_text)
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    ).points

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
        for hit in results
    ]


def filter_by_threshold(results: list[dict], threshold: float) -> list[dict]:
    """Keep only results with similarity score >= threshold."""
    return [r for r in results if r["score"] >= threshold]


# --- LLM System Prompt ---

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


# --- LLM Call ---

def build_user_message(input_statement: str, db_results: list[dict]) -> str:
    """Format the user message with the input statement and all DB results."""
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


def call_llm(client: OpenAI, input_statement: str, db_results: list[dict]) -> dict:
    """Send filtered results to LLM and parse JSON response."""
    user_message = build_user_message(input_statement, db_results)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw_content = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw_content.startswith("```"):
        lines = raw_content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_content = "\n".join(lines).strip()

    return json.loads(raw_content)


# --- Result Assembly ---

def build_verification_result(
    input_statement: str,
    llm_response: dict,
    db_results: list[dict],
    all_raw_results: list[dict],
    threshold: float,
) -> dict:
    """Assemble the full verification result combining LLM analysis with DB metadata."""
    result = {
        "vstupny_vyrok": input_statement,
        "status": "zhoda" if llm_response.get("zhoda") else "bez_zhody",
        "verdikt": llm_response.get("verdikt", "Nedostatok dát"),
        "verdikt_label": VERDICT_LABEL.get(
            llm_response.get("verdikt", ""), llm_response.get("verdikt", "Nedostatok dát")
        ),
        "odovodnenie_llm": llm_response.get("odovodnenie_llm", ""),
        "zdrojovy_vyrok": llm_response.get("zdrojovy_vyrok"),
        "zdroj": None,
        "pouzity_prah": threshold,
        "pocet_nad_prahom": len(db_results),
        "pocet_celkom": len(all_raw_results),
    }

    # If match found, attach full metadata from the matched DB result
    idx = llm_response.get("index_zhody")
    if llm_response.get("zhoda") and idx is not None and 1 <= idx <= len(db_results):
        matched = db_results[idx - 1]
        result["zdroj"] = {
            "vyrok": matched["vyrok"],
            "vyhodnotenie": matched["vyhodnotenie"],
            "odovodnenie": matched["odovodnenie"],
            "oblast": matched["oblast"],
            "datum": matched["datum"],
            "meno": matched["meno"],
            "politicka_strana": matched["politicka_strana"],
            "skore_podobnosti": matched["score"],
        }

    return result


def build_no_data_result(
    input_statement: str, all_raw_results: list[dict], threshold: float
) -> dict:
    """Build structured result when no DB results pass the threshold."""
    top_score = all_raw_results[0]["score"] if all_raw_results else 0.0
    return {
        "vstupny_vyrok": input_statement,
        "status": "bez_zhody",
        "verdikt": "Nedostatok dát",
        "verdikt_label": "NEDOSTATOK DÁT",
        "odovodnenie_llm": (
            f"Žiadny výsledok z databázy nepresiahol prah podobnosti {threshold}. "
            f"Najvyššie skóre: {top_score:.4f}."
        ),
        "zdrojovy_vyrok": None,
        "zdroj": None,
        "pouzity_prah": threshold,
        "pocet_nad_prahom": 0,
        "pocet_celkom": len(all_raw_results),
    }


# --- Output Formatting ---

def print_human_readable(result: dict) -> None:
    """Pretty-print result for terminal."""
    status = result["status"]
    print(f"\n{'=' * 72}")

    if status == "zhoda":
        verdikt = result["verdikt_label"]
        print(f"  VERDIKT: {verdikt}")
        print(f"{'=' * 72}")
        print(f"\n  Vstupný výrok:   {result['vstupny_vyrok']}")
        print(f"  Zdrojový výrok:  {result['zdrojovy_vyrok']}")
        print(f"\n  Odôvodnenie LLM: {result['odovodnenie_llm']}")

        src = result.get("zdroj")
        if src:
            print(f"\n  --- Zdroj z databázy ---")
            print(f"  Politik:         {src['meno']} ({src['politicka_strana']})")
            print(f"  Pôvodný verdikt: {src['vyhodnotenie']}")
            print(f"  Oblasť:          {src['oblast'] or 'N/A'}")
            print(f"  Dátum:           {src['datum'] or 'N/A'}")
            print(f"  Skóre:           {src['skore_podobnosti']:.4f}")
            odov = src.get("odovodnenie", "")
            if odov:
                print(f"  Odôvodnenie:     {odov[:400]}{'...' if len(odov) > 400 else ''}")
    else:
        print(f"  VERDIKT: NEDOSTATOK DÁT")
        print(f"{'=' * 72}")
        print(f"\n  Vstupný výrok:   {result['vstupny_vyrok']}")
        print(f"\n  Dôvod:           {result['odovodnenie_llm']}")

    print(f"\n  [Prah: {result['pouzity_prah']} | "
          f"Nad prahom: {result['pocet_nad_prahom']}/{result['pocet_celkom']}]")
    print(f"{'=' * 72}\n")


def print_json(result: dict) -> None:
    """Print result as formatted JSON."""
    print(json.dumps(result, ensure_ascii=False, indent=2))


# --- Main Loop & CLI ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overovač politických výrokov - fact-check verification"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results in JSON format instead of human-readable",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Similarity score threshold (default: {DEFAULT_THRESHOLD})",
    )
    return parser.parse_args()


def verify_statement(
    statement: str,
    qdrant: QdrantClient,
    llm_client: OpenAI,
    threshold: float,
) -> dict:
    """Full verification pipeline for a single statement."""
    # Step 1: Search Qdrant
    all_results = search_similar(qdrant, statement, TOP_K)

    if not all_results:
        return build_no_data_result(statement, all_results, threshold)

    # Step 2: Filter by threshold
    above_threshold = filter_by_threshold(all_results, threshold)

    if not above_threshold:
        return build_no_data_result(statement, all_results, threshold)

    # Step 3: LLM evaluation
    llm_response = call_llm(llm_client, statement, above_threshold)

    # Step 4: Assemble result
    return build_verification_result(
        statement, llm_response, above_threshold, all_results, threshold
    )


def main():
    args = parse_args()

    # Initialize connections once
    qdrant = QdrantClient(host=EC2_IP, port=QDRANT_PORT)
    llm_client = get_openrouter_client()

    print(f"Overovač výrokov | Qdrant: {EC2_IP}:{QDRANT_PORT} | Prah: {args.threshold}")
    print("Zadajte politický výrok na overenie. Prázdny riadok alebo Ctrl+C na ukončenie.\n")

    while True:
        try:
            statement = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nUkončenie.")
            break

        if not statement:
            print("Ukončenie.")
            break

        try:
            result = verify_statement(statement, qdrant, llm_client, args.threshold)
        except json.JSONDecodeError as e:
            print(f"Chyba: LLM vrátil neplatný JSON: {e}\n", file=sys.stderr)
            continue
        except requests.exceptions.ConnectionError:
            print(
                "Chyba: Nedá sa pripojiť k serveru (embedding/Qdrant).\n",
                file=sys.stderr,
            )
            continue
        except requests.exceptions.HTTPError as e:
            print(f"Chyba HTTP: {e}\n", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Neočakávaná chyba: {e}\n", file=sys.stderr)
            continue

        if args.json:
            print_json(result)
        else:
            print_human_readable(result)


if __name__ == "__main__":
    main()
