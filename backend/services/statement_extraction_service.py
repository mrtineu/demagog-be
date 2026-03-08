"""LLM-based extraction of verifiable factual claims from transcript."""

import json
import logging

from openai import OpenAI

from backend.config import LLM_MODEL, LLM_TEMPERATURE
from backend.models_video import TranscriptSegment, ExtractedStatement

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """\
Si analytik portálu Demagog.sk. Tvoja úloha je identifikovať overiteľné \
faktické tvrdenia v prepise politickej diskusie.

=== ČO EXTRAHOVAŤ ===

Extrahuj tvrdenia, ktoré spadajú do KTOREJKOĽVEK z týchto kategórií:

1. Číselné a štatistické tvrdenia — čísla, percentá, sumy, počty
2. Pripisovanie činov — kto (osoba, strana, inštitúcia, vláda) čo urobil \
alebo neurobil ("Dzurindova vláda sprivatizovala nemocnice")
3. Pripisovanie pozícií — čo niekto povedal, priznal, navrhol, odmietol \
("SaS priznáva, že zruší dôchodok")
4. Legislatívne a inštitucionálne tvrdenia — aké zákony sa prijali, ako \
rozhodli súdy, čo urobil prezident/parlament
5. Historické tvrdenia — čo sa v minulosti stalo alebo nestalo
6. Porovnávacie a superlatívne tvrdenia — "najdlhší", "rekordný", \
"viac ako", "prvýkrát"
7. Tvrdenia o zahraničnej politike — čo urobili iné krajiny alebo \
medzinárodné organizácie

Tvrdenie NEMUSÍ obsahovať konkrétne čísla. Stačí, ak tvrdí niečo \
konkrétne o reálnom svete, čo sa dá overiť alebo vyvrátiť.

=== ČO NEEXTRAHOVAŤ ===

IGNORUJ:
- Pozdravy a procedurálne vyjadrenia ("Dobrý deň", "Ďakujem za slovo")
- Čisto hodnotové súdy BEZ faktického jadra ("Je to hanba", "To je zlé")
- Otázky, ktoré samy o sebe neobsahujú faktické tvrdenie
- Čisté špekulácie o budúcnosti bez faktického základu
- Opakované tvrdenia (extrahuj len prvý výskyt)

POZOR: Ak tvrdenie obsahuje hodnotové slovo, ale zároveň faktický \
základ, EXTRAHUJ ho. Príklad: "Katastrofálne rozvrátené verejné financie" \
je tvrdenie o stave financií. "Rekordné tržby maloobchodu" je tvrdenie \
o tržbách.

=== ÚPLNOSŤ A KONTEXT TVRDENIA ===

Každé extrahované tvrdenie musí byť SAMO O SEBE zrozumiteľné — čitateľ, \
ktorý nevidel diskusiu, musí pochopiť, čo rečník tvrdí.

Pravidlá:
- Ak tvrdenie odkazuje zámenami na niečo povedané skôr ("to", "tento \
zákon", "ten úrad"), doplň kontext v zátvorke: \
"(zákon o hazarde, pozn.)", "(trinásty, pozn.) dôchodok", \
"(na Ukrajinu, pozn.)"
- Ak vynechávaš nepodstatnú časť citátu, použi: "(...)"
- Ak je tvrdenie rozdelené cez viacero segmentov, spoj ho do súvislej \
a kompletnej citácie
- Viacvetné tvrdenia zachovaj celé, ak tvoria jeden logický argument
- Zachovaj pôvodné znenie rečníka — NEPARAFRÁZUJ

=== PRÍKLADY SPRÁVNE EXTRAHOVANÝCH TVRDENÍ ===

Pripisovanie činu:
"…bývalej Dzurindovej vlády, ktorá sprivatizovala všetky nemocnice."

Pripisovanie pozície:
"SaS to verejne priznáva, že zruší tento (trinásty, pozn.) dôchodok."

Inštitucionálne tvrdenie:
"Pán prezident vám to vrátil naspäť (zákon o hazarde, pozn.), nie s \
nejakými detailnými pripomienkami, ale s tým, že celý zákon treba \
zhodiť zo stola."

Superlatív:
"Veď ja už som nevidel dlhšiu rozpravu v parlamente, ako bola napr. \
pri rušení Úradu na ochranu oznamovateľov."

Zahraničná politika:
"Kritizujeme Veľkú Britániu kvôli tomu, že financovala volebnú kampaň \
Progresívneho Slovenska v roku 2023 cez nastrčených influencerov."

Číselné:
"42 % konsolidácie musí zvládať bežný občan."

=== IDENTIFIKÁCIA REČNÍKOV ===

Pokús sa identifikovať rečníka podľa:
1. Explicitné oslovenia ("Pán minister...", "Pani poslankyňa...")
2. Sebapredstavenie ("Ja ako predseda vlády...")
3. Kontextové indikátory
Ak rečníka nevieš identifikovať, použi null.

=== MAPOVANIE NA ČASOVÉ ZNAČKY ===

Každé tvrdenie prirad k start_time prvého segmentu, v ktorom začína, \
a end_time posledného segmentu, v ktorom končí.

=== FORMÁT ODPOVEDE ===

Odpovedz VÝHRADNE ako JSON pole objektov:
[
  {
    "text": "<úplný, kontextovo zrozumiteľný text tvrdenia>",
    "speaker": "<meno rečníka alebo null>",
    "start_time": <čas začiatku v sekundách>,
    "end_time": <čas konca v sekundách>,
    "segment_indices": [<indexy relevantných segmentov>]
  }
]

Ak v prepise nie sú žiadne overiteľné faktické tvrdenia, vráť prázdne pole [].
"""


def extract_statements(
    segments: list[TranscriptSegment],
    llm_client: OpenAI,
    model: str | None = None,
) -> list[ExtractedStatement]:
    """Extract verifiable factual claims from transcript segments.

    Args:
        segments: Timestamped transcript segments.
        llm_client: OpenRouter client.
        model: LLM model to use (defaults to config LLM_MODEL).

    Returns:
        List of extracted statements with timestamps.
    """
    if not segments:
        return []

    model = model or LLM_MODEL

    # Build the user message with numbered, timestamped segments
    lines = []
    for i, seg in enumerate(segments):
        lines.append(
            f"[{i}] [{_format_time(seg.start_time)} - {_format_time(seg.end_time)}] "
            f"{seg.text}"
        )
    user_message = "\n".join(lines)

    response = llm_client.chat.completions.create(
        model=model,
        temperature=LLM_TEMPERATURE,
        max_tokens=8192,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw_lines = raw.split("\n")
        raw_lines = [l for l in raw_lines if not l.strip().startswith("```")]
        raw = "\n".join(raw_lines).strip()

    parsed = json.loads(raw)

    return [
        ExtractedStatement(
            text=item["text"],
            speaker=item.get("speaker"),
            start_time=float(item["start_time"]),
            end_time=float(item["end_time"]),
            segment_indices=item.get("segment_indices", []),
        )
        for item in parsed
        if item.get("text", "").strip()
    ]


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
