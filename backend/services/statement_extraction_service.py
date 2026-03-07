"""LLM-based extraction of verifiable factual claims from transcript."""

import json
import logging

from openai import OpenAI

from backend.config import LLM_MODEL, LLM_TEMPERATURE
from backend.models_video import TranscriptSegment, ExtractedStatement

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """\
Si analytik politických diskusií. Tvoja úloha je identifikovať overiteľné \
faktické tvrdenia v prepise politickej diskusie.

=== ČO EXTRAHOVAŤ ===

Extrahuj IBA tvrdenia, ktoré sú:
1. Overiteľné fakty - číselné údaje, štatistiky, historické udalosti, \
   legislatívne skutočnosti
2. Konkrétne - obsahujú špecifické čísla, dátumy, mená, zákony
3. Priraditeľné - povedané konkrétnym rečníkom (ak je identifikovateľný)

=== ČO NEEXTRAHOVAŤ ===

IGNORUJ:
- Pozdravy, procedurálne vyjadrenia ("Dobrý deň", "Ďakujem za slovo")
- Osobné názory a hodnotové súdy ("Myslím, že je to zlé")
- Otázky (aj rétorické)
- Predpovede do budúcnosti
- Vágne tvrdenia bez konkrétnych údajov
- Opakované tvrdenia (extrahuj len prvý výskyt)

=== IDENTIFIKÁCIA REČNÍKOV ===

Pokús sa identifikovať rečníka podľa:
1. Explicitné oslovenia v diskusii ("Pán minister...", "Pani poslankyňa...")
2. Sebapredstavenie ("Ja ako predseda vlády...")
3. Kontextové indikátory
Ak rečníka nevieš identifikovať, použi null.

=== ZACHOVANIE PÔVODNÉHO ZNENIA ===

Tvrdenie zachovaj čo najbližšie pôvodnému zneniu rečníka. NE parafrázuj. \
Ak je tvrdenie rozdelené cez viacero segmentov, spoj ho do jednej \
koherentnej vety, ale zachovaj pôvodné slová.

=== MAPOVANIE NA ČASOVÉ ZNAČKY ===

Každé tvrdenie prirad k start_time prvého segmentu, v ktorom začína, \
a end_time posledného segmentu, v ktorom končí.

=== FORMÁT ODPOVEDE ===

Odpovedz VÝHRADNE ako JSON pole objektov:
[
  {
    "text": "<presný text faktického tvrdenia>",
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
