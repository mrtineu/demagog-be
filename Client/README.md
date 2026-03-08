# Statement Verification System

Automated fact-checking tool for Slovak political statements. It combines semantic search over a database of 22,000+ pre-verified statements with an LLM-powered analysis pipeline and a web research fallback.

## Technologies

| Component | Technology | Role |
|-----------|-----------|------|
| Vector Database | **Qdrant** (port 6333) | Stores statement embeddings for similarity search |
| Embedding Model | **BAAI/bge-m3** via Infinity API (port 7997) | Converts text to 384-dimensional vectors (multilingual, supports Slovak) |
| LLM | **Google Gemini 3 Flash** via OpenRouter | Analyzes retrieved results and produces verdicts |
| Web Search | **Tavily Search API** | Fallback web research when no database match is found |
| Language | **Python 3** | `openai`, `qdrant-client`, `requests`, `tavily-python`, `python-dotenv` |
| Infrastructure | **AWS EC2** (m7i-flex.large) | Hosts Qdrant and Infinity in Docker containers |

## How It Works

### Overview

`verify.py` is an interactive CLI tool. The user enters a political statement in Slovak, and the system returns a verdict: **Pravda** (True), **Nepravda** (False), **Zavadzajuce** (Misleading), or **Neoveritelne** (Unverifiable). When no database match exists, `research_agent.py` searches the web as a fallback.

### Verification Workflow

```
User enters a statement
        |
        v
+-----------------------------+
|  1. EMBED                   |
|  Statement -> 384-dim vector|
|  (BAAI/bge-m3 via Infinity) |
+-------------+---------------+
              v
+-----------------------------+
|  2. SEARCH                  |
|  Query Qdrant for top 5     |
|  most similar statements    |
+-------------+---------------+
              v
+-----------------------------+
|  3. FILTER                  |
|  Keep results with cosine   |
|  similarity >= 0.60         |
+-------------+---------------+
              v
+-----------------------------+
|  4. VALIDATE INPUT          |
|  Reject fragments (<4 words |
|  or <50% of match length)   |
+-------------+---------------+
              v
+-----------------------------+
|  5. LLM ANALYSIS            |
|  Gemini compares input to   |
|  DB results using a strict  |
|  Slovak methodology prompt  |
+-------------+---------------+
              v
        +---------+
        | Match?  |
        +----+----+
         Yes |         No
         +---+         +---+
         v                 v
  +-------------+   +------------------+
  | Return      |   | Web Research     |
  | verdict     |   | Fallback         |
  | from DB     |   | (research_agent) |
  +-------------+   +------------------+
```

### Step-by-Step Details

**1. Embedding** -- The input statement is sent to the Infinity API running BAAI/bge-m3 on the EC2 instance. It returns a 384-dimensional vector.

**2. Semantic Search** -- The vector is used to query Qdrant, which returns the 5 nearest neighbors from the database of 22,000+ pre-verified political statements. Each result includes the original statement text, verdict, justification, politician name, party, date, and topic.

**3. Threshold Filtering** -- Only results with a cosine similarity score of 0.60 or higher are kept. If nothing passes, the system reports "Nedostatok dat" (Insufficient data) and can trigger the web research fallback.

**4. Input Validation** -- A completeness check rejects obviously fragmentary inputs (fewer than 4 words, or less than 50% of the best match's length). Fragments are rejected immediately without web research.

**5. LLM Analysis** -- The filtered results and the input statement are sent to Gemini 3 Flash via OpenRouter (temperature 0.1 for deterministic output). The system prompt encodes the full Demagog.sk fact-checking methodology:

- Use **only** the provided database results, no external knowledge.
- Require an **exact semantic match** -- same numbers, same trend direction, same time period.
- Detect **negation** ("ne-", "nie", "nikto", "ziadny") which flips meaning even when similarity scores are high.
- Treat **opinions, predictions, and value judgments** as Neoveritelne.
- Output a structured JSON with the verdict, reasoning, and matched source index.

**6. Result Assembly** -- The LLM's JSON response is combined with database metadata (politician, party, date, original justification, similarity score) into a final result.

**7. Output** -- Displayed as a formatted terminal report or as raw JSON (with `--json` flag).

### Web Research Fallback (`research_agent.py`)

When the database has no match, `research_agent.py` performs independent web-based verification:

1. **Query Generation** -- The LLM generates 2-3 optimized search queries from the Slovak statement.
2. **Phase 1 Search** -- Tavily searches only trusted domains (government sites, Eurostat, Reuters, Slovak news agencies, fact-checkers like demagog.sk).
3. **Phase 2 Search** -- If Phase 1 is insufficient, a broad search runs with 400+ conspiracy/propaganda domains excluded (RT, Sputnik, Infowars, etc.).
4. **Content Extraction** -- Full page content is retrieved from the top 5 URLs (capped at 3,000 chars each).
5. **LLM Analysis** -- Gemini analyzes the web sources, citing specific quotes and URLs.
6. **Safeguard Enforcement** -- Programmatic rules override the LLM if needed:
   - Fewer than 2 independent sources -> forced to **Neoveritelne**
   - Sources contradict each other -> forced to **Neoveritelne**
   - Low confidence with a definitive verdict -> forced to **Neoveritelne**

The research agent configuration (trusted domains, excluded domains, search settings, LLM parameters) lives in `config/research_config.json`.

## Usage

```bash
# Interactive mode (default)
python verify.py

# With JSON output
python verify.py --json

# Custom similarity threshold
python verify.py --threshold 0.7

# Disable web research fallback
python verify.py --no-research
```

Inside the interactive prompt:

```
>>> Slovensko ma najnizsiu nezamestnanost v EU

================================================================================
  VERDIKT: PRAVDA
================================================================================
  Vstupny vyrok:   Slovensko ma najnizsiu nezamestnanost v EU
  Zdrojovy vyrok:  Slovakia has lowest unemployment in EU
  Odovodnenie LLM: ...

  --- Zdroj z databazy ---
  Politik:         Peter Pellegrini (SMER-SD)
  Povodny verdikt: Pravda
  Skore:           0.8721
================================================================================
```

## Possible Verdicts

| Verdict | Meaning |
|---------|---------|
| **Pravda** | The statement is true |
| **Nepravda** | The statement is false |
| **Zavadzajuce** | The statement is misleading |
| **Neoveritelne** | The statement cannot be verified |
| **Nedostatok dat** | No matching data found in DB or web |

## Safeguards

- The LLM is explicitly forbidden from using its own training knowledge -- only provided sources.
- Negation detection prevents false matches on statements with opposite meaning.
- Input fragments are rejected before reaching the LLM.
- Web research requires 2+ independent sources for any definitive verdict.
- All output is labeled as an automatic suggestion -- final verification must be performed by a human analyst.
