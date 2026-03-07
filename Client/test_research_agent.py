"""Unit tests for research_agent.py"""

import json
from unittest.mock import MagicMock, patch

import pytest

from research_agent import (
    build_research_result,
    build_research_user_message,
    enforce_conservative_safeguards,
    filter_excluded,
    generate_search_queries,
    get_all_trusted_domains,
    load_config,
    search_web,
)


# ---------------------------------------------------------------------------
# filter_excluded
# ---------------------------------------------------------------------------


class TestFilterExcluded:
    def test_removes_exact_domain_match(self):
        results = [
            {"url": "https://hlavnespravy.sk/article/123", "title": "Test"},
            {"url": "https://dennikn.sk/article/456", "title": "Good"},
        ]
        excluded = ["hlavnespravy.sk"]
        filtered = filter_excluded(results, excluded)
        assert len(filtered) == 1
        assert filtered[0]["title"] == "Good"

    def test_removes_subdomain_match(self):
        results = [
            {"url": "https://www.rt.com/news/article", "title": "Bad"},
            {"url": "https://sme.sk/article", "title": "Good"},
        ]
        excluded = ["rt.com"]
        filtered = filter_excluded(results, excluded)
        assert len(filtered) == 1
        assert filtered[0]["title"] == "Good"

    def test_keeps_all_when_none_excluded(self):
        results = [
            {"url": "https://dennikn.sk/article", "title": "A"},
            {"url": "https://sme.sk/article", "title": "B"},
        ]
        excluded = ["rt.com", "hlavnespravy.sk"]
        filtered = filter_excluded(results, excluded)
        assert len(filtered) == 2

    def test_handles_empty_results(self):
        assert filter_excluded([], ["rt.com"]) == []

    def test_handles_empty_excluded_list(self):
        results = [{"url": "https://example.com", "title": "A"}]
        assert len(filter_excluded(results, [])) == 1

    def test_handles_missing_url(self):
        results = [{"title": "No URL"}]
        filtered = filter_excluded(results, ["example.com"])
        assert len(filtered) == 1  # missing url -> can't match, keep it


# ---------------------------------------------------------------------------
# enforce_conservative_safeguards
# ---------------------------------------------------------------------------


class TestEnforceConservativeSafeguards:
    def test_overrides_verdict_with_one_source(self):
        response = {
            "verdikt": "Pravda",
            "istota": "vysoká",
            "pocet_podpornych_zdrojov": 1,
            "pouzite_zdroje": [{"url": "https://stat.gov.sk"}],
            "protirecie": None,
        }
        result = enforce_conservative_safeguards(response)
        assert result["verdikt"] == "Neoveriteľné"
        assert result["safeguard_override"] is True
        assert result["povodny_verdikt_llm"] == "Pravda"

    def test_overrides_verdict_with_contradictions(self):
        response = {
            "verdikt": "Nepravda",
            "istota": "stredná",
            "pocet_podpornych_zdrojov": 3,
            "pouzite_zdroje": [
                {"url": "https://a.sk"},
                {"url": "https://b.sk"},
                {"url": "https://c.sk"},
            ],
            "protirecie": "Zdroje sa líšia v údajoch o HDP",
        }
        result = enforce_conservative_safeguards(response)
        assert result["verdikt"] == "Neoveriteľné"
        assert result["safeguard_override"] is True

    def test_overrides_low_confidence_definitive_verdict(self):
        response = {
            "verdikt": "Zavádzajúce",
            "istota": "nízka",
            "pocet_podpornych_zdrojov": 3,
            "pouzite_zdroje": [
                {"url": "https://a.sk"},
                {"url": "https://b.sk"},
                {"url": "https://c.sk"},
            ],
            "protirecie": None,
        }
        result = enforce_conservative_safeguards(response)
        assert result["verdikt"] == "Neoveriteľné"
        assert result["safeguard_override"] is True

    def test_overrides_when_source_records_fewer_than_claimed(self):
        response = {
            "verdikt": "Pravda",
            "istota": "vysoká",
            "pocet_podpornych_zdrojov": 3,
            "pouzite_zdroje": [{"url": "https://a.sk"}],  # only 1 record
            "protirecie": None,
        }
        result = enforce_conservative_safeguards(response)
        assert result["verdikt"] == "Neoveriteľné"
        assert result["safeguard_override"] is True

    def test_does_not_override_valid_response(self):
        response = {
            "verdikt": "Pravda",
            "istota": "vysoká",
            "pocet_podpornych_zdrojov": 3,
            "pouzite_zdroje": [
                {"url": "https://a.sk"},
                {"url": "https://b.sk"},
                {"url": "https://c.sk"},
            ],
            "protirecie": None,
        }
        result = enforce_conservative_safeguards(response)
        assert result["verdikt"] == "Pravda"
        assert result["safeguard_override"] is False

    def test_does_not_override_neoveritelne(self):
        response = {
            "verdikt": "Neoveriteľné",
            "istota": "nízka",
            "pocet_podpornych_zdrojov": 0,
            "pouzite_zdroje": [],
            "protirecie": None,
        }
        result = enforce_conservative_safeguards(response)
        assert result["verdikt"] == "Neoveriteľné"
        assert result["safeguard_override"] is False


# ---------------------------------------------------------------------------
# get_all_trusted_domains
# ---------------------------------------------------------------------------


class TestGetAllTrustedDomains:
    def test_flattens_categories(self):
        config = {
            "trusted_domains": {
                "cat_a": ["a.sk", "b.sk"],
                "cat_b": ["c.sk"],
            }
        }
        result = get_all_trusted_domains(config)
        assert set(result) == {"a.sk", "b.sk", "c.sk"}

    def test_deduplicates(self):
        config = {
            "trusted_domains": {
                "cat_a": ["a.sk", "b.sk"],
                "cat_b": ["b.sk", "c.sk"],
            }
        }
        result = get_all_trusted_domains(config)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# build_research_user_message
# ---------------------------------------------------------------------------


class TestBuildResearchUserMessage:
    def test_formats_statement_and_sources(self):
        results = [
            {
                "url": "https://stat.gov.sk/data",
                "title": "Štatistiky",
                "content": "snippet text",
                "raw_content": "full content text",
            }
        ]
        msg = build_research_user_message("Test výrok", results)
        assert "Test výrok" in msg
        assert "stat.gov.sk" in msg
        assert "Štatistiky" in msg
        assert "full content text" in msg

    def test_truncates_long_content(self):
        results = [
            {
                "url": "https://example.com",
                "title": "Long",
                "content": "short",
                "raw_content": "x" * 5000,
            }
        ]
        msg = build_research_user_message("Test", results)
        assert "[skrátené]" in msg

    def test_handles_empty_results(self):
        msg = build_research_user_message("Test", [])
        assert "Test" in msg
        assert "Nájdených webových zdrojov: 0" in msg


# ---------------------------------------------------------------------------
# build_research_result
# ---------------------------------------------------------------------------


class TestBuildResearchResult:
    def test_basic_structure(self):
        llm_response = {
            "verdikt": "Pravda",
            "istota": "vysoká",
            "odovodnenie_llm": "Dôvod",
            "pouzite_zdroje": [{"url": "https://a.sk"}],
            "pocet_podpornych_zdrojov": 1,
            "protirecie": None,
            "safeguard_override": False,
        }
        result = build_research_result("Test výrok", llm_response, 5)
        assert result["status"] == "webovy_vyskum"
        assert result["typ_overenia"] == "webovy_vyskum"
        assert result["verdikt"] == "Pravda"
        assert result["verdikt_label"] == "PRAVDA"
        assert result["pocet_najdenych_zdrojov"] == 5
        assert result["zdrojovy_vyrok"] is None
        assert result["zdroj"] is None

    def test_unknown_verdict_label(self):
        llm_response = {
            "verdikt": "SomethingNew",
            "istota": "nízka",
            "odovodnenie_llm": "",
            "pouzite_zdroje": [],
            "pocet_podpornych_zdrojov": 0,
            "protirecie": None,
        }
        result = build_research_result("Test", llm_response, 0)
        assert result["verdikt_label"] == "SomethingNew"


# ---------------------------------------------------------------------------
# Integration-style tests (with mocking)
# ---------------------------------------------------------------------------


class TestSearchWebMocked:
    @patch("research_agent.TavilyClient")
    @patch("research_agent.os.getenv", return_value="fake-key")
    def test_phase1_sufficient_skips_phase2(self, mock_env, mock_tavily_cls):
        """When Phase 1 returns enough results, Phase 2 is not called."""
        mock_client = MagicMock()
        mock_tavily_cls.return_value = mock_client
        mock_client.search.return_value = {
            "results": [
                {"url": "https://a.sk/1", "title": "A", "content": "c", "score": 0.9},
                {"url": "https://b.sk/2", "title": "B", "content": "c", "score": 0.8},
                {"url": "https://c.sk/3", "title": "C", "content": "c", "score": 0.7},
            ]
        }

        from research_agent import search_web

        config = {
            "trusted_domains": {"cat": ["a.sk", "b.sk", "c.sk"]},
            "excluded_domains": [],
            "search_settings": {
                "max_results": 10,
                "search_depth": "advanced",
                "topic": "news",
                "min_sources_for_verdict": 2,
            },
        }
        results = search_web("test statement", config)
        assert len(results) == 3
        # search should have been called only once (Phase 1)
        assert mock_client.search.call_count == 1

    @patch("research_agent.TavilyClient")
    @patch("research_agent.os.getenv", return_value="fake-key")
    def test_phase2_triggered_when_phase1_insufficient(self, mock_env, mock_tavily_cls):
        """When Phase 1 returns <2 results, Phase 2 is called."""
        mock_client = MagicMock()
        mock_tavily_cls.return_value = mock_client
        mock_client.search.side_effect = [
            {"results": [{"url": "https://a.sk/1", "title": "A", "content": "c", "score": 0.9}]},
            {"results": [{"url": "https://d.com/2", "title": "D", "content": "c", "score": 0.8}]},
        ]

        from research_agent import search_web

        config = {
            "trusted_domains": {"cat": ["a.sk"]},
            "excluded_domains": ["rt.com"],
            "search_settings": {
                "max_results": 10,
                "search_depth": "advanced",
                "topic": "news",
                "min_sources_for_verdict": 2,
            },
        }
        results = search_web("test statement", config)
        assert len(results) == 2
        assert mock_client.search.call_count == 2


# ---------------------------------------------------------------------------
# generate_search_queries
# ---------------------------------------------------------------------------


class TestGenerateSearchQueries:
    def test_returns_multiple_queries(self):
        """LLM returns valid JSON array of queries."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps([
            "slovensko HDP EÚ ranking",
            "Slovakia GDP per capita EU",
        ])
        mock_client.chat.completions.create.return_value = mock_response

        config = load_config()
        queries = generate_search_queries(
            "slovensko je najbohatší štát v EU", mock_client, config
        )
        assert len(queries) == 2
        assert queries[0] == "slovensko HDP EÚ ranking"
        assert queries[1] == "Slovakia GDP per capita EU"

    def test_strips_markdown_fences(self):
        """Handles markdown code fences around JSON."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            '```json\n["query one", "query two"]\n```'
        )
        mock_client.chat.completions.create.return_value = mock_response

        config = load_config()
        queries = generate_search_queries("test", mock_client, config)
        assert queries == ["query one", "query two"]

    def test_fallback_on_invalid_json(self):
        """Falls back to raw statement when LLM returns garbage."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json"
        mock_client.chat.completions.create.return_value = mock_response

        config = load_config()
        queries = generate_search_queries("test výrok", mock_client, config)
        assert queries == ["test výrok"]

    def test_fallback_on_api_error(self):
        """Falls back to raw statement when LLM call fails."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")

        config = load_config()
        queries = generate_search_queries("test výrok", mock_client, config)
        assert queries == ["test výrok"]

    def test_caps_at_five_queries(self):
        """Limits output to 5 queries maximum."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            [f"query {i}" for i in range(10)]
        )
        mock_client.chat.completions.create.return_value = mock_response

        config = load_config()
        queries = generate_search_queries("test", mock_client, config)
        assert len(queries) == 5

    def test_fallback_on_empty_list(self):
        """Falls back when LLM returns empty array."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[]"
        mock_client.chat.completions.create.return_value = mock_response

        config = load_config()
        queries = generate_search_queries("test výrok", mock_client, config)
        assert queries == ["test výrok"]


# ---------------------------------------------------------------------------
# search_web with multiple queries
# ---------------------------------------------------------------------------


class TestSearchWebMultiQuery:
    @patch("research_agent.TavilyClient")
    @patch("research_agent.os.getenv", return_value="fake-key")
    def test_multiple_queries_deduplicates(self, mock_env, mock_tavily_cls):
        """Multiple queries merge results and deduplicate by URL."""
        mock_client = MagicMock()
        mock_tavily_cls.return_value = mock_client
        mock_client.search.side_effect = [
            {"results": [
                {"url": "https://eurostat.ec.europa.eu/1", "title": "A", "content": "c"},
                {"url": "https://sme.sk/1", "title": "B", "content": "c"},
            ]},
            {"results": [
                {"url": "https://eurostat.ec.europa.eu/1", "title": "A", "content": "c"},
                {"url": "https://dennikn.sk/1", "title": "C", "content": "c"},
            ]},
        ]

        config = {
            "trusted_domains": {"cat": ["eurostat.ec.europa.eu", "sme.sk", "dennikn.sk"]},
            "excluded_domains": [],
            "search_settings": {
                "max_results": 10,
                "search_depth": "advanced",
                "topic": "general",
                "min_sources_for_verdict": 2,
            },
        }
        results = search_web(
            "test", config,
            queries=["slovensko HDP EÚ", "Slovakia GDP EU"],
        )
        assert len(results) == 3
        urls = [r["url"] for r in results]
        assert len(urls) == len(set(urls))

    @patch("research_agent.TavilyClient")
    @patch("research_agent.os.getenv", return_value="fake-key")
    def test_none_queries_uses_statement(self, mock_env, mock_tavily_cls):
        """When queries=None, falls back to using statement as query."""
        mock_client = MagicMock()
        mock_tavily_cls.return_value = mock_client
        mock_client.search.return_value = {
            "results": [
                {"url": "https://a.sk/1", "title": "A", "content": "c"},
                {"url": "https://b.sk/1", "title": "B", "content": "c"},
            ]
        }

        config = {
            "trusted_domains": {"cat": ["a.sk", "b.sk"]},
            "excluded_domains": [],
            "search_settings": {
                "max_results": 10,
                "search_depth": "advanced",
                "topic": "general",
                "min_sources_for_verdict": 2,
            },
        }
        results = search_web("raw statement", config, queries=None)
        assert len(results) == 2
        assert mock_client.search.call_count == 1
