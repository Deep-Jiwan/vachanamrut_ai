"""The MCP tools are the contract with Claude/Gemini; they must stay stable."""
import asyncio

from vachanamrut_rag import server as srv

EXPECTED_TOOLS = {
    "search", "get_discourse", "get_paragraph", "verify_quote",
    "lookup_term", "list_discourses", "resolve_reference",
}


def test_all_tools_are_registered():
    tools = asyncio.run(srv.server.list_tools())
    assert {t.name for t in tools} == EXPECTED_TOOLS


def test_every_tool_documents_itself():
    """Descriptions are how the model decides which tool to call."""
    for tool in asyncio.run(srv.server.list_tools()):
        assert tool.description and len(tool.description) > 40, tool.name


def test_instructions_state_the_citation_rule():
    text = srv.INSTRUCTIONS
    assert "verify_quote" in text
    assert "Gadhadã I-37.5" in text
    assert "never paraphrase" in text.lower()


def test_search_returns_citations_and_text():
    payload = srv.search("how can one know whether the mind has been conquered", limit=3)
    assert payload["results"]
    for result in payload["results"]:
        assert result["citation"] and result["text"]


def test_get_discourse_accepts_a_gujarati_ordinal():
    payload = srv.get_discourse("Gadhada Pratham 37")
    assert payload["citation"] == "Gadhadã I-37"
    assert len(payload["paragraphs"]) == 11
    assert payload["paragraphs"][0]["citation"] == "Gadhadã I-37.1"


def test_get_discourse_explains_an_unresolvable_reference():
    payload = srv.get_discourse("Book of Genesis 3")
    assert "error" in payload and "hint" in payload


def test_get_paragraph_requires_a_paragraph_number():
    assert "error" in srv.get_paragraph("Gadhadã I-37")
    assert srv.get_paragraph("Gadhadã I-37.5")["citation"] == "Gadhadã I-37.5"


def test_verify_quote_round_trip():
    quote = srv.get_paragraph("Gadhadã I-1.6")["text"].strip("“”")
    assert srv.verify_quote(quote)["verbatim"] is True


def test_list_discourses_filters_by_section_and_title():
    assert srv.list_discourses(section="Panchãlã")["count"] == 7
    found = srv.list_discourses(title_contains="Balance Sheet")
    assert [d["citation"] for d in found["discourses"]] == ["Gadhadã I-38"]


def test_resolve_reference_normalises_exam_paper_style():
    payload = srv.resolve_reference("GADHADA PRATHAM 35")
    assert payload["references"][0]["canonical"] == "Gadhadã I-35"
