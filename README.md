# Vachanãmrut RAG

Retrieval over the **Vachanãmrut** — the 273 discourses of Bhagwãn Swãminãrãyan
(1819–1829), in the official English translation — served over **MCP** so that
Claude, Gemini or any MCP client can answer questions about the scripture and
**quote it exactly**, with a citation down to the paragraph.

```
Q: Towards whom does Shriji Mahãrãj not develop a liking, even if He tries?

A: Towards someone who serves Him but has no bhakti for God in his heart.

   Gadhadã I-37.5 — "As for Me, I have no affection towards any of My
   relatives. Moreover, a person may be serving Me, but if there is no
   bhakti for God in his heart, I cannot develop a liking for him – even
   if I try. Even if he is as virtuous as Nãradji, if he lacks bhakti for
   God, I do not like him."
```

## Why paragraph-level citation

The Vachanãmrut numbers its own paragraphs inside numbered discourses, and the
tradition already cites it that way (`Gadhadã I-37.5`). So a paragraph is both
the smallest unit that can be quoted exactly and the unit a reference points
at. This system retrieves paragraphs, never arbitrary text windows, and every
result carries the citation for the text it holds — so the model answering has
the exact words and the exact reference in hand and never has to reconstruct
either from memory.

`verify_quote` closes the loop: it checks wording against the corpus and
rejects paraphrases, so a quotation can be validated before it is presented as
scripture.

## Quick start

```bash
pip install -e .                       # runtime
pip install -e '.[build]'              # + pdfplumber, to rebuild from the PDF

./scripts/build_all.sh                 # PDF -> corpus -> index -> eval  (~2 min)

vachanamrut search "how can one know whether the mind has been conquered"
vachanamrut show "Gadhada Pratham 37"
vachanamrut show "Gadhadã I-37.5"
vachanamrut verify "I have no affection towards any of My relatives"
vachanamrut term vairagya
```

The built corpus is committed, so `build_all.sh` is only needed if you change
the extraction.

## Connecting it to Claude or Gemini

The server speaks MCP over stdio. Point any client at `vachanamrut-mcp`.

**Claude Code**

```bash
claude mcp add vachanamrut -- vachanamrut-mcp
```

**Claude Desktop** — `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vachanamrut": {
      "command": "vachanamrut-mcp",
      "env": { "VACHANAMRUT_CORPUS": "/absolute/path/to/vachanamrut_ai/data/corpus" }
    }
  }
}
```

**Gemini CLI** — `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "vachanamrut": {
      "command": "vachanamrut-mcp",
      "env": { "VACHANAMRUT_CORPUS": "/absolute/path/to/vachanamrut_ai/data/corpus" }
    }
  }
}
```

If you did not `pip install -e .`, use
`"command": "python3", "args": ["-m", "vachanamrut_rag.server"]` with
`PYTHONPATH` set to the repo's `src`.

### Tools the server exposes

| Tool | Purpose |
|---|---|
| `search` | Find paragraphs on a topic. Returns verbatim text + citation + neighbouring paragraphs. |
| `get_discourse` | A whole discourse by reference, in any convention. |
| `get_paragraph` | One numbered paragraph, for an exact quotation. |
| `verify_quote` | Confirm wording appears verbatim; returns the true citation. |
| `lookup_term` | The book's own glossary (vairãgya, ekãntik dharma, swabhãv …). |
| `list_discourses` | Browse or find a discourse by title. |
| `resolve_reference` | Normalise `GADHADA PRATHAM 35` → `Gadhadã I-35`. |

The server also ships `instructions` telling the client model to quote rather
than paraphrase, and to verify when unsure.

### References in any convention

Study questions, the book's own cross-references and everyday usage all spell
references differently. All of these resolve to the same discourse:

```
Gadhadã I-35   Gadhada I-35   GADHADA PRATHAM 35   Gadh-I.35   G.I.35   gadhada 1-35
```

Gujarati ordinals map as `Pratham → I`, `Madhya → II`, `Antya → III`.
Diacritics are optional everywhere, in references and in search queries
(`vairagya` finds `vairãgya`).

## How retrieval works

Questions about this text arrive in three shapes, so three signals are combined:

1. **Reference routing.** Study questions usually name their discourse
   (*"GADHADA PRATHAM 35: … 3. Towards whom does Shriji Maharaj not develop a
   liking?"*). A named discourse is boosted hard, but not made mandatory —
   circulating question papers do misfile questions, so a strong match
   elsewhere can still win.
2. **BM25** over paragraph text, with diacritics folded, an inflection-only
   stemmer that leaves domain terms intact, and extra weight on how many of the
   query's *rare* terms a paragraph covers — so one discriminative word
   ("Bharatji", "Vishalyakarani") is not outvoted by three common ones.
3. **Dense similarity**, when an embedding backend is configured (optional).

Lexical and dense rankings are fused with Reciprocal Rank Fusion. Results come
back with their neighbouring paragraphs, because an answer is usually unusable
without the question it answers, which sits in the paragraph before it.

### Optional dense embeddings

The system is fully functional with **no model download and no API key** — BM25
plus reference routing is what the numbers below measure. Dense retrieval is a
strict upgrade:

```bash
export VACHANAMRUT_EMBEDDINGS=local:BAAI/bge-base-en-v1.5   # offline after first download
export VACHANAMRUT_EMBEDDINGS=voyage:voyage-3               # VOYAGE_API_KEY
export VACHANAMRUT_EMBEDDINGS=openai:text-embedding-3-large # OPENAI_API_KEY
export VACHANAMRUT_EMBEDDINGS=gemini:text-embedding-004     # GEMINI_API_KEY
python3 scripts/build_embeddings.py
```

## Corpus

| | |
|---|---|
| Discourses | 274 — 262 main + 11 Additional (Amdãvãd diocese) + Bhugol-Khagol |
| Paragraphs | 2,837 (all citable) |
| Words | 235,591 |
| Glossary terms | 476 |
| Endnotes | 15 (Appendix A) |
| Retrievable chunks | 3,328 (derived at load, not stored) |

Each discourse carries its title, date (Vikram Samvat *and* Gregorian), place,
speakers, page numbers, footnotes and cross-references. Each paragraph carries
its speaker and whether it is the setting, a question or an answer.

### Extraction fidelity

Verbatim quoting is the whole point, so extraction is done at the character
level using font size and x-position rather than by taking each page's text:

- **Intra-word spaces.** Justified text makes naive extractors produce
  `enti rely`. Character clustering avoids it. A test asserts it stays fixed.
- **Footnotes and endnote markers** interleave with body text in reading order
  and are separated by font size (8.2pt vs 9.8pt), so they cannot splice
  themselves into a paragraph and corrupt a quotation.
- **Paragraph numbers** are 6.8pt glyphs in the left margin, captured as
  structure rather than text.
- **Glossary terms** are separated from definitions by font (Helvetica-Bold),
  not by an x-threshold, which would cut long terms mid-word.

The build validates the result and **fails** rather than shipping a corpus with
holes: all 274 discourses present, section counts matching the canon, paragraph
numbering contiguous in every discourse, no empty paragraphs.

Four typesetting errors in the source PDF are handled explicitly:

| Location | Error in the book |
|---|---|
| Gadhadã I-13 | closing marker opens with a single `\|` |
| Gadhadã I-33 | closing marker omits its running number |
| Gadhadã II-54 | closing marker omits the section name |
| Loyã-12 | heading printed `Loya-12`, without the diacritic |

## Evaluation

22 questions whose answers were read off the text — mostly real Satsang study
questions — scored on whether the discourse containing the answer is retrieved.

| Metric | Lexical only |
|---|---|
| recall@1 | 81.8% |
| recall@3 | 90.9% |
| recall@10 | 100% |
| MRR | 0.875 |

```bash
python3 scripts/evaluate.py -v
```

Ground truth was read from the text rather than taken from the question papers:
two of the sample question sets are filed under the wrong discourse number in
circulating papers (questions labelled *Gadhada Pratham 35* are answered in
**Gadhadã I-37**, and the next set in **Gadhadã I-38**). This is exactly why
reference routing boosts rather than restricts.

## Known limitations

- **Devanagari verses are not in the text.** The PDF has no usable character
  map for them, so they extract as unmapped glyphs and are dropped. Every
  verse's romanised transliteration, English translation and source are kept as
  footnotes, so nothing is lost in meaning — but a handful of paragraphs show an
  empty `‘ ’` where a Gujarati proverb stood.
- **The source PDF is truncated.** It ends mid-Appendix B; Appendix C
  (Cosmogony) and Appendix D (Classification of Hindu Scriptures) are not in the
  file, though the glossary references them.
- **The 11 Additional Vachanãmruts have no titles** — that is how the book
  prints them, not a parsing gap.
- **Retrieval is over the English translation only.** Answers depend on the
  translators' word choices; the Gujarati original is not indexed.

## Layout

```
scripts/     extract_pdf.py  extract_reference.py  build_corpus.py
             build_embeddings.py  evaluate.py  build_all.sh
src/vachanamrut_rag/
             ids.py        reference resolution across conventions
             parse.py      lines -> discourses/paragraphs
             corpus.py     corpus model and chunking
             index.py      BM25 + domain tokenizer
             embed.py      optional dense backends
             retrieve.py   hybrid search, routing, quote verification
             server.py     MCP server
             cli.py        command line access
data/corpus/ discourses.jsonl  glossary.jsonl  endnotes.jsonl  stats.json
             (retrieval chunks and the BM25 index are derived at startup)
tests/       57 tests, plus eval_questions.jsonl
```

## Source text

*The Vachanãmrut: Spiritual Discourses of Bhagwãn Swãminãrãyan* (English
translation), Swãminãrãyan Aksharpith, Amdãvãd, 2001. Copyright in the
translation is held by the publisher; this repository contains tooling for
searching and citing it, and is intended for personal study.
