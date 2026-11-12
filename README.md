# Fridge Notes

Find recipes from the ingredients you have. Fridge Notes searches a small collection of recipe files, shows which ingredients match, and uses a local Ollama model to choose from the retrieved recipes.

For example, try “black beans, tortillas and lime” or “paneer and green peas”. Each result links back to its source recipe and shows ingredients you haven't mentioned.

## Run locally

You'll need Python 3.11+ and, for generated recommendations, [Ollama](https://ollama.com/) with a model installed. The first run downloads the Hugging Face embedding model; subsequent runs can use the local cache.

```sh
cd fridge-rag
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m fridge.cli index
./run.sh
```

Open **http://127.0.0.1:8501**. The app binds to loopback only. To change the port: `PORT=8502 ./run.sh`.

For grounded answers, run [Ollama](https://ollama.com/) locally, with a completion model available:

```sh
ollama list
# If needed, start Ollama using its app or `ollama serve`.
# If no model is installed, an optional small model is:
ollama pull qwen2.5:0.5b
```

Choose an installed model in the sidebar. The CLI defaults to `llama3.1:latest`; use `--model qwen2.5:0.5b` if you installed the smaller model above. Recipe search works without Ollama, but generated recommendations require it.

### Try it

1. Open the app and click the black-bean example, then **Find my dinner**.
2. Inspect the skillet's matched ingredients, score components and exact source evidence.
3. Click **Ask Ollama for a grounded answer**. The local model selects source recipe IDs; the app verifies those IDs and copies the original ingredients and steps.
4. Upload a recipe in the documented format and refresh. Repeated imports do not multiply chunks.

The main list shows candidates that pass the same relevance filter used for Ollama. Weaker candidates are available in the collapsed **Other search results** section for inspection; they are not recommendations or search history.

Equivalent terminal demo:

```sh
.venv/bin/python -m fridge.cli query --generate
.venv/bin/python -m fridge.cli query --query "Peanut butter noodles soy sauce carrot lime"
```

## Architecture

```text
Recipe files → chunks → Hugging Face embeddings → SQLite
Fridge query → semantic + lexical + ingredient scoring → relevant recipes
             → Ollama selects source IDs → validated citations and source steps
```

SQLite stores float32 vector blobs and source metadata, with atomic index updates and cascading chunk deletion. Cosine similarity is computed with NumPy over the small local corpus; there is no separate vector database service. At larger scale, replace the full vector scan with a vector index.

Hybrid score = `0.55 × positive cosine + 0.30 × ingredient coverage + 0.15 × lexical overlap`. Coverage is the fraction of a recipe's non-pantry ingredient names recognized in the query; lexical overlap is the fraction of query word types present in the recipe. A small explicit alias map handles common plurals, including peas / green peas. The known typo “green peak” is interpreted as “green peas” before retrieval, with that interpretation disclosed in the UI; this is not general-purpose spell correction. Scores are ranking signals, **not confidence probabilities**. Pantry exclusions: salt, black pepper, water, oil, olive oil.

Ollama receives retrieved titles and ingredient metadata, then selects source IDs. The app checks those IDs and displays ingredients and steps directly from the source rather than generating them. The model's short explanations are not independently fact-checked. With no ingredient overlap and cosine below 0.40, generation abstains; candidates below half the best ingredient coverage are also excluded when there is overlap. These are heuristics, not calibrated confidence scores.

## Add recipes

Bundle: `recipes/`. Imports: `data/uploads/`. The sidebar accepts UTF-8 `.md` / `.txt`, one recipe per file, up to 256,000 bytes. Ingredient names must be bare names (quantities belong in the steps) for reliable matching:

```markdown
# My rice skillet
## Ingredients
- rice
- egg
- onion
- olive oil
## Steps
1. Cook half a cup of rice according to its package.
2. Soften chopped onion in olive oil.
3. Add rice and one egg, stirring until the egg is fully cooked.
```

The importer validates content before saving it and uses content-addressed filenames. Refresh the index after changing or deleting recipe files; changed recipes are re-embedded and removed ones are deleted from the index. Valid uploads remain on disk if indexing fails.

To remove an imported recipe, delete its specific file from `data/uploads/` and refresh the index. Single-user operation is intended: avoid concurrent refreshes or editing corpus files during a refresh.

## Validation and retrieval evaluation

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m fridge.cli evaluate
```

The tests cover parser/corpus validity, incremental update/delete, failed embedding rollback, model mismatch, quantity-tolerant matching, safe imports, overlap chunking, abstention, malformed/invented model sources and initial Streamlit rendering. Unit retrieval tests use an explicitly fake encoder for fast deterministic checks; the evaluation uses **real HuggingFace embeddings**.

`evaluation.json` contains 18 queries with expected recipes, including paraphrases and spelling regressions. On the bundled corpus, the most recent run returned **16/18 at rank 1**, **18/18 in the top 3**, and **0.935 MRR@3**. The report, including individual ranks, is written to `data/evaluation-report.json`. This small in-corpus check measures retrieval, not answer quality or general cooking expertise.

## Provenance, privacy and limitations

- The 13 bundled recipes are original illustrative demo content, not scraped cookbooks, and have not been professionally tested. The collection is small; add recipes to cover more ingredients and cuisines.
- Recipe text, embeddings and generated answers stay on this machine. The first model download contacts HuggingFace; Ollama model pulls contact its registry. Streamlit usage collection is disabled by the launch script.
- This is **not** an allergy, diet, storage-safety or nutrition assistant. Quantities, negation (“no peanuts”), substitutions, cooking equipment and dietary constraints are not validated. Missing means “not mentioned,” not “not in your kitchen.” Check every source recipe yourself.
- Stale is a quality description, not a safety guarantee. Discard moldy tortillas or bread. Refrigerate opened canned beans promptly and use within 3-4 days. Never use food you suspect is spoiled.
- Prompt instructions describe user queries and retrieved records as untrusted data; structural validation enforces source membership, not semantic truth of generated rationale. Prompt injection cannot be eliminated by a prompt alone.
- No PDF ingestion, OCR, scraping or automatic substitutions. Retrieval uses a full SQLite vector scan and is intended for a small, single-user collection.
- First embedding load may take time; CPU generation with an 8B model can take a minute or more. If Ollama times out or returns invalid structured output the UI reports that failure and keeps the evidence available.

## Project map

`app.py` is the Streamlit interface; `fridge/core.py` handles ingestion/indexing/retrieval; `fridge/generation.py` is the Ollama boundary and source validation; `fridge/cli.py` exposes indexing, query and evaluation; `recipes/` is the original corpus; `tests/` contains automated checks. Generated state is under `data/`, downloads under `.cache/`, and dependencies under `.venv/`.
