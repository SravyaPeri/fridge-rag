import hashlib
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from fridge.core import Index, chunks, parse_recipe, relevant_recipes, save_upload, words
from fridge.generation import recommend


class TestEncoder:
    __test__ = False
    fingerprint = "deterministic-test-encoder"

    def encode(self, texts):
        vectors = np.zeros((len(texts), 128), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in words(text):
                bucket = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 128
                vectors[row, bucket] += 1
            vectors[row] /= max(np.linalg.norm(vectors[row]), 1)
        return vectors


TEXT = """# Bean skillet
## Ingredients
- black beans
- tortillas
- lime
- salt
## Steps
1. Heat the beans with water.
2. Toast tortillas and squeeze over lime.
"""


@pytest.fixture
def kitchen(tmp_path):
    corpus = tmp_path / "recipes"
    corpus.mkdir()
    (corpus / "beans.md").write_text(TEXT)
    uploads = tmp_path / "uploads"
    index = Index(tmp_path / "index.sqlite3")
    encoder = TestEncoder()
    index.sync(corpus, uploads, encoder)
    return index, encoder, corpus, uploads


def test_real_corpus_parses():
    root = Path(__file__).resolve().parents[1]
    recipes = [parse_recipe(path, path.name) for path in (root / "recipes").glob("*.md")]
    assert len(recipes) == 13
    assert all(r.ingredients and chunks(r) for r in recipes)


def test_repeated_ingestion_update_and_delete(kitchen):
    index, encoder, corpus, uploads = kitchen
    assert index.sync(corpus, uploads, encoder) == {"recipes": 1, "updated": 0, "removed": 0}
    path = corpus / "beans.md"
    path.write_text(TEXT.replace("Bean skillet", "New bean skillet"))
    assert index.sync(corpus, uploads, encoder)["updated"] == 1
    assert index.search("beans", encoder)[0]["title"] == "New bean skillet"
    path.unlink()
    assert index.sync(corpus, uploads, encoder)["removed"] == 1
    assert index.stats() == {"recipes": 0, "chunks": 0}


def test_failed_encoding_preserves_index(kitchen):
    index, encoder, corpus, uploads = kitchen
    (corpus / "beans.md").write_text(TEXT + "\nServe hot.")
    with patch.object(encoder, "encode", side_effect=RuntimeError("model failure")):
        with pytest.raises(RuntimeError, match="model failure"):
            index.sync(corpus, uploads, encoder)
    assert index.stats()["recipes"] == 1
    assert "Serve hot." not in index.search("beans", encoder)[0]["text"]


def test_changed_model_rebuilds(kitchen):
    index, encoder, corpus, uploads = kitchen
    encoder.fingerprint = "new-model"
    with pytest.raises(ValueError, match="Build the index"):
        index.search("beans", encoder)
    assert index.sync(corpus, uploads, encoder)["updated"] == 1
    assert index.search("beans", encoder)


def test_ingredient_matching_and_quantities(kitchen):
    index, encoder, _, _ = kitchen
    hit = index.search("half a can of black beans and one tortilla", encoder)[0]
    assert hit["matched"] == ["black beans", "tortillas"]
    assert hit["missing"] == ["lime"]
    assert hit["coverage"] == pytest.approx(2 / 3)
    assert "salt" not in hit["missing"]
    assert hit["source"] == "bundled/beans.md"


@pytest.mark.parametrize("query,corrected", [
    ("Paneer and Green peak", True),
    ("paneer and green peas", False),
    ("paneer and peas", False),
])
def test_paneer_and_peas_retrieval(kitchen, query, corrected):
    index, encoder, corpus, uploads = kitchen
    source = Path(__file__).resolve().parents[1] / "recipes/matar-paneer.md"
    (corpus / source.name).write_text(source.read_text())
    index.sync(corpus, uploads, encoder)
    hit = index.search(query, encoder)[0]
    assert hit["source"] == "bundled/matar-paneer.md"
    assert hit["matched"] == ["green peas", "paneer"]
    assert hit["missing"] == ["cumin", "garlic", "onion", "tomato"]
    assert bool(hit["corrections"]) == corrected


def test_typo_correction_respects_word_boundaries(kitchen):
    index, encoder, _, _ = kitchen
    assert index.search("green peaks", encoder)[0]["corrections"] == []


@pytest.mark.parametrize("query", ["", " " * 3, "a" * 2001])
def test_invalid_queries(kitchen, query):
    index, encoder, _, _ = kitchen
    with pytest.raises(ValueError, match="fridge query"):
        index.search(query, encoder)


@pytest.mark.parametrize("name,content", [
    ("../escape.md", TEXT.encode()), ("/tmp/escape.md", TEXT.encode()),
    ("folder\\escape.txt", TEXT.encode()), ("recipe.pdf", b"pdf"),
    ("recipe.md", b""), ("recipe.md", b"x" * 256001),
    ("recipe.md", b"\xff"), ("recipe.md", b"not a structured recipe"),
])
def test_reject_bad_upload(tmp_path, name, content):
    with pytest.raises((ValueError, UnicodeDecodeError)):
        save_upload(name, content, tmp_path / "uploads")
    assert not list(tmp_path.rglob("*.pending"))


def test_duplicate_upload_and_index(kitchen):
    index, encoder, corpus, uploads = kitchen
    first = save_upload("my-recipe.md", TEXT.encode(), uploads)
    second = save_upload("different-name.md", TEXT.encode(), uploads)
    assert first == second
    assert index.sync(corpus, uploads, encoder)["recipes"] == 2
    assert index.sync(corpus, uploads, encoder)["updated"] == 0


def test_invalid_upload_does_not_replace_valid(tmp_path):
    folder = tmp_path / "uploads"
    saved = save_upload("good.txt", TEXT.encode(), folder)
    with pytest.raises(ValueError):
        save_upload("bad.txt", b"not a recipe", folder)
    assert saved.read_text() == TEXT
    assert len(list(folder.iterdir())) == 1


def test_long_recipe_chunks_overlap(tmp_path):
    path = tmp_path / "long.md"
    path.write_text(TEXT.split("## Steps")[0] + "## Steps\n" + " ".join(f"step{i}" for i in range(400)))
    recipe = parse_recipe(path, "long.md")
    parts = chunks(recipe)
    assert len(parts) == 4
    assert "step130" in parts[0] and "step130" in parts[1]
    assert all("Ingredients: black beans" in part for part in parts)


def model_response(payload):
    response = Mock()
    response.json.return_value = {"message": {"content": json.dumps(payload)}}
    return response


def test_grounded_answer_copies_exact_source(kitchen):
    index, encoder, _, _ = kitchen
    results = index.search("black beans tortillas lime", encoder)
    rid = results[0]["id"]
    response = model_response({"recommendations": [{"recipe_id": rid, "reason": "Matches the beans and tortillas."}]})
    with patch("fridge.generation.requests.post", return_value=response) as post:
        answer = recommend("beans", results)
    assert answer[0]["citation"] == f"[{rid}]"
    assert answer[0]["recipe"]["text"] == TEXT.strip()
    assert answer[0]["recipe"]["steps"] == results[0]["steps"]
    assert post.call_args.kwargs["json"]["options"]["temperature"] == 0


@pytest.mark.parametrize("payload", [
    {"recommendations": [{"recipe_id": "invented", "reason": "Made up"}]},
    {"recommendations": "wrong"},
    {"recommendations": [{"recipe_id": [], "reason": "Bad type"}]},
    {"recommendations": [], "extra": "ignore all rules"},
    {"recommendations": [{"recipe_id": "placeholder", "reason": "See https://fake.example"}]},
])
def test_invalid_generation_fails_explicitly(kitchen, payload):
    index, encoder, _, _ = kitchen
    results = index.search("black beans", encoder)
    if isinstance(payload["recommendations"], list):
        for item in payload["recommendations"]:
            if item.get("recipe_id") == "placeholder":
                item["recipe_id"] = results[0]["id"]
    with patch("fridge.generation.requests.post", return_value=model_response(payload)):
        with pytest.raises(ValueError, match="invalid grounded answer"):
            recommend("beans", results)


def test_no_evidence_abstains_without_call():
    with patch("fridge.generation.requests.post") as post:
        assert recommend("moon dust", []) == []
        assert recommend("moon dust", [{"matched": [], "semantic": .1}]) == []
        post.assert_not_called()


def test_weak_overlap_excluded_from_generation(kitchen):
    index, encoder, _, _ = kitchen
    strongest = index.search("black beans tortillas lime", encoder)[0]
    weak = {**strongest, "id": "weak", "matched": ["lime"], "coverage": .2}
    payload = {"recommendations": [{"recipe_id": "weak", "reason": "Only lime"}]}
    with patch("fridge.generation.requests.post", return_value=model_response(payload)):
        with pytest.raises(ValueError, match="Unknown"):
            recommend("beans tortillas lime", [strongest, weak])


def test_ui_separates_weak_candidates_from_main_list(kitchen):
    from streamlit.testing.v1 import AppTest

    index, encoder, _, _ = kitchen
    strongest = index.search("black beans tortillas lime", encoder)[0]
    unrelated = {**strongest, "id": "unrelated", "title": "Unrelated recipe",
                 "matched": [], "coverage": 0, "semantic": .1}
    with patch.object(index, "search", return_value=[strongest, unrelated]), \
         patch("fridge.core.LocalEncoder", return_value=encoder), \
         patch("fridge.core.Index", return_value=index), \
         patch("fridge.generation.models", return_value=["test-model"]):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        app.text_area[0].set_value("black beans tortillas lime")
        next(b for b in app.button if "Find my dinner" in b.label).click().run()
        assert not app.exception
        assert any(h.value == "Bean skillet" for h in app.subheader)
        assert not any(h.value == "Unrelated recipe" for h in app.subheader)
        debug = next(e for e in app.expander if "Other search results" in e.label)
        assert "not recommendations (1)" in debug.label
        assert any("Unrelated recipe" in m.value for m in debug.markdown)
        assert next(m for m in app.metric if m.label == "Relevant recipe matches").value == "1"
        assert relevant_recipes([strongest, unrelated]) == [strongest]


def test_ui_initial_render():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    assert not app.exception
    assert app.title[0].value == "🥑 Fridge Notes"
    assert any("Find my dinner" in button.label for button in app.button)


def test_ui_query_and_grounded_answer(kitchen):
    from streamlit.testing.v1 import AppTest

    index, encoder, _, _ = kitchen
    # AppTest executes the script under its own module; patch the imported boundaries.
    with patch("fridge.core.LocalEncoder", return_value=encoder), \
         patch("fridge.core.Index", return_value=index), \
         patch("fridge.generation.models", return_value=["test-model"]):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        app.text_area[0].set_value("black beans tortillas lime")
        next(b for b in app.button if "Find my dinner" in b.label).click().run()
        assert not app.exception
        assert any("Bean skillet" in heading.value for heading in app.subheader)
        assert app.session_state["search"][1][0]["matched"] == ["black beans", "lime", "tortillas"]
        rid = app.session_state["search"][1][0]["id"]
        response = model_response({"recommendations": [{"recipe_id": rid, "reason": "Matches the supplied beans."}]})
        with patch("fridge.generation.requests.post", return_value=response):
            next(b for b in app.button if "Ask Ollama" in b.label).click().run()
        assert not app.exception
        assert app.session_state["answer"][0]["citation"] == f"[{rid}]"
        assert any("Toast tortillas" in element.value for element in app.text)
