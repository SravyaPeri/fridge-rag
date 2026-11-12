from __future__ import annotations

import time

import requests
import streamlit as st

from fridge.core import ROOT, Index, LocalEncoder, relevant_recipes, save_upload
from fridge.generation import SAFETY, models, recommend

st.set_page_config(page_title="Fridge Notes | Local RAG Kitchen", page_icon="🥑", layout="wide")
st.markdown("""
<style>
.stApp {background: #faf8f3}
h1,h2,h3 {color: #204e3a}
div[data-testid="stMetric"] {background: #eaf1e7; padding: 18px; border-radius: 14px}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 16px}
.eyebrow {color:#73836a;letter-spacing:3px;font-size:12px;font-weight:700}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="eyebrow">LOCAL FIRST · EVIDENCE ALWAYS</div>', unsafe_allow_html=True)
st.title("🥑 Fridge Notes")
st.subheader("A little less food waste. A little more dinner.")
st.write("Tell me what you have. I’ll search real recipes, show what’s missing, and keep the sources on the table.")


@st.cache_resource
def encoder() -> LocalEncoder:
    return LocalEncoder()


index = Index(ROOT / "data/recipes.sqlite3")
with st.sidebar:
    st.header("Your local kitchen")
    st.caption("HuggingFace embeddings → SQLite → hybrid retrieval → Ollama")
    stats = index.stats()
    st.metric("Recipes indexed", stats["recipes"])
    st.caption(f"{stats['chunks']} searchable chunks · all-MiniLM-L6-v2 · 384 dimensions")
    try:
        installed = models()
        st.success("Ollama connected")
    except (requests.RequestException, ValueError, KeyError) as error:
        installed = []
        st.warning("Ollama unavailable — retrieval still works")
        st.caption(str(error))
    model = st.selectbox("Local answer model", installed) if installed else None
    if st.button("Build / refresh recipe index", type="primary", use_container_width=True):
        try:
            with st.spinner("Loading local embeddings and indexing recipes…"):
                report = index.sync(ROOT / "recipes", ROOT / "data/uploads", encoder())
            st.session_state["index_report"] = report
            st.session_state.pop("search", None)
            st.rerun()
        except Exception as error:
            st.error(f"Index build failed: {error}")
    if "index_report" in st.session_state:
        st.caption(f"Last sync: {st.session_state['index_report']}")
    with st.expander("Add your own recipe"):
        st.caption("UTF-8 .txt or .md, max 256 KB. One recipe per file. Use bare ingredient names.")
        st.code("# Recipe title\n## Ingredients\n- egg\n- rice\n## Steps\n1. Cook thoroughly.")
        upload = st.file_uploader("Recipe file", type=["md", "txt"])
        if st.button("Import and refresh", disabled=upload is None):
            try:
                save_upload(upload.name, upload.getvalue(), ROOT / "data/uploads")
                with st.spinner("Updating recipe index…"):
                    report = index.sync(ROOT / "recipes", ROOT / "data/uploads", encoder())
                st.session_state["index_report"] = report
                st.session_state.pop("search", None)
                st.rerun()
            except Exception as error:
                st.error(f"Import/index failed: {error}. A valid upload is retained for retry.")
    st.caption("Bundled recipes are original demo content, not scraped cookbooks.")

examples = [
    "Half a can of black beans, stale tortillas, and a sad lime",
    "I have chickpeas, tomato and spinach",
    "Peanut butter, noodles, soy sauce and a carrot",
]
st.caption("START WITH A FRIDGE CHALLENGE")
for column, example in zip(st.columns(3), examples):
    if column.button(example, use_container_width=True):
        st.session_state["query"] = example

with st.form("fridge"):
    query = st.text_area("What’s in your fridge?", key="query",
                         placeholder="A lonely potato, two eggs, half an onion…", max_chars=2000)
    submitted = st.form_submit_button("Find my dinner →", type="primary")
if submitted:
    st.session_state.pop("search", None)
    st.session_state.pop("answer", None)
    try:
        started = time.perf_counter()
        with st.spinner("Embedding your fridge and retrieving recipe evidence…"):
            results = index.search(query, encoder())
        st.session_state["search"] = (query, results, time.perf_counter() - started)
    except Exception as error:
        st.error(f"Retrieval failed: {error}")

if "search" in st.session_state:
    searched_query, results, elapsed = st.session_state["search"]
    relevant = relevant_recipes(results)
    relevant_ids = {r["id"] for r in relevant}
    other_results = [r for r in results if r["id"] not in relevant_ids]
    st.divider()
    a, b, c = st.columns(3)
    a.metric("Relevant recipe matches", len(relevant))
    b.metric("Retrieval time", f"{elapsed:.2f}s")
    c.metric("Answer mode", "Local LLM" if model else "Retrieval only")
    st.caption(f"Results for: {searched_query}")
    st.caption("Recognized ingredient names: " + ((", ".join(results[0]["recognized"]) or "none") if results else "none"))
    if results and results[0]["corrections"]:
        st.info("Ingredient spelling interpreted as: " + "; ".join(results[0]["corrections"]))
    st.info("Missing means not mentioned, not definitely absent. Quantities, negation, allergies and dietary restrictions are not validated.")
    if not results:
        st.warning("No indexed recipes. Build the index from the sidebar first.")
    else:
        if not relevant:
            st.warning("No relevant recipe matches in this collection. Add a recipe with your ingredients using the sidebar, then refresh the index.")
        st.header("1 · Recipes matching your query")
        st.caption("Potential matches for this search, not past searches or saved history. Check unmentioned ingredients before cooking.")
        for result in relevant:
            with st.container(border=True):
                st.subheader(result["title"])
                st.caption(f"[{result['id']}] · {result['source']} · hybrid {result['score']:.3f}")
                left, right = st.columns(2)
                left.write("**You mentioned:** " + (", ".join(result["matched"]) or "No recognized overlap"))
                right.write("**Not mentioned:** " + (", ".join(result["missing"]) or "All non-pantry ingredient names matched"))
                with st.expander("Source recipe & retrieval trace"):
                    st.text(result["text"])
                    st.caption(f"Semantic cosine {result['semantic']:.3f} · lexical overlap {result['lexical']:.3f} · ingredient coverage {result['coverage']:.0%}")
                    st.caption("Score = 55% semantic + 30% ingredient coverage + 15% lexical. Scores are not probabilities.")
                    st.text("Best evidence chunk:\n" + result["evidence"])
        if other_results:
            with st.expander(f"Other search results — not recommendations ({len(other_results)})", expanded=False):
                st.caption("Retrieval debug view: weaker candidates from this query, not search history. They were excluded from the main list and will not be sent to Ollama.")
                for result in other_results:
                    st.write(f"**{result['title']}**")
                    st.caption(f"hybrid {result['score']:.3f} · matched ingredients: {', '.join(result['matched']) or 'none'} · source: {result['source']} [{result['id']}]")
                    st.text(result["evidence"])
        st.header("2 · Let the local chef choose")
        st.caption("The model selects source IDs and explains its choice. Ingredients and cooking steps below are copied from the sources, never generated.")
        if st.button("Ask Ollama for a grounded answer", disabled=model is None or not relevant):
            st.session_state.pop("answer", None)
            try:
                with st.spinner("Your local model is reading the evidence…"):
                    answer = recommend(searched_query, results, model)
                st.session_state["answer"] = answer
            except Exception as error:
                st.error(f"Generation failed; no substitute answer was generated: {error}")
        if "answer" in st.session_state:
            if not st.session_state["answer"]:
                st.warning("No supported recommendation. Try a different query or add a relevant recipe.")
            for item in st.session_state["answer"]:
                recipe = item["recipe"]
                with st.container(border=True):
                    st.subheader(f"{recipe['title']} {item['citation']}")
                    st.text(item["reason"])
                    st.caption("AI-generated rationale; source ID verified. Rationale is not independently fact-checked.")
                    st.write("**Ingredients (source):** " + ", ".join(recipe["ingredients"]))
                    st.write("**Not mentioned:** " + (", ".join(recipe["missing"]) or "None, excluding pantry staples"))
                    st.text(recipe["steps"])
                    st.caption(f"Source: {recipe['source']} {item['citation']}")
st.divider()
st.caption(SAFETY)
