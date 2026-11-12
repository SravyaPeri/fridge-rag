from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_FILE_BYTES = 256_000
ALIASES = {
    "tortilla": "tortillas", "black bean": "black beans", "chickpea": "chickpeas",
    "garbanzo beans": "chickpeas", "eggs": "egg", "tomatoes": "tomato",
    "potatoes": "potato", "onions": "onion", "limes": "lime",
    "lemons": "lemon", "scallions": "green onion", "spring onions": "green onion",
    "bell peppers": "bell pepper", "peanut butter": "peanut butter",
    "peas": "green peas", "green pea": "green peas",
}
QUERY_CORRECTIONS = {"green peak": "green peas"}
PANTRY = {"salt", "black pepper", "water", "olive oil", "oil"}


class Encoder(Protocol):
    fingerprint: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


class LocalEncoder:
    fingerprint = MODEL

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL, cache_folder=str(ROOT / ".cache/models"))

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def canonical(text: str) -> str:
    text = re.sub(r"\s+", " ", text.lower().strip())
    return ALIASES.get(text, text)


def mentions(text: str, ingredient: str) -> bool:
    names = {ingredient, *(k for k, v in ALIASES.items() if v == ingredient)}
    return any(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text.lower()) for name in names)


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{2,}", text.lower()))


@dataclass(frozen=True)
class Recipe:
    id: str
    title: str
    ingredients: list[str]
    steps: str
    source: str
    text: str
    digest: str


def parse_recipe(path: Path, source: str) -> Recipe:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"{source}: file exceeds {MAX_FILE_BYTES:,} bytes")
    text = path.read_text(encoding="utf-8").strip()
    if "\x00" in text:
        raise ValueError(f"{source}: binary content is not a recipe")
    match = re.fullmatch(
        r"#?\s*([^\n]+)\n+(?:##?\s*)?Ingredients\s*\n(.*?)\n"
        r"(?:##?\s*)?(?:Steps|Instructions)\s*\n(.+)", text, re.I | re.S,
    )
    if not match:
        raise ValueError(f"{source}: expected a title, Ingredients section, then Steps section")
    title, ingredient_text, steps = match.groups()
    ingredients = [canonical(line.strip().lstrip("-* ").strip())
                   for line in ingredient_text.splitlines() if line.strip()]
    if not ingredients or len(ingredients) > 60 or len(title) > 160:
        raise ValueError(f"{source}: use 1-60 ingredient names and a title under 160 characters")
    if any(len(i) > 100 for i in ingredients):
        raise ValueError(f"{source}: ingredient names must be under 100 characters")
    return Recipe(
        hashlib.sha256(source.encode()).hexdigest()[:12], title.strip(),
        list(dict.fromkeys(ingredients)), steps.strip(), source, text,
        hashlib.sha256(text.encode()).hexdigest(),
    )


def chunks(recipe: Recipe) -> list[str]:
    tokens = recipe.steps.split()
    prefix = f"{recipe.title}\nIngredients: {', '.join(recipe.ingredients)}\n"
    return [prefix + " ".join(tokens[start:start + 160])
            for start in range(0, len(tokens), 130)]


def relevant_recipes(results: list[dict]) -> list[dict]:
    eligible = [r for r in results if r["matched"] or r["semantic"] >= .40]
    best_coverage = max((r["coverage"] for r in eligible), default=0)
    if best_coverage:
        eligible = [r for r in eligible if r["coverage"] >= best_coverage * .5]
    return eligible


class Index:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS recipes (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, ingredients TEXT NOT NULL,
                    steps TEXT NOT NULL, source TEXT UNIQUE NOT NULL, text TEXT NOT NULL,
                    digest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    recipe_id TEXT NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
                    position INTEGER NOT NULL, text TEXT NOT NULL, vector BLOB NOT NULL,
                    PRIMARY KEY (recipe_id, position)
                );
            """)

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def sync(self, corpus: Path, uploads: Path, encoder: Encoder) -> dict[str, int]:
        recipes = []
        for folder, label in ((corpus, "bundled"), (uploads, "uploads")):
            if folder.exists():
                for path in sorted(folder.iterdir()):
                    if path.is_file() and not path.is_symlink() and path.suffix.lower() in {".txt", ".md"}:
                        recipes.append(parse_recipe(path, f"{label}/{path.name}"))
        with self.connect() as db:
            old = {r["id"]: r["digest"] for r in db.execute("SELECT id,digest FROM recipes")}
            model = db.execute("SELECT value FROM metadata WHERE key='model'").fetchone()
        changed = [r for r in recipes if old.get(r.id) != r.digest
                   or model is None or model["value"] != encoder.fingerprint]
        # Embed before the transaction: a download/encoding failure preserves the old index.
        encoded = [(r, chunks(r)) for r in changed]
        vectors = encoder.encode([c for _, cs in encoded for c in cs]) if encoded else []
        deleted = set(old) - {r.id for r in recipes}
        with self.connect() as db:
            for rid in deleted:
                db.execute("DELETE FROM recipes WHERE id=?", (rid,))
            offset = 0
            for recipe, texts in encoded:
                db.execute("DELETE FROM recipes WHERE id=?", (recipe.id,))
                db.execute("INSERT INTO recipes VALUES (?,?,?,?,?,?,?)", (
                    recipe.id, recipe.title, json.dumps(recipe.ingredients),
                    recipe.steps, recipe.source, recipe.text, recipe.digest,
                ))
                for position, text in enumerate(texts):
                    vector = np.asarray(vectors[offset], dtype=np.float32)
                    db.execute("INSERT INTO chunks VALUES (?,?,?,?)",
                               (recipe.id, position, text, vector.tobytes()))
                    offset += 1
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('model',?)", (encoder.fingerprint,))
        return {"recipes": len(recipes), "updated": len(changed), "removed": len(deleted)}

    def stats(self) -> dict[str, int]:
        with self.connect() as db:
            return {name: db.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
                    for name in ("recipes", "chunks")}

    def search(self, query: str, encoder: Encoder, limit: int = 4) -> list[dict]:
        if not query.strip() or len(query) > 2000:
            raise ValueError("Enter a fridge query between 1 and 2,000 characters.")
        corrections = []
        for typo, replacement in QUERY_CORRECTIONS.items():
            query, count = re.subn(rf"(?<!\w){re.escape(typo)}(?!\w)", replacement, query, flags=re.I)
            if count:
                corrections.append(f"{typo} → {replacement}")
        with self.connect() as db:
            model = db.execute("SELECT value FROM metadata WHERE key='model'").fetchone()
            if model is None or model["value"] != encoder.fingerprint:
                raise ValueError("Build the index with the current embedding model first.")
            recipes = list(db.execute("SELECT * FROM recipes"))
            rows = list(db.execute("SELECT * FROM chunks"))
        if not rows:
            return []
        vector = np.asarray(encoder.encode([query])[0], dtype=np.float32)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Embedding model returned an invalid query vector.")
        vector = vector / norm
        semantic: dict[str, tuple[float, str]] = {}
        for row in rows:
            stored = np.frombuffer(row["vector"], dtype=np.float32)
            score = float(np.dot(vector, stored) / max(float(np.linalg.norm(stored)), 1e-12))
            if row["recipe_id"] not in semantic or score > semantic[row["recipe_id"]][0]:
                semantic[row["recipe_id"]] = (score, row["text"])
        vocabulary = {i for r in recipes for i in json.loads(r["ingredients"])} - PANTRY
        available = {i for i in vocabulary if mentions(query, i)}
        # Longest matching names win: "peanut butter" must not imply "butter".
        available = {i for i in available if not any(i != j and i in j for j in available)}
        results = []
        for row in recipes:
            ingredients = json.loads(row["ingredients"])
            required = set(ingredients) - PANTRY
            matched = sorted(required & available)
            missing = sorted(required - available)
            coverage = len(matched) / max(len(required), 1)
            lexical = len(words(query) & words(row["text"])) / max(len(words(query)), 1)
            sem, evidence = semantic[row["id"]]
            score = .55 * max(0, sem) + .30 * coverage + .15 * lexical
            results.append({
                **dict(row), "ingredients": ingredients, "semantic": sem,
                "lexical": lexical, "coverage": coverage, "score": score,
                "matched": matched, "missing": missing, "evidence": evidence,
                "recognized": sorted(available),
                "corrections": corrections,
            })
        return sorted(results, key=lambda r: (-r["score"], r["id"]))[:limit]


def save_upload(name: str, content: bytes, folder: Path) -> Path:
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Upload filename must not contain a path.")
    suffix = Path(name).suffix.lower()
    if suffix not in {".md", ".txt"} or not content or len(content) > MAX_FILE_BYTES:
        raise ValueError("Upload a nonempty .md or .txt file up to 256 KB.")
    content.decode("utf-8")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{hashlib.sha256(content).hexdigest()[:20]}{suffix}"
    if target.is_symlink():
        raise ValueError("Upload destination must not be a symbolic link.")
    temporary = folder / f".{target.name}.pending"
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("A pending upload already exists; retry after checking the upload directory.")
    try:
        with temporary.open("xb") as file:
            file.write(content)
        parse_recipe(temporary, name)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
