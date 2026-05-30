import os
import re
from urllib.parse import urlparse

import chromadb
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

CHROMA_DB_PATH = "./chroma_db"
RECIPE_COLLECTION_NAME = "ricette_giallozafferano"

graph_db = Neo4jGraph(
    url=os.environ.get("NEO4J_URI"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD"),
)

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection_ricette = chroma_client.get_collection(name=RECIPE_COLLECTION_NAME)


def parse_recipe_document(document: str) -> dict:
    title_match = re.search(
        r"TITOLO:\s*(.*?)\s*INGREDIENTI:",
        document,
        re.DOTALL | re.IGNORECASE,
    )

    ingredients_match = re.search(
        r"INGREDIENTI:\s*(.*?)\s*PREPARAZIONE:",
        document,
        re.DOTALL | re.IGNORECASE,
    )

    preparation_match = re.search(
        r"PREPARAZIONE:\s*(.*)",
        document,
        re.DOTALL | re.IGNORECASE,
    )

    titolo = title_match.group(1).strip() if title_match else ""
    ingredienti_text = ingredients_match.group(1).strip() if ingredients_match else ""
    preparazione = preparation_match.group(1).strip() if preparation_match else ""

    ingredienti = []

    for line in ingredienti_text.splitlines():
        line = line.strip()
        if line.startswith("-"):
            ingredienti.append(line[1:].strip())

    return {
        "titolo": titolo,
        "ingredienti": ingredienti,
        "preparazione": preparazione,
    }


def normalize_ingredient(raw_ingredient: str) -> str:
    ingredient = raw_ingredient.strip()

    ingredient = re.sub(
        r"\b\d+([,.]\d+)?\s*(g|kg|ml|l|litri|cucchiai|cucchiaio|cucchiaini|cucchiaino)\b",
        "",
        ingredient,
        flags=re.IGNORECASE,
    )

    ingredient = re.sub(r"\bq\.b\.\b", "", ingredient, flags=re.IGNORECASE)
    ingredient = re.sub(r"\([^)]*\)", "", ingredient)
    ingredient = re.sub(r"\s+", " ", ingredient)

    return ingredient.strip(" ,.-")


COOKING_TECHNIQUES = [
    "bollire",
    "lessare",
    "rosolare",
    "soffriggere",
    "friggere",
    "cuocere",
    "infornare",
    "grigliare",
    "marinare",
    "mantecare",
    "impastare",
    "frullare",
    "montare",
    "tagliare",
    "tritare",
    "stufare",
    "saltare",
    "mescolare",
    "affettare",
    "sbucciare",
]


def extract_techniques(preparazione: str) -> list[str]:
    text = preparazione.lower()
    found = []

    for technique in COOKING_TECHNIQUES:
        if technique in text:
            found.append(technique)

    return sorted(set(found))


def create_constraints():
    queries = [
        "CREATE CONSTRAINT recipe_title IF NOT EXISTS FOR (r:Recipe) REQUIRE r.title IS UNIQUE",
        "CREATE CONSTRAINT ingredient_name IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE",
        "CREATE CONSTRAINT technique_name IF NOT EXISTS FOR (t:Technique) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT url_value IF NOT EXISTS FOR (u:Url) REQUIRE u.url IS UNIQUE",
    ]

    for query in queries:
        graph_db.query(query)

    print("[OK] Constraints create/verificate.")


def populate_recipe_kg(start_from: int = 0, limit: int | None = None):
    print("[INFO] Lettura ricette da ChromaDB...")

    results = collection_ricette.get()

    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    total_recipes = len(ids)

    end = total_recipes if limit is None else min(start_from + limit, total_recipes)

    print(f"[INFO] Ricette totali: {total_recipes}")
    print(f"[INFO] Partenza da indice: {start_from}")
    print(f"[INFO] Fine indice: {end}")

    for i in range(start_from, end):
        metadata = {}

        try:
            doc_id = ids[i]
            document = documents[i]
            metadata = metadatas[i] if metadatas and metadatas[i] else {}

            parsed = parse_recipe_document(document)

            titolo = metadata.get("titolo") or parsed["titolo"]
            url = metadata.get("url", "")
            source_name = metadata.get("source", "giallozafferano")
            domain = urlparse(url).netloc if url else ""

            if not titolo:
                print(f"[SKIP] Ricetta senza titolo: {doc_id}")
                continue

            ingredienti = [
                normalize_ingredient(x)
                for x in parsed["ingredienti"]
                if normalize_ingredient(x)
            ]

            tecniche = extract_techniques(parsed["preparazione"])

            graph_db.query(
                """
                MERGE (r:Recipe {title: $title})
                SET r.url = $url,
                    r.source = $source,
                    r.chroma_id = $chroma_id

                MERGE (u:Url {url: $url})
                SET u.domain = $domain,
                    u.source_name = $source

                MERGE (r)-[:HAS_URL]->(u)
                """,
                params={
                    "title": titolo,
                    "url": url,
                    "source": source_name,
                    "domain": domain,
                    "chroma_id": doc_id,
                },
            )

            for ingrediente in ingredienti:
                graph_db.query(
                    """
                    MATCH (r:Recipe {title: $title})
                    MERGE (i:Ingredient {name: $ingredient})
                    MERGE (r)-[:USES_INGREDIENT]->(i)
                    """,
                    params={
                        "title": titolo,
                        "ingredient": ingrediente,
                    },
                )

            for tecnica in tecniche:
                graph_db.query(
                    """
                    MATCH (r:Recipe {title: $title})
                    MERGE (t:Technique {name: $technique})
                    MERGE (r)-[:USES_TECHNIQUE]->(t)
                    """,
                    params={
                        "title": titolo,
                        "technique": tecnica,
                    },
                )

            print(
                f"[OK] {i + 1}/{total_recipes} "
                f"{titolo} "
                f"({len(ingredienti)} ingredienti, {len(tecniche)} tecniche)"
            )

        except Exception as e:
            print(
                f"[ERR] Ricetta indice {i} " f"({metadata.get('titolo', 'N/A')}): {e}"
            )


def print_kg_summary():
    print("\n========== KG SUMMARY ==========")

    queries = {
        "Recipe": "MATCH (n:Recipe) RETURN count(n) AS count",
        "Ingredient": "MATCH (n:Ingredient) RETURN count(n) AS count",
        "Technique": "MATCH (n:Technique) RETURN count(n) AS count",
        "Url": "MATCH (n:Url) RETURN count(n) AS count",
        "Post": "MATCH (n:Post) RETURN count(n) AS count",
        "Topic": "MATCH (n:Topic) RETURN count(n) AS count",
        "Entity": "MATCH (n:Entity) RETURN count(n) AS count",
        "Source": "MATCH (n:Source) RETURN count(n) AS count",
    }

    for label, query in queries.items():
        try:
            res = graph_db.query(query)
            count = res[0]["count"] if res else 0
            print(f"{label}: {count}")
        except Exception as e:
            print(f"{label}: errore ({e})")


if __name__ == "__main__":
    create_constraints()

    populate_recipe_kg(start_from=2711)

    print_kg_summary()
