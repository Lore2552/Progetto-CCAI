import argparse
import json
from collections import Counter, defaultdict
from statistics import mean, median

import chromadb


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_preview(text: str, max_chars: int = 1000) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text[:max_chars] + (" [...continua...]" if len(text) > max_chars else "")


def detect_collection(client, preferred=None):
    collections = [c.name for c in client.list_collections()]

    if preferred:
        if preferred not in collections:
            raise ValueError(
                f"Collection '{preferred}' non trovata. Disponibili: {collections}"
            )
        return preferred, collections

    if "ricette_giallozafferano" in collections:
        return "ricette_giallozafferano", collections

    if "archivio_ricette" in collections:
        return "archivio_ricette", collections

    if collections:
        return collections[0], collections

    raise ValueError("Nessuna collection trovata in ChromaDB.")


def search_recipes_by_keyword(ids, metadatas, documents, keyword, search_mode="title"):
    print_section(f"RICERCA KEYWORD: '{keyword}'")

    keyword_lower = keyword.lower()
    matches = []

    for i in range(len(ids)):
        meta = metadatas[i] if metadatas and metadatas[i] else {}
        doc = documents[i] if documents and documents[i] else ""

        titolo = meta.get("titolo", "")
        url = meta.get("url", "")

        if search_mode == "title":
            searchable_text = titolo.lower()
        elif search_mode == "document":
            searchable_text = doc.lower()
        else:
            searchable_text = f"{titolo} {doc}".lower()

        if keyword_lower in searchable_text:
            matches.append(
                {
                    "id": ids[i],
                    "titolo": titolo,
                    "url": url,
                    "doc_length": len(doc),
                    "preview": safe_preview(doc, 800),
                }
            )

    print(f"Modalità ricerca: {search_mode}")
    print(f"Trovati {len(matches)} risultati.\n")

    for idx, recipe in enumerate(matches, start=1):
        print("-" * 80)
        print(f"[{idx}] {recipe['titolo']}")
        print(f"ID: {recipe['id']}")
        print(f"URL: {recipe['url']}")
        print(f"Lunghezza documento: {recipe['doc_length']} caratteri")
        print("\nPreview:")
        print(recipe["preview"])
        print()

    if not matches:
        print("Nessun risultato trovato.")


def analyze_metadata(metadatas):
    all_keys = set()
    key_counter = Counter()
    value_examples = defaultdict(list)

    for meta in metadatas:
        if not meta:
            continue

        for key, value in meta.items():
            all_keys.add(key)
            key_counter[key] += 1

            if len(value_examples[key]) < 5 and value not in value_examples[key]:
                value_examples[key].append(value)

    print_section("1. CAMPI METADATA TROVATI")
    print(sorted(all_keys))

    print_section("2. FREQUENZA CAMPI METADATA")
    for key, count in key_counter.most_common():
        print(f"{key}: {count}/{len(metadatas)}")

    print_section("3. ESEMPI VALORI METADATA")
    for key in sorted(value_examples.keys()):
        print(f"\n{key}:")
        for value in value_examples[key]:
            print(f"  - {value}")


def analyze_documents(documents):
    lengths = [len(doc) for doc in documents if doc]

    print_section("4. STATISTICHE DOCUMENTI")

    if not lengths:
        print("Nessun documento testuale trovato.")
        return

    print(f"Numero documenti non vuoti: {len(lengths)}")
    print(f"Lunghezza minima: {min(lengths)} caratteri")
    print(f"Lunghezza massima: {max(lengths)} caratteri")
    print(f"Lunghezza media: {mean(lengths):.2f} caratteri")
    print(f"Lunghezza mediana: {median(lengths):.2f} caratteri")


def guess_title_fields(metadatas):
    possible_title_keys = [
        "title",
        "titolo",
        "name",
        "nome",
        "recipe_name",
        "ricetta",
    ]

    available_keys = set()
    for meta in metadatas:
        if meta:
            available_keys.update(meta.keys())

    return [k for k in possible_title_keys if k in available_keys]


def analyze_duplicates(metadatas, documents):
    print_section("5. ANALISI DUPLICATI / POSSIBILI VERSIONI")

    title_keys = guess_title_fields(metadatas)

    if title_keys:
        print(f"Possibili campi titolo trovati: {title_keys}")
        title_key = title_keys[0]

        titles = [
            meta.get(title_key) for meta in metadatas if meta and meta.get(title_key)
        ]

        title_counter = Counter(titles)
        duplicates = {
            title: count for title, count in title_counter.items() if count > 1
        }

        print(f"Numero titoli totali: {len(titles)}")
        print(f"Numero titoli unici: {len(title_counter)}")
        print(f"Titoli duplicati: {len(duplicates)}")

        if duplicates:
            print("\nEsempi titoli duplicati:")
            for title, count in list(duplicates.items())[:20]:
                print(f"- {title}: {count}")
        else:
            print("Nessun duplicato esatto di titolo trovato.")

    else:
        print("Nessun campo titolo evidente nei metadata.")
        print(
            "Provo una stima dei duplicati usando i primi 120 caratteri del documento."
        )

        prefixes = [safe_preview(doc, 120) for doc in documents if doc]

        prefix_counter = Counter(prefixes)
        duplicates = {
            prefix: count for prefix, count in prefix_counter.items() if count > 1
        }

        print(f"Possibili duplicati da prefisso documento: {len(duplicates)}")

        for prefix, count in list(duplicates.items())[:10]:
            print(f"- {prefix}: {count}")


def analyze_recipe_specific_fields(metadatas, documents):
    print_section("6. CAMPI UTILI PER IL KNOWLEDGE GRAPH")

    expected_groups = {
        "titolo": ["title", "titolo", "name", "nome", "recipe_name", "ricetta"],
        "chef/autore": ["chef", "author", "autore", "creator", "created_by"],
        "categoria": ["category", "categoria", "course", "portata"],
        "url/fonte": ["url", "source", "fonte", "link"],
        "difficoltà": ["difficulty", "difficolta", "difficoltà"],
        "tempo": ["time", "tempo", "prep_time", "cook_time", "total_time"],
        "porzioni": ["servings", "porzioni", "dosi"],
        "ingredienti": ["ingredients", "ingredienti"],
        "procedimento": ["procedure", "procedimento", "preparation", "preparazione"],
    }

    available_keys = set()
    for meta in metadatas:
        if meta:
            available_keys.update(meta.keys())

    for label, candidates in expected_groups.items():
        found = [c for c in candidates if c in available_keys]
        if found:
            print(f"✓ {label}: {found}")
        else:
            print(f"✗ {label}: non trovato nei metadata")

    print("\nControllo parole chiave nei documenti testuali:")

    keywords = [
        "ingredienti",
        "preparazione",
        "procedimento",
        "consigli",
        "conservazione",
        "varianti",
        "difficoltà",
        "tempo",
        "dosi",
    ]

    joined_sample = "\n".join(doc.lower() for doc in documents[:50] if doc)

    for kw in keywords:
        if kw.lower() in joined_sample:
            print(f"✓ '{kw}' presente nei testi")
        else:
            print(f"✗ '{kw}' non rilevato nei primi documenti")


def print_sample_records(ids, metadatas, documents, n=5, preview_chars=1200):
    print_section(f"7. ESEMPI RECORD — primi {n}")

    total = min(n, len(ids))

    for i in range(total):
        print("\n" + "-" * 80)
        print(f"RECORD {i + 1}")
        print("-" * 80)

        print(f"ID: {ids[i]}")

        meta = metadatas[i] if metadatas and metadatas[i] else {}
        print("\nMETADATA:")
        print(json.dumps(meta, indent=2, ensure_ascii=False))

        doc = documents[i] if documents and documents[i] else ""
        print(f"\nDOCUMENT LENGTH: {len(doc)} caratteri")
        print("\nDOCUMENT PREVIEW:")
        print(safe_preview(doc, preview_chars))


def export_summary(ids, metadatas, documents, output_path):
    summary = []

    for i in range(len(ids)):
        meta = metadatas[i] if metadatas and metadatas[i] else {}
        doc = documents[i] if documents and documents[i] else ""

        summary.append(
            {
                "id": ids[i],
                "metadata": meta,
                "document_length": len(doc),
                "document_preview": safe_preview(doc, 500),
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_section("8. EXPORT")
    print(f"Summary esportato in: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Ispeziona una collection ChromaDB di ricette."
    )

    parser.add_argument(
        "--db-path",
        default="./chroma_db",
        help="Path della directory ChromaDB. Default: ./chroma_db",
    )

    parser.add_argument(
        "--collection",
        default=None,
        help="Nome collection da aprire. Se omesso prova ricette_giallozafferano, poi archivio_ricette.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Numero di record di esempio da stampare. Default: 5",
    )

    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1200,
        help="Numero di caratteri da mostrare per ogni documento. Default: 1200",
    )

    parser.add_argument(
        "--export",
        default=None,
        help="Path JSON opzionale per esportare un riassunto dei record.",
    )

    parser.add_argument(
        "--search",
        default=None,
        help="Keyword da cercare nei titoli o documenti, esempio: carbonara.",
    )

    parser.add_argument(
        "--search-mode",
        choices=["title", "document", "all"],
        default="title",
        help="Dove cercare la keyword: title, document o all. Default: title.",
    )

    args = parser.parse_args()

    print_section("0. CONNESSIONE A CHROMADB")
    print(f"DB path: {args.db_path}")

    client = chromadb.PersistentClient(path=args.db_path)

    collection_name, collection_names = detect_collection(
        client,
        preferred=args.collection,
    )

    print(f"Collection disponibili: {collection_names}")
    print(f"Collection selezionata: {collection_name}")

    collection = client.get_collection(name=collection_name)

    results = collection.get()

    print("\nChiavi restituite da Chroma:")
    print(results.keys())

    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    print(f"\nRecord totali: {len(ids)}")

    if not ids:
        print("La collection è vuota.")
        return

    if args.search:
        search_recipes_by_keyword(
            ids=ids,
            metadatas=metadatas,
            documents=documents,
            keyword=args.search,
            search_mode=args.search_mode,
        )
        return

    analyze_metadata(metadatas)
    analyze_documents(documents)
    analyze_duplicates(metadatas, documents)
    analyze_recipe_specific_fields(metadatas, documents)
    print_sample_records(
        ids,
        metadatas,
        documents,
        n=args.samples,
        preview_chars=args.preview_chars,
    )

    if args.export:
        export_summary(ids, metadatas, documents, args.export)


if __name__ == "__main__":
    main()
