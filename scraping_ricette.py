import requests
from bs4 import BeautifulSoup
import chromadb
import time
import re
import json
import hashlib
import os

# Configurazione ChromaDB
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "ricette_giallozafferano")

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
chroma_collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

BASE_URL = "https://www.giallozafferano.it"

CATEGORIES = [
    "Antipasti",
    "Primi",
    "Secondi-piatti",
    "Dolci-e-Desserts",
    "Lievitati",
    "Piatti-Unici",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}


def clean_text(text):
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


def _extract_recipe_links_from_html(html):
    """Estrae link ricetta da una pagina categoria."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for script_tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = (script_tag.string or script_tag.get_text() or "").strip()
        if not raw:
            continue

        try:
            data = json.loads(raw)
        except Exception:
            continue

        payloads = data if isinstance(data, list) else [data]

        for payload in payloads:
            if not isinstance(payload, dict):
                continue

            if payload.get("@type") != "ItemList":
                continue

            for item in payload.get("itemListElement", []):
                if not isinstance(item, dict):
                    continue

                recipe_url = item.get("url", "")

                if (
                    isinstance(recipe_url, str)
                    and "ricette.giallozafferano.it" in recipe_url
                ):
                    links.add(recipe_url.replace("http://", "https://"))

    if not links:
        for a in soup.find_all("a", href=True):
            href = a["href"]

            if "ricette.giallozafferano.it" in href and href.endswith(".html"):
                links.add(href.replace("http://", "https://"))

    return links


def get_all_ricette_links():
    """Scorre le categorie e raccoglie link ricetta."""
    all_links = set()

    for category in CATEGORIES:
        print(f"[INFO] Scansione categoria: {category}")

        page = 1
        empty_pages = 0
        previous_page_signature = None

        while True:
            if page == 1:
                url = f"{BASE_URL}/ricette-cat/{category}/"
            else:
                url = f"{BASE_URL}/ricette-cat/page{page}/{category}/"

            try:
                resp = requests.get(url, headers=HEADERS, timeout=20)
            except Exception as e:
                print(f"[ERR] Errore richiesta {url}: {e}")
                break

            if resp.status_code != 200:
                print(
                    f"[INFO] Fine pagine per {category}: {url} status {resp.status_code}"
                )
                break

            page_links = _extract_recipe_links_from_html(resp.text)

            signature = hashlib.md5(
                "|".join(sorted(page_links)).encode("utf-8")
            ).hexdigest()

            if (
                previous_page_signature is not None
                and signature == previous_page_signature
            ):
                print(f"[INFO] Stop {category}: pagina {page} replica la precedente.")
                break

            previous_page_signature = signature

            if not page_links:
                empty_pages += 1
                print(f"[INFO] Nessuna ricetta in {category} pagina {page}.")

                if empty_pages >= 2:
                    break
            else:
                empty_pages = 0
                before = len(all_links)
                all_links.update(page_links)
                added = len(all_links) - before

                print(
                    f"[INFO] {category} pagina {page}: "
                    f"trovati {len(page_links)} link ({added} nuovi)"
                )

            page += 1
            time.sleep(1)

    return sorted(all_links)


def get_existing_recipe_ids():
    """Legge tutti gli ID già presenti nella collection."""
    try:
        return set(chroma_collection.get().get("ids", []))
    except Exception:
        return set()


def get_existing_recipe_urls():
    """Legge gli URL già presenti nei metadati."""
    urls = set()

    try:
        metadatas = chroma_collection.get().get("metadatas", [])

        for meta in metadatas:
            if isinstance(meta, dict):
                url = meta.get("url")

                if isinstance(url, str) and url.strip():
                    urls.add(url.strip())

    except Exception:
        pass

    return urls


def estrai_ricetta(url):
    """Estrae titolo, ingredienti e preparazione da una ricetta."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
    except Exception:
        return {
            "titolo": "",
            "ingredienti": [],
            "preparazione": "",
            "url": url,
        }

    if resp.status_code != 200:
        return {
            "titolo": "",
            "ingredienti": [],
            "preparazione": "",
            "url": url,
        }

    soup = BeautifulSoup(resp.text, "html.parser")

    titolo = ""
    ingredienti = []
    preparazione = ""

    for script_tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = (script_tag.string or script_tag.get_text() or "").strip()

        if not raw:
            continue

        try:
            data = json.loads(raw)
        except Exception:
            continue

        payloads = data if isinstance(data, list) else [data]

        for payload in payloads:
            if not isinstance(payload, dict):
                continue

            if payload.get("@type") != "Recipe":
                continue

            titolo = clean_text(payload.get("name", ""))

            raw_ingredients = payload.get("recipeIngredient", [])

            if isinstance(raw_ingredients, list):
                ingredienti = [clean_text(x) for x in raw_ingredients if str(x).strip()]

            raw_instructions = payload.get("recipeInstructions", [])
            steps = []

            if isinstance(raw_instructions, list):
                for step in raw_instructions:
                    if isinstance(step, dict):
                        txt = step.get("text", "")

                        if txt:
                            steps.append(clean_text(txt))

                    elif isinstance(step, str):
                        steps.append(clean_text(step))

            preparazione = " ".join([s for s in steps if s])

    if not titolo:
        h1 = soup.find("h1")
        titolo = clean_text(h1.get_text()) if h1 else ""

    if not ingredienti:
        ingredienti = [
            clean_text(li.get_text())
            for li in soup.select(".gz-ingredient, .gz-ingredient-item")
        ]

    if not preparazione:
        preparazione = " ".join(
            [
                clean_text(p.get_text())
                for p in soup.select(".gz-content-recipe-step p, .gz-content p")
            ]
        )

    return {
        "titolo": titolo,
        "ingredienti": ingredienti,
        "preparazione": preparazione,
        "url": url,
    }


def build_document(ricetta):
    return (
        f"TITOLO: {ricetta['titolo']}\n\n"
        f"INGREDIENTI:\n"
        + "\n".join(f"- {ing}" for ing in ricetta["ingredienti"])
        + "\n\n"
        f"PREPARAZIONE:\n{ricetta['preparazione']}"
    )


def main():
    print(f"[INFO] ChromaDB path: {CHROMA_DB_PATH}")
    print(f"[INFO] Collection target: {CHROMA_COLLECTION_NAME}")

    if CHROMA_COLLECTION_NAME != "ricette_giallozafferano":
        print(
            "[WARN] Attenzione: stai salvando in una collection diversa da "
            "'ricette_giallozafferano'."
        )

    print("[INFO] Raccolta link ricette...")
    ricette_links = get_all_ricette_links()

    print(f"[INFO] Trovati {len(ricette_links)} link di ricette.")

    existing_ids = get_existing_recipe_ids()
    existing_urls = get_existing_recipe_urls()

    print(f"[INFO] ID già presenti in collection: {len(existing_ids)}")
    print(f"[INFO] URL già presenti in collection: {len(existing_urls)}")

    saved = 0
    skipped = 0
    incomplete = 0
    errors = 0

    for idx, link in enumerate(ricette_links, start=1):
        try:
            ricetta_id = (
                f"giallozafferano_{hashlib.md5(link.encode('utf-8')).hexdigest()[:12]}"
            )

            if ricetta_id in existing_ids or link in existing_urls:
                print(f"[SKIP] Già presente nel DB: {link}")
                skipped += 1
                continue

            ricetta = estrai_ricetta(link)

            if ricetta["titolo"] and ricetta["preparazione"]:
                document = build_document(ricetta)

                chroma_collection.add(
                    documents=[document],
                    metadatas=[
                        {
                            "titolo": ricetta["titolo"],
                            "url": ricetta["url"],
                            "source": "giallozafferano",
                            "doc_type": "recipe",
                            "num_ingredienti": len(ricetta["ingredienti"]),
                        }
                    ],
                    ids=[ricetta_id],
                )

                existing_ids.add(ricetta_id)
                existing_urls.add(link)

                saved += 1

                print(
                    f"[OK] {idx}/{len(ricette_links)} Salvata: " f"{ricetta['titolo']}"
                )

            else:
                incomplete += 1
                print(f"[WARN] Ricetta incompleta: {link}")

        except Exception as e:
            errors += 1
            print(f"[ERR] Errore su {link}: {e}")

        time.sleep(1)

    print("\n========== RACCOLTA COMPLETATA ==========")
    print(f"Collection: {CHROMA_COLLECTION_NAME}")
    print(f"Ricette salvate: {saved}")
    print(f"Ricette già presenti/skippate: {skipped}")
    print(f"Ricette incomplete: {incomplete}")
    print(f"Errori: {errors}")


if __name__ == "__main__":
    main()
