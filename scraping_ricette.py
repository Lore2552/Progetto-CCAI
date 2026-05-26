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
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "archivio_ricette")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
chroma_collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

BASE_URL = "https://www.giallozafferano.it"

# Categorie principali da scandire per prendere quante piu ricette possibili.
CATEGORIES = [
    "Antipasti",
    "Primi",
    "Secondi-piatti",
    "Dolci-e-Desserts",
    "Lievitati",
    "Piatti-Unici",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}


def _extract_recipe_links_from_html(html):
    """Estrae link ricetta da una pagina categoria (JSON-LD + fallback HTML)."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    # Fonte principale: JSON-LD con ItemList, molto piu stabile dei selettori CSS.
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

    # Fallback: se manca JSON-LD, prova a prendere eventuali link diretti a ricette.
    if not links:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "ricette.giallozafferano.it" in href and href.endswith(".html"):
                links.add(href.replace("http://", "https://"))

    return links


def get_all_ricette_links():
    """Scorre tutte le categorie e tutte le pagine finche trova ricette."""
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

            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                print(
                    f"[INFO] Fine pagine per {category}: {url} (status {resp.status_code})"
                )
                break

            page_links = _extract_recipe_links_from_html(resp.text)

            # Difesa anti-loop: se una pagina replica la precedente, interrompe.
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
                    f"[INFO] {category} pagina {page}: trovati {len(page_links)} link ({added} nuovi)"
                )

            page += 1
            time.sleep(1)

    return sorted(all_links)


def get_existing_recipe_ids():
    """Legge tutti gli ID gia presenti in collection per fare skip dei duplicati."""
    try:
        return set(chroma_collection.get().get("ids", []))
    except Exception:
        return set()


def get_existing_recipe_urls():
    """Legge gli URL gia presenti nei metadati per deduplicare anche tra ID storici diversi."""
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
    return list(links)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def estrai_ricetta(url):
    """Estrae titolo, ingredienti e preparazione da una ricetta di GialloZafferano."""
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        return {"titolo": "", "ingredienti": [], "preparazione": "", "url": url}

    soup = BeautifulSoup(resp.text, "html.parser")

    titolo = ""
    ingredienti = []
    preparazione = ""

    # Strategia principale: JSON-LD Recipe.
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

    # Fallback CSS se JSON-LD non disponibile.
    if not titolo:
        titolo = soup.find("h1").get_text(strip=True) if soup.find("h1") else ""
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


def main():
    print(f"[INFO] ChromaDB path: {CHROMA_DB_PATH}")
    print(f"[INFO] Collection target: {CHROMA_COLLECTION_NAME}")
    print("[INFO] Raccolta link ricette...")
    ricette_links = get_all_ricette_links()
    print(f"[INFO] Trovati {len(ricette_links)} link di ricette.")

    existing_ids = get_existing_recipe_ids()
    existing_urls = get_existing_recipe_urls()
    print(f"[INFO] ID gia presenti in collection: {len(existing_ids)}")
    print(f"[INFO] URL gia presenti in collection: {len(existing_urls)}")

    ricette = []
    for idx, link in enumerate(ricette_links):
        try:
            ricetta_id = f"ricetta_{hashlib.md5(link.encode('utf-8')).hexdigest()[:12]}"

            if ricetta_id in existing_ids or link in existing_urls:
                print(f"[SKIP] Gia presente nel DB: {link}")
                continue

            ricetta = estrai_ricetta(link)
            if ricetta["titolo"] and ricetta["preparazione"]:
                ricette.append(ricetta)
                # Salva su ChromaDB
                chroma_collection.add(
                    documents=[
                        f"{ricetta['titolo']}\nIngredienti: {', '.join(ricetta['ingredienti'])}\nPreparazione: {ricetta['preparazione']}"
                    ],
                    metadatas=[{"titolo": ricetta["titolo"], "url": ricetta["url"]}],
                    ids=[ricetta_id],
                )
                existing_ids.add(ricetta_id)
                existing_urls.add(link)
                print(f"[OK] Salvata: {ricetta['titolo']}")
            else:
                print(f"[WARN] Ricetta incompleta: {link}")
        except Exception as e:
            print(f"[ERR] Errore su {link}: {e}")
        time.sleep(1)
    print(f"[INFO] Raccolta completata. Ricette salvate: {len(ricette)}")


if __name__ == "__main__":
    main()
