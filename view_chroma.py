import chromadb
import json
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Visualizza contenuto di una collection ChromaDB"
    )
    parser.add_argument(
        "--collection",
        default="archivio_ricette",
        help="Nome collection da aprire (default: auto-detect tra ricette_giallozafferano e archivio_ricette)",
    )
    args = parser.parse_args()

    db_path = "./chroma_db"

    print(f"Connessione a ChromaDB in: {db_path}...")

    try:
        # Inizializza il client verso la cartella locale
        client = chromadb.PersistentClient(path=db_path)

        collection_names = [c.name for c in client.list_collections()]

        if args.collection:
            collection_name = args.collection
        elif "ricette_giallozafferano" in collection_names:
            collection_name = "ricette_giallozafferano"
        else:
            collection_name = "archivio_ricette"

        print(f"Collection disponibili: {collection_names}")
        print(f"Collection selezionata: {collection_name}")

        # Recupera la collection che usi nel progetto
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(
            f"Errore nella connessione o nel recupero della collection '{collection_name}':"
        )
        print(e)
        return

    # Ottiene tutto il contenuto. .get() senza parametri recupera tutti i record
    results = collection.get()

    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    if not ids:
        print(f"La collection '{collection_name}' esiste ma è vuota.")
        return

    print(f"\n=========================================")
    print(f" TROVATI {len(ids)} RECORD IN '{collection_name}' ")
    print(f"=========================================\n")

    for i in range(len(ids)):
        print(f"--- RECORD {i+1} ---")
        print(f"ID      : {ids[i]}")

        meta = metadatas[i] if metadatas and metadatas[i] else {}
        print(f"Metadati: {json.dumps(meta, indent=2, ensure_ascii=False)}")

        doc = documents[i] if documents and documents[i] else "Nessun contenuto"
        # Mostro solo l'inizio del documento se è troppo lungo per non intasare il terminale
        preview_len = 6000
        doc_preview = (
            doc if len(doc) <= preview_len else doc[:preview_len] + " [...continua...]"
        )

        print(f"Testo   : {doc_preview}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
