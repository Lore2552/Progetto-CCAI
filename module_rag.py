import os
from collections import defaultdict
import chromadb
import cohere
from dotenv import load_dotenv

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_ricette = chroma_client.get_or_create_collection(
    name="ricette_giallozafferano"
)
collection_posts = chroma_client.get_or_create_collection(name="archivio_posts")

cohere_api_key = os.environ.get("COHERE_API_KEY")
co = cohere.Client(api_key=cohere_api_key) if cohere_api_key else None


def rrf_fusion(
    dense_results: list, keyword_results: list, k: int = 60, top_n: int = 10
) -> list:
    """Combina i risultati di Dense Retrieval e Keyword Search usando il Reciprocal Rank Fusion (RRF)."""
    rrf_scores = defaultdict(float)
    doc_map = {}

    for rank, doc in enumerate(dense_results):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] += 1.0 / (k + (rank + 1))

    for rank, doc in enumerate(keyword_results):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] += 1.0 / (k + (rank + 1))

    sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_docs[:top_n]]


def cohere_reranker(query: str, documents: list, top_n: int = 3) -> list:
    """Reranker reale che utilizza le API di Cohere (modello Multilingual v3)."""
    if not documents:
        return []

    if not co:
        print(
            "      [Reranker WARNING] Client Cohere non inizializzato (API Key mancante). Salto il reranking."
        )
        return documents[:top_n]

    print(f"      [Reranker] Esecuzione Reranking API su {len(documents)} documenti...")
    texts = [doc["text"] for doc in documents]

    try:
        response = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=texts,
            top_n=top_n,
        )

        final_documents = []
        for result in response.results:
            orig_idx = result.index
            matched_doc = documents[orig_idx]
            matched_doc["rerank_score"] = result.relevance_score
            final_documents.append(matched_doc)

        return final_documents

    except Exception as e:
        print(
            f"      [Reranker ERRORE] Chiamata fallita a Cohere: {e}. Fallback sui primi {top_n} risultati RRF."
        )
        return documents[:top_n]
