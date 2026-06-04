import os
import json
import chromadb
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_neo4j import Neo4jGraph
from langchain_core.tools import tool
import requests
import re

# from bs4 import BeautifulSoup
import trafilatura
import datetime
from collections import defaultdict
from rank_bm25 import BM25Okapi
import cohere

load_dotenv()

PLANNING_FILE = "planning_queue.json"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
# llm = ChatOllama(model="llama3.1:8b", temperature=0.5)

ddg_search = DuckDuckGoSearchAPIWrapper(max_results=5)

def load_planning_queue() -> List[Dict[str, str]]:
    """Carica la coda di pianificazione dal file JSON."""
    if os.path.exists(PLANNING_FILE):
        try:
            with open(PLANNING_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"   [Planner] Errore lettura file di piano: {e}. Ricreo coda.")
    return []

def save_planning_queue(queue: List[Dict[str, str]]):
    """Salva la coda di pianificazione su file JSON."""
    try:
        with open(PLANNING_FILE, "w", encoding="utf-8") as f:
            json.dump(queue, f, ensure_ascii=False, indent=4)
        print(f"   [Planner] Coda di pianificazione aggiornata e salvata su file.")
    except Exception as e:
        print(f"   [Planner] Errore nel salvataggio della coda: {e}")


graph_db = Neo4jGraph(
    url=os.environ.get("NEO4J_URI"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD"),
)


chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_ricette = chroma_client.get_or_create_collection(
    name="ricette_giallozafferano"
)
collection_posts = chroma_client.get_or_create_collection(name="archivio_posts")

KG_FILE = "knowledge_graph.graphml"

cohere_api_key = os.environ.get("COHERE_API_KEY")
co = cohere.Client(api_key=cohere_api_key) if cohere_api_key else None


# ==========================================
# 1. DEFINIZIONE DELLO STATO
# ==========================================
class AgentState(TypedDict):
    user_request: str
    kg_context: Any
    planned_topics: List[str]
    current_topic: str
    current_topic_type: str 
    editorial_justification: str
    raw_resources: List[Dict[str, str]]
    verified_resources: List[Dict[str, str]]
    draft: str
    human_feedback: str
    status: str
    revision_count: int
    rejected_topics: List[str]
    reasoning_trace: List[str]


# ==========================================
# 2. DEFINIZIONE DEI NODI (AGENTI)
# ==========================================


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

    # Controllo di sicurezza se l'API key non è configurata
    if not co:
        print(
            "      [Reranker WARNING] Client Cohere non inizializzato (API Key mancante). Salto il reranking."
        )
        return documents[:top_n]

    print(
        f"      [Reranker] Esecuzione Reranking API (Cohere v3) su {len(documents)} documenti..."
    )

    # Estraiamo solo i testi da inviare all'API
    texts = [doc["text"] for doc in documents]

    try:
        # Chiamata API a Cohere
        response = co.rerank(
            model="rerank-v3.5",  # Usiamo l'ultima versione stabile v3.5 (o 'rerank-multilingual-v3.0')
            query=query,
            documents=texts,
            top_n=top_n,
        )

        final_documents = []
        # Cohere restituisce gli indici ordinati per rilevanza descrescente
        for result in response.results:
            orig_idx = result.index
            matched_doc = documents[orig_idx]
            # Salviamo lo score di rilevanza restituito da Cohere per tracciabilità
            matched_doc["rerank_score"] = result.relevance_score
            final_documents.append(matched_doc)

        return final_documents

    except Exception as e:
        print(
            f"      [Reranker ERRORE] Chiamata fallita a Cohere: {e}. Fallback sui primi {top_n} risultati RRF."
        )
        # Fallback difensivo: se l'API va in crash (es. timeout o rate limit), non blocchiamo l'applicazione
        return documents[:top_n]


_LAST_KG_RAG_RESULT = {}


@tool
def kg_rag_tool(topic: str) -> str:
    """Usa questo tool per ottenere il contesto completo di una ricetta combinando dati strutturati (Knowledge Graph)
    e testi estesi (Vector Database) con tecniche avanzate di Hybrid Search (Dense + BM25), Fusione RRF e Cohere Reranking.
    """
    _LAST_KG_RAG_RESULT.clear()  # Svuoto il dizionario delle interazioni precedenti per evitare leak

    print(
        f"      [KG-RAG Tool] Avvio recupero ibrido avanzato per la ricetta: '{topic}'..."
    )

    # -------------------------------------------------------------------------
    # 1. QUERY RECIPE KG (Incluso URL)
    # -------------------------------------------------------------------------
    cypher_query = """
    MATCH (r:Recipe)
    WHERE $topic CONTAINS r.title OR r.title CONTAINS $topic
    OPTIONAL MATCH (r)-[:USES_INGREDIENT]->(i:Ingredient)
    OPTIONAL MATCH (r)-[:USES_TECHNIQUE]->(t:Technique)
    RETURN r.title AS recipe, 
           r.url AS url,
           collect(DISTINCT i.name) AS ingredients, 
           collect(DISTINCT t.name) AS techniques
    """
    try:
        risultato = graph_db.query(cypher_query, params={"topic": topic})
    except Exception as e:
        print(f"      [KG-RAG Tool] Errore KG: {e}")
        risultato = []

    ingredienti_str = ""
    tecniche_str = ""
    recipe_url = "Non disponibile"
    expanded_query = topic

    if risultato and risultato[0]["recipe"] is not None:
        dati = risultato[0]
        ingredienti_str = ", ".join(dati["ingredients"])
        tecniche_str = ", ".join(dati["techniques"])
        recipe_url = dati.get("url", "Non disponibile")

        # 2. QUERY EXPANSION
        expanded_query = f"{topic} {ingredienti_str} {tecniche_str}".strip()
        print(
            f"      [KG-RAG Tool] Trovate info nel KG. Query espansa: '{expanded_query}'"
        )
    else:
        print(
            "      [KG-RAG Tool] Nessuna informazione strutturata trovata nel KG. Uso la query base."
        )

    # -------------------------------------------------------------------------
    # STRATEGIA DI LOOKUP PER I METADATI
    # -------------------------------------------------------------------------
    id_to_metadata = {}  # Salverà la corrispondenza ID -> Dizionario Metadati

    # Recupero dei documenti del corpus per BM25
    try:
        tutti_i_doc = collection_ricette.get()
        all_documents = tutti_i_doc.get("documents", [])
        all_ids = tutti_i_doc.get("ids", [])
        all_metadatas = tutti_i_doc.get("metadatas", []) or []

        # Popoliamo il lookup con i metadati di tutto il corpus per il BM25
        for idx, doc_id in enumerate(all_ids):
            if idx < len(all_metadatas) and all_metadatas[idx]:
                id_to_metadata[doc_id] = all_metadatas[idx]

    except Exception as e:
        print(f"Errore recupero documenti complessivi per BM25: {e}")
        all_documents, all_ids = [], []

    testo_recuperato = ""
    documenti_recuperati = []

    if all_documents:
        # -------------------------------------------------------------------------
        # 3. DENSE RETRIEVAL SU CHROMA (Chiediamo i top 20)
        # -------------------------------------------------------------------------
        dense_docs = []
        try:
            risultati_chroma = collection_ricette.query(
                query_texts=[expanded_query], n_results=20
            )
            if (
                risultati_chroma
                and risultati_chroma.get("documents")
                and risultati_chroma["documents"][0]
            ):
                chroma_metas = risultati_chroma.get("metadatas", [[]])[0] or []

                for idx, doc_text in enumerate(risultati_chroma["documents"][0]):
                    doc_id = risultati_chroma["ids"][0][idx]

                    # Salva nel lookup se i metadati sono presenti nella query dense
                    if idx < len(chroma_metas) and chroma_metas[idx]:
                        id_to_metadata[doc_id] = chroma_metas[idx]

                    dense_docs.append(
                        {
                            "id": doc_id,
                            "text": doc_text[:2000],
                        }
                    )
        except Exception as e:
            print(f"Errore durante la ricerca Dense in ChromaDB: {e}")

        # -------------------------------------------------------------------------
        # 4. KEYWORD SEARCH (BM25 - Chiediamo i top 20)
        # -------------------------------------------------------------------------
        keyword_docs = []
        try:
            tokenized_corpus = [doc.lower().split(" ") for doc in all_documents]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = expanded_query.lower().split(" ")

            doc_scores = bm25.get_scores(tokenized_query)
            top_indices = sorted(
                range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
            )[:20]

            for idx in top_indices:
                if doc_scores[idx] > 0:
                    keyword_docs.append(
                        {"id": all_ids[idx], "text": all_documents[idx][:2000]}
                    )
        except Exception as e:
            print(f"Errore durante la ricerca Keyword (BM25): {e}")

        # -------------------------------------------------------------------------
        # 5. FUSIONE RISULTATI (RRF)
        # -------------------------------------------------------------------------
        fused_docs = rrf_fusion(
            dense_results=dense_docs, keyword_results=keyword_docs, k=60, top_n=20
        )

        # -------------------------------------------------------------------------
        # 6. COHERE RERANKING
        # -------------------------------------------------------------------------
        final_docs = cohere_reranker(
            query=expanded_query, documents=fused_docs, top_n=10
        )

        if final_docs:
            # --- MODIFICA CRITICA: Re-iniettiamo la chiave 'metadata' pescando dal lookup ---
            documenti_recuperati = [
                {
                    "id": doc.get("id"),
                    "testo": doc.get("text", ""),
                    "indice": idx + 1,
                    "metadata": id_to_metadata.get(
                        doc.get("id"), {}
                    ),  # <-- Recupero sicuro del metadato originale
                }
                for idx, doc in enumerate(final_docs)
            ]

            testo_recuperato = "\n\n--- FRAMMENTO VETTORIALE ---\n".join(
                [
                    f"[DOCUMENTO {doc['indice']} - ID: {doc['id']}]\n{doc['testo']}"
                    for doc in documenti_recuperati
                ]
            )
        else:
            testo_recuperato = "Nessun documento testuale di supporto trovato dopo il processo di filtraggio avanzato."
    else:
        testo_recuperato = "Database dei documenti vuoto."

    # -------------------------------------------------------------------------
    # 7. PROMPT RAG
    # -------------------------------------------------------------------------
    risposta_tool = f"RISULTATI DEL RECUPERO IBRIDO AVANZATO PER '{topic}':\n\n"
    risposta_tool += f"[1] DATI STRUTTURATI (REGOLE TASSATIVE DAL KNOWLEDGE GRAPH):\n"
    risposta_tool += f"- URL Ricetta: {recipe_url}\n"
    risposta_tool += f"- Ingredienti Obbligatori: {ingredienti_str if ingredienti_str else 'Nessuno specificato'}\n"
    risposta_tool += f"- Tecniche Richieste: {tecniche_str if tecniche_str else 'Nessuna specificata'}\n\n"
    risposta_tool += f"[2] TESTI DETTAGLIATI (COHERE RERANKED MULTILINGUAL):\n"
    risposta_tool += testo_recuperato

    _LAST_KG_RAG_RESULT["risultato"] = {
        "risposta_finale": risposta_tool,
        "testo_recuperato": testo_recuperato,
        "documenti_recuperati": documenti_recuperati,  # Ora contiene correttamente i metadati
        "metadata_kg": {
            "topic": topic,
            "url": recipe_url,
            "ingredienti": ingredienti_str,
            "tecniche": tecniche_str,
        },
    }

    return risposta_tool


@tool
def valuta_documento_locale(target_topic: str) -> str:
    """Valuta in modo critico se i testi delle ricette recuperate nel database locale sono PERTINENTI rispetto al target_topic richiesto."""

    print(
        f"      [Fact Checker Tool] Valutazione documenti rispetto al target: '{target_topic}'..."
    )

    # 1. INIZIALIZZAZIONE SICURA: Definiamo le variabili all'inizio dello scope.
    # In questo modo, qualsiasi cosa succeda nel ciclo, Python non darà MAI più un NameError.
    nuovo_testo = (
        "Nessun documento locale pertinente o parziale trovato dopo la validazione."
    )
    valutazioni = []
    documenti_validi = []

    risultato_kg_rag = _LAST_KG_RAG_RESULT.get("risultato", {})
    documenti_recuperati = risultato_kg_rag.get("documenti_recuperati", [])

    if not documenti_recuperati:
        return json.dumps(
            {
                "target_topic": target_topic,
                "valutazioni": [],
                "esito_generale": "NO",
                "messaggio": "Nessun documento recuperato da valutare.",
            }
        )

    # 2. CICLO DI VALUTAZIONE DEI DOCUMENTI
    for doc in documenti_recuperati:
        indice = doc.get("indice")
        doc_id = doc.get("id")
        document_text = doc.get("testo", "")

        # ESTRAZIONE PULITA DEL TITOLO: Peschiamo direttamente dal dizionario 'metadata'
        metadata = doc.get("metadata", {})
        titolo_ricetta_originale = metadata.get("titolo", "Titolo Sconosciuto")

        print(
            f"      [Fact Checker Tool] Valutazione documento {indice} ({titolo_ricetta_originale}) rispetto al target: '{target_topic}'..."
        )

        prompt = f"""Devi valutare se questo frammento (chunk) estratto dal database locale è utile per scrivere un post su: '{target_topic}'.
        CONTESTO DEL DOCUMENTO:
        - Titolo della ricetta originale di appartenenza: "{titolo_ricetta_originale}"
        - Testo del frammento:
        {document_text}

        REGOLE DI VALUTAZIONE TASSATIVE:
        1. CONTROLLO TITOLO (BLOCCANTE): Guarda PRIMA DI TUTTO il "Titolo della ricetta originale". Se descrive un piatto strutturalmente diverso dal target richiesto (es. Target: "Cannoli" -> Titolo Originale: "Treccine"), rispondi "NO". Non importa se il testo contiene ingredienti in comune.
        2. Rispondi "SI" SE: Il "Titolo originale" corrisponde a '{target_topic}' (comprese le varianti specifiche richieste nel target) E il frammento contiene passaggi o ingredienti di quella ricetta. Trattandosi di un chunk, è normale che contenga solo una parte delle istruzioni; valutalo come "SI" perché appartiene alla ricetta giusta.
        3. Rispondi "PARZIALE" SE: Il "Titolo originale" è la ricetta BASE del target, ma manca palesemente la variante specifica richiesta (es. Target: "Cannoli ricotta e pistacchio" -> Titolo Originale: "Cannoli alla ricotta"). Il chunk è utile come base, ma incompleto.
        4. Rispondi "NO" SE: Il frammento è completamente irrilevante o decontestualizzato.

        Rispondi in modo secco iniziando la riga SOLO con la parola "SI", "PARZIALE" o "NO", seguita da un trattino e una brevissima motivazione.
        """

        try:
            valutazione = llm.invoke([HumanMessage(content=prompt)]).content.strip()
        except Exception as e:
            valutazione = f"NO - (Errore valutazione: {e})"

        val_upper = valutazione.upper()
        is_valid = val_upper.startswith("SI") or val_upper.startswith("PARZIALE")

        dati_valutazione = {
            "indice": indice,
            "id": doc_id,
            "titolo_originale": titolo_ricetta_originale,
            "valutazione": valutazione,
            "status": "ACCETTATO" if is_valid else "SCARTATO",
        }

        if is_valid:
            documenti_validi.append(doc)
            valutazioni.append(dati_valutazione)

    # 3. ELABORAZIONE FINALE ED AGGIORNAMENTO STATO (FUORI DAL CICLO FOR)
    _LAST_KG_RAG_RESULT["risultato"]["documenti_recuperati"] = documenti_validi

    if documenti_validi:
        nuovo_testo = "\n\n--- FRAMMENTO VETTORIALE ---\n".join(
            [
                f"[DOCUMENTO {doc['indice']} - ID: {doc['id']} - TITOLO ORIGINALE: {doc.get('metadata', {}).get('titolo', 'Sconosciuto')}]\n{doc['testo']}"
                for doc in documenti_validi
            ]
        )
    else:
        nuovo_testo = (
            "Nessun documento locale pertinente o parziale trovato dopo la validazione."
        )

    _LAST_KG_RAG_RESULT["risultato"]["testo_recuperato"] = nuovo_testo

    if "metadata_kg" in _LAST_KG_RAG_RESULT.get("risultato", {}):
        meta = _LAST_KG_RAG_RESULT["risultato"]["metadata_kg"]
        risposta_filtrata = f"RISULTATI DEL RECUPERO IBRIDO AVANZATO PER '{meta.get('topic', target_topic)}':\n\n"
        risposta_filtrata += (
            "[1] DATI STRUTTURATI (REGOLE TASSATIVE DAL KNOWLEDGE GRAPH):\n"
        )
        risposta_filtrata += f"- URL Ricetta: {meta.get('url', 'N/A')}\n"
        risposta_filtrata += f"- Ingredienti Obbligatori: {meta.get('ingredienti', 'Nessuno specificato')}\n"
        risposta_filtrata += (
            f"- Tecniche Richieste: {meta.get('tecniche', 'Nessuna specificata')}\n\n"
        )
        risposta_filtrata += (
            f"[2] TESTI DETTAGLIATI (FILTRATI SOLO PER PERTINENZA):\n{nuovo_testo}"
        )

        _LAST_KG_RAG_RESULT["risultato"]["risposta_finale"] = risposta_filtrata

    return json.dumps(
        {
            "target_topic": target_topic,
            "valutazioni": valutazioni,
        }
    )


@tool
def cerca_e_leggi_sul_web(query: str) -> str:
    """Cerca sul web, estrae il main content e restituisce il testo di PIÙ pagine trovate per permettere il confronto."""
    print(f"      [Web Tool] Eseguo ricerca web e Scraping DIRETTO per: '{query}'...")

    banned_domains = [
        "wikipedia.org",
        "youtube.com",
        "facebook.com",
        "instagram.com",
        "twitter.com",
        "tiktok.com",
        "pinterest.com",
        "amazon.",
        "ebay.",
    ]

    for attempt in range(3):
        current_query = (
            query
            if attempt == 0
            else f"{query} ricetta procedimenti dosi -video -wikipedia (attempt {attempt+1})"
        )

        try:
            results = ddg_search.results(current_query, max_results=5)
            opzioni_estratte = []

            for r in results:
                url = r["link"]

                if any(domain in url.lower() for domain in banned_domains):
                    continue

                try:
                    # Trafilatura scarica ed estrae solo il contenuto principale (articolo/ricetta)
                    # ignorando menu, footer, commenti e sidebar in automatico.
                    downloaded = trafilatura.fetch_url(url)
                    if downloaded:
                        testo_pulito = trafilatura.extract(
                            downloaded,
                            include_comments=False,
                            include_tables=True,
                            no_fallback=False,
                        )

                        if testo_pulito:
                            testo_lower = testo_pulito.lower()
                            ha_keyword_ricetta = any(
                                k in testo_lower
                                for k in [
                                    "ingredient",
                                    "preparazion",
                                    "procediment",
                                    "dosi",
                                ]
                            )

                            if len(testo_pulito) > 500 and ha_keyword_ricetta:
                                # Salviamo chiaramente l'URL nel testo passato al LLM
                                testo_formattato = f"\n--- OPZIONE WEB {len(opzioni_estratte) + 1} ---\nURL_FONTE: {url}\nCONTENUTO:\n{testo_pulito[:4000]}\n"
                                opzioni_estratte.append(testo_formattato)

                    if len(opzioni_estratte) >= 3:
                        break

                except Exception:
                    pass

            if opzioni_estratte:
                return "".join(opzioni_estratte)

        except Exception as e:
            print(f"      [Web Tool] Errore nel tentativo {attempt+1}: {e}")

    return "Non è stato possibile estrarre i contenuti delle pagine pertinenti. Riprova con una query diversa."


def topic_planner(state: AgentState) -> dict:
    print("-> Esecuzione Topic Planner con Coda Persistente (Primo, Secondo, Dolce)...")

    # 1. Recupero degli argomenti passati dal Knowledge Graph per evitarli
    cypher_query = "MATCH (t:Topic) RETURN t.name AS name"
    risultati = graph_db.query(cypher_query)
    past_topics = [res["name"] for res in risultati if res.get("name")]
    rejected_topics = state.get("rejected_topics", [])
    all_avoids = past_topics + rejected_topics

    print(f"   [KG] Argomenti passati da evitare: {past_topics}")
    if rejected_topics:
        print(f"   [Feedback] Argomenti scartati in questa sessione: {rejected_topics}")

    # 2. Caricamento della coda persistente da file JSON
    queue = load_planning_queue()
    print(f"   [Planner] Coda attuale letta da file: {queue}")

    current_topic = ""
    current_topic_type = ""

    last_status = state.get("status", "")

    if queue and last_status != "rejected_topic":
        next_item = queue.pop(0)
        current_topic = next_item["ricetta"]
        current_topic_type = next_item["tipo"]
        print(f"   [Planner] Consumata ricetta dalla coda: '{current_topic}' ({current_topic_type})")
    elif last_status == "rejected_topic" and rejected_topics:
        print(f"   [Planner] Rilevato topic scartato. Rigenerazione della categoria specifica per mantenere l'equilibrio...")

    if current_topic:
        all_avoids.append(current_topic)
        
    # 3. Identificazione delle categorie mancanti (Primo, Secondo, Dolce)
    tipi_presenti = [item["tipo"] for item in queue]
    tipi_mancanti = []
    for tipo in ["Primo", "Secondo", "Dolce"]:
        if tipo not in tipi_presenti:
            tipi_mancanti.append(tipo)

    # Evitiamo che l'LLM generi ricette uguali a quelle già ferme in coda
    for item in queue:
        all_avoids.append(item["ricetta"])

    justification = state.get("editorial_justification", "Nessuna giustificazione precedente.")

    # 4. Chiediamo all'LLM di riempire i gap strutturali della coda
    if tipi_mancanti:
        print(f"   [Planner] Tipi di piatto da pianificare: {tipi_mancanti}")
    
        prompt = (
            f"La richiesta dell'utente è: '{state['user_request']}'.\n"
            f"KNOWLEDGE GRAPH (RICETTE DA EVITARE ASSOLUTAMENTE): {all_avoids}.\n"
            f"Devi generare esattamente una nuova, famosa e autentica ricetta della cucina tradizionale italiana per ciascuna di queste categorie mancanti: {tipi_mancanti}.\n"
            f"Punta a un livello di specificità medio (es. 'Risotto alla Milanese', 'Saltimbocca alla Romana', 'Tiramisù').\n\n"
            f"REGOLA DI FORMATTAZIONE TASSATIVA:\n"
            f"Rispondi SOLO ed esclusivamente con un oggetto JSON valido contenente due chiavi:\n"
            f"1. 'giustificazione_editoriale': Spiega esplicitamente in una frase perché la scelta di queste specifiche ricette garantisce diversità, copertura del dominio ed una strategia editoriale coerente rispetto al passato del blog.\n"
            f"2. 'sequenza_piano': Un array di oggetti, dove ciascun oggetto contiene le chiavi 'tipo' e 'ricetta'.\n\n"
            f"Esempio di formato richiesto:\n"
            f"{{\n"
            f'  "giustificazione_editoriale": "Pianifico un Primo romano per coprire il centro Italia e un Dolce piemontese per bilanciare il menu strutturale.",\n'
            f'  "sequenza_piano": [\n'
            f'    {{"tipo": "Primo", "ricetta": "Spaghetti alla Carbonara"}},\n'
            f'    {{"tipo": "Dolce", "ricetta": "Panna Cotta"}}\n'
            f'  ]\n'
            f"}}"
        )

        try:
            # CORREZIONE: Invocazione corretta del modello LLM
            response = llm.invoke([HumanMessage(content=prompt)]).content.strip()

            # Rimozione di eventuali blocchi di codice markdown (```json ... ```)
            if response.startswith("```"):
                response = re.sub(r"^```[a-zA-Z]*\n|```$", "", response, flags=re.M).strip()

            # Estrattore difensivo Regex: isola solo l'array JSON valido
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            json_clean = json_match.group(0) if json_match else response

            data_pianificata = json.loads(json_clean)
            justification = data_pianificata.get("giustificazione_editoriale", "Pianificazione strategica per la diversità del menu.")
            nuovi_piatti = data_pianificata.get("sequenza_piano", [])
            
            if isinstance(nuovi_piatti, list):
                # Se siamo al primo avvio (coda vuota), estraiamo subito il primo elemento generato come topic corrente
                if not current_topic and nuovi_piatti:
                    primo_estratto = nuovi_piatti.pop(0)
                    current_topic = primo_estratto["ricetta"]
                    print(f"   [Planner] Primo avvio: '{current_topic}' impostato come topic corrente.")

                for piatto in nuovi_piatti:
                    ricetta_nome = piatto.get("ricetta", "").strip()
                    tipo_nome = piatto.get("tipo", "").strip().capitalize()
                    
                    if ricetta_nome and tipo_nome:
                        ricetta_lower = ricetta_nome.lower()
                        is_duplicate = False
                        for avoid_item in all_avoids:
                            avoid_lower = avoid_item.lower()
                            if ricetta_lower in avoid_lower or avoid_lower in ricetta_lower:
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            queue.append({"tipo": tipo_nome, "ricetta": ricetta_nome})
                            all_avoids.append(ricetta_nome)
                            if tipo_nome in tipi_mancanti:
                                tipi_mancanti.remove(tipo_nome)
                        else:
                            print(f"   [Planner BUG-SHIELD] Scartato duplicato semantico: '{ricetta_nome}'")


        except Exception as e:
            print(f"   [Planner Errore LLM/Parsing JSON] {e}. Imposto fallback d'emergenza.")
            # Fallback d'emergenza se l'LLM sbaglia il JSON o va in timeout
            if not current_topic:
                current_topic = "Spaghetti alla Carbonara"
            if not queue:
                queue = [
                    {"tipo": "Secondo", "ricetta": "Cotoletta alla Milanese"},
                    {"tipo": "Dolce", "ricetta": "Panna Cotta"},
                ]
            justification = "Fallback applicato per mantenere la continuità editoriale."
    

    # Salva lo stato aggiornato della coda
    save_planning_queue(queue)

    # Estraiamo una lista piatta di stringhe per lo stato di LangGraph
    planned_list = [item["ricetta"] for item in queue]

    print(f"   [Planner] Topic Corrente in lavorazione: '{current_topic}'")
    print(f"   [Planner] Prossimi in coda per le prossime volte: {planned_list}")

    return {
        "planned_topics": planned_list,
        "current_topic": current_topic,
        "current_topic_type": current_topic_type,
        "editorial_justification": justification,
        "status": "planning_done",
    }



import re


def resource_researcher(state: AgentState) -> dict:
    topic = state["current_topic"]
    trace = state.get("reasoning_trace", [])

    print(f"-> Esecuzione Resource Researcher (con paradigma ReAct) per: {topic}")

    tools = [kg_rag_tool, valuta_documento_locale, cerca_e_leggi_sul_web]
    react_agent = create_agent(llm, tools)

    prompt = f"""Devi raccogliere dati completi su come si prepara questa ricetta: '{topic}'.
    Hai a disposizione 3 tool per la ricerca:
    1. 'kg_rag_tool': Cerca la ricetta nel database locale ibrido. Restituisce le regole fisse del Knowledge Graph (ingredienti e tecniche obbligatorie) e il frammento di testo completo della ricetta.
    2. 'valuta_documento_locale': Usa questo tool (fornendo in input SOLO il parametro 'target_topic') per far valutare oggettivamente i documenti appena recuperati dal DB locale ed escludere automaticamente quelli irrilevanti.
    3. 'cerca_e_leggi_sul_web': Cerca ed estrae integralmente i passaggi da pagine internet. Utilizzalo se la ricetta locale è assente, parziale o per colmare i dettagli mancanti.

    Flusso Operativo Richiesto:
    1. REASONING: Inizia ogni azione con 'Thought: ...'
    2. Usa SEMPRE come prima azione il tool 'kg_rag_tool'.
    3. Usa SUBITO il tool 'valuta_documento_locale' (chiamalo SOLO con il parametro target_topic, il testo è già memorizzato dal tool precedente). Se la risposta dice che copre interamente la variante, FERMATI ed esponi tu una sintesi che unisce ingredienti e passaggi di preparazione del piatto in un testo descrittivo.
    4. Se l'esito della validazione del database locale è 'PARZIALE' perché manca un ingrediente o un dettaglio specifico rispetto al topic, la ricerca sul web deve essere MIRATA a prendere le informazioni mancanti o la ricetta COMPLETA per integrarle.
    5. Se il tool locale non trova nulla o sei insoddisfatto, usa 'cerca_e_leggi_sul_web' per ottenere l'intera ricetta. Questo tool andrà a restituire diverse fonti web, devi prendere la più inerente al nostro topic (anche basandoti sul nome della ricetta e gli ingredienti). 
    6. FALLBACK: Se anche dopo le stampe e le ricerche sul server web non si trova nulla di perfettamente completo, procedi prendendo assieme le informazioni e frammenti 'parziali' trovati fino ad ora, assemblando il draft con ciò che hai.
    7. Alla fine di tutti i controlli, unisci ciò che hai ottenuto di buono sia in locale che sul web. Come TUA risposta FINALE restituisci un unico riassunto descrittivo.

    REGOLA FONDAMENTALE PER LA RISPOSTA FINALE:
    Se durante l'esecuzione hai usato il tool 'cerca_e_leggi_sul_web', inserisci alla fine del tuo riassunto una riga esatta con l'URL scelto:
    URL_SELEZIONATO: <url_della_fonte_web>
    Se NON hai usato il tool web perché il database locale era già sufficiente, scrivi semplicemente:
    URL_SELEZIONATO: DB_Locale
    """

    print("   [ReAct] Avvio l'agente. Attendi l'elaborazione...")
    try:
        response = react_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        print("   [ReAct] Elaborazione conclusa!")

        final_answer = ""
        for msg in response.get("messages", []):
            if getattr(msg, "type", "") == "ai" and msg.content:
                final_answer = msg.content
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    thought_str = f"Thought: {msg.content.strip()}"
                    trace.append(thought_str)
                    print(f"   [ReAct] {thought_str}")
                    for tool_call in msg.tool_calls:
                        action_str = f"Action: Chiamo il tool '{tool_call['name']}' con query {tool_call['args']}"
                        trace.append(action_str)
                        print(f"   [ReAct] {action_str}")
            elif getattr(msg, "type", "") == "tool":
                obs_str = f"Observation: Risultato del tool {msg.tool_call_id}. (Letti {len(msg.content)} caratteri)"
                trace.append(obs_str)
                print(f"   [ReAct] {obs_str}\n")

        ha_usato_il_web = False
        url_da_tool = []

        for msg in response.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == "cerca_e_leggi_sul_web":
                        ha_usato_il_web = True

            if getattr(msg, "type", "") == "tool":
                for line in str(msg.content).split("\n"):
                    if "URL_FONTE:" in line:
                        url_da_tool.append(line.split("URL_FONTE:")[1].strip())

        url_scelto = "DB_Locale"
        if ha_usato_il_web:
            url_scelto = "Nessun URL trovato"

            for line in final_answer.split("\n"):
                clean_line = line.replace("**", "").replace("*", "")
                if "URL_SELEZIONATO:" in clean_line.upper():
                    parti = re.split(r"(?i)URL_SELEZIONATO:", clean_line)
                    if len(parti) > 1:
                        url_scelto = parti[1].strip()
                        break

            if url_scelto == "Nessun URL trovato":
                links_nel_testo = re.findall(r"(https?://[^\s]+)", final_answer)
                if links_nel_testo:
                    url_scelto = links_nel_testo[-1]

            if url_scelto == "Nessun URL trovato" and url_da_tool:
                url_scelto = url_da_tool[0]

        raw = [{"url": url_scelto, "content": final_answer, "title": topic}]

        contesto_rag_effettivo = _LAST_KG_RAG_RESULT.get("risultato", {}).get(
            "risposta_finale",
            "Nessun contesto DB locale o non attivato in modo sufficiente.",
        )

    except Exception as e:
        print(f"   [Errore ReAct Agent] {e}")
        raw = []
        contesto_rag_effettivo = None

    return {
        "raw_resources": raw,
        "kg_context": contesto_rag_effettivo,
        "reasoning_trace": trace,
        "status": "research_done",
    }


def quality_fact_checker(state: AgentState) -> dict:
    print("-> Esecuzione Quality & Fact Checker (Verifica Contenuti)...")
    raw = state.get("raw_resources", [])
    topic = state["current_topic"]
    valid_links = []

    risultati = graph_db.query("MATCH (t:Topic) RETURN t.name AS name")
    past_topics = [res["name"] for res in risultati if res.get("name")]

    for res in raw:
        content = res.get("content", "")[:3500]
        prompt = (
            f"Sei un Fact Checker. Ricetta sintetica presentata: '{topic}'.\n\n"
            f"KNOWLEDGE GRAPH (RICETTE GIA' SCRITTE): {past_topics}\n"
            f"Testo della ricetta da validare:\n'{content}'.\n\n"
            f"La ricetta è già inclusa nel knowledge graph? Se si, rispondi 'NO'.\n"
            f"Il procedimento espone la preparazione di questo piatto ed è valido? Rispondi esattamente 'SI' o 'NO'."
        )
        try:
            eval_res = llm.invoke([HumanMessage(content=prompt)]).content.upper()
            if "SI" in eval_res or "SÌ" in eval_res:
                valid_links.append(res)
                print("   [LLM Evaluator] Output di ricerca approvato.")
            else:
                print(f"   [LLM Evaluator] Scartato o non valido. Resp: {eval_res}")
        except Exception:
            valid_links.append(res)

    return {"verified_resources": valid_links, "status": "fact_checking_done"}


def drafter(state: AgentState) -> dict:
    print("-> Esecuzione Drafter (Generazione testo in corso)...")

    claims_query = """
    MATCH (s:Entity)-[r]->(o:Entity)
    RETURN s.name + ' ' + type(r) + ' ' + o.name AS claim 
    LIMIT 10
    """
    try:
        risultati_grafo = graph_db.query(claims_query)
        past_claims = [res["claim"] for res in risultati_grafo if res.get("claim")]
    except Exception as e:
        past_claims = []
        print(f"   [Avviso] Impossibile recuperare triple dal Grafo: {e}")

    feedback = state.get("human_feedback", "")
    previous_draft = state.get("draft", "")
    topic = state["current_topic"]
    topic_type = state.get("current_topic_type", "Piatto della tradizione")
    justification_context = state.get("editorial_justification", "")
    contesto_rag = state.get("kg_context", "Nessun contesto RAG locale trovato.")

    # Fallback sicuro: controlla sia verified_resources che raw_resources
    sources = state.get("verified_resources") or state.get("raw_resources") or []

    # Estraiamo l'URL pulito ricavato dal Resource Researcher
    url_fonte = sources[0].get("url", "DB_Locale") if sources else "DB_Locale"
    sources_text = "\n".join([s.get("content", "")[:1000] for s in sources])

    prompt = (
        f"Scrivi un coinvolgente articolo di blog su come preparare la ricetta: '{topic}'.\n\n"
        f"Questo piatto è classificato nella categoria: {topic_type}.\n"
        f"Strategia editoriale associata: {justification_context}\n\n"
        f"=== CONTESTO RAG UFFICIALE (Knowledge Graph + Vector DB locale) ===\n"
        f"{contesto_rag}\n"
        f"===================================================================\n\n"
        f"Usa questi frammenti di ricerca web come integrazione secondaria:\n{sources_text}\n\n"
        f"KNOWLEDGE GRAPH (AFFERMAZIONI PASSATE DEL BLOG): {past_claims}\n\n"
        f"REGOLE FONDAMENTALI (PENA IL RIFIUTO DELLO SCRITTO):\n"
        f"1. DEVI rispettare tassativamente gli ingredienti obbligatori e le tecniche indicati nel 'CONTESTO RAG UFFICIALE'.\n"
        f"2. Inizia la tua risposta ESATTAMENTE con 'TITOLO: <il tuo titolo accattivante>' sulla primissima riga.\n"
        f"3. L'articolo deve essere in italiano.\n"
        f"4. Assicurati che l'articolo sia COERENTE con le affermazioni passate del blog. Non contraddirle.\n"
        f"5. Se pertinente, CONNETTI il nuovo post a una di queste vecchie informazioni passate.\n"
        f"6. Scrivi in modo diretto come se fossi uno chef (VIETATO usare scuse, premesse o frasi introduttive come 'ecco l'articolo').\n"
        f"7. L'articolo deve contenere necessariamente la lista degli ingredienti della ricetta trattata.\n"
        f"8. L'articolo deve contenere necessariamente la preparazione dettagliata della ricetta trattata.\n"
        f"9. CITAZIONI IN LINEA OBBLIGATORIE (REQUISITO CRITICO): Il tuo testo finale deve dimostrare chiaramente l'uso combinato delle fonti. Applica queste citazioni nel testo:\n"
        f"   - Quando inserisci un ingrediente obbligatorio o una tecnica del database locale, usa: [Fonte: Knowledge Graph].\n"
        f"   - Quando descrivi passaggi pratici, tempi o dettagli presi dal web, usa tassativamente l'URL: [Fonte: {url_fonte}].\n"
        f"   Esempio: 'Aggiungete il guanciale [Fonte: Knowledge Graph] e fatelo rosolare a fuoco lento per circa 10 minuti [Fonte: {url_fonte}].'\n"
        f"10. DICITURA FINALE OBBLIGATORIA: Alla fine dell'articolo, lascia una riga vuota e inserisci la fonte globale. "
    )

    # Configurazione dinamica della stringa finale in base alla provenienza dei dati
    if url_fonte == "DB_Locale" or url_fonte == "Nessun URL trovato":
        prompt += "Scrivi ESATTAMENTE: 'Fonte: Ricetta del database locale'."
    else:
        prompt += f"Scrivi ESATTAMENTE: 'Fonte web: {url_fonte}'."

    # Gestione dell'eventuale ciclo di feedback umano
    if feedback and previous_draft:
        prompt += f"\n\nEcco la tua BOZZA PRECEDENTE:\n---\n{previous_draft}\n---\n\nCRITICO: L'utente ha rifiutato la bozza precedente con questo feedback: '{feedback}'. DEVI riscrivere pesantemente la bozza. Ricorda SEMPRE di focalizzarti su UNA SOLA RICETTA."

    # 4. Invocazione del Modello
    try:
        response_text = llm.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        print(
            f"\n[ERRORE FATALE] Impossibile generare la bozza a causa di un errore API: {e}"
        )
        raise SystemExit(1)

    # 5. Parsing del Titolo e pulizia del testo finale
    new_title = topic
    draft = response_text

    lines = response_text.split("\n")
    for i, line in enumerate(lines):
        clean_line = line.strip().upper()
        if clean_line.startswith("TITOLO:") or clean_line.startswith("**TITOLO:**"):
            # Estraiamo il titolo pulito eliminando markdown e virgolette
            new_title = (
                line.split(":", 1)[1]
                .strip()
                .replace("**", "")
                .replace('"', "")
                .replace("'", "")
            )
            # Ricostruiamo il draft escludendo la riga del titolo
            draft = "\n".join(lines[:i] + lines[i + 1 :]).strip()
            break

    return {
        "draft": draft,
        "current_topic": new_title,
        "status": "draft_ready",
        "verified_resources": sources,  # Manteniamo popolato lo stato per i nodi successivi
        "revision_count": state.get("revision_count", 0) + 1,
    }


def human_review(state: AgentState) -> dict:
    import tkinter as tk
    from tkinter import messagebox
    from tkinter.scrolledtext import ScrolledText

    print("-> Esecuzione Human Review (Apertura finestra grafica in corso)...")

    draft_content = state.get("draft", "")
    topic = state.get("current_topic", "Senza Titolo")

    result = {}

    def on_approve():
        result["status"] = "approved"
        result["human_feedback"] = ""
        root.destroy()

    def on_reject():
        rejected = state.get("rejected_topics", []) + [state["current_topic"]]
        result["status"] = "rejected_topic"
        result["human_feedback"] = "Notizia scartata dall'utente."
        result["rejected_topics"] = rejected
        root.destroy()

    def on_rewrite():
        feedback = feedback_entry.get().strip()
        if not feedback:
            messagebox.showwarning(
                "Attenzione",
                "Devi inserire un feedback (es. 'Scrivilo più corto' o 'Usa un tono più formale') per poter riscrivere l'articolo.",
            )
            return
        result["status"] = "rewrite"
        result["human_feedback"] = feedback
        root.destroy()

    root = tk.Tk()
    root.title(f"Revisione Bozza: {topic}")
    root.geometry("850x650")

    justification_text = state.get("editorial_justification", "Nessuna giustificazione fornita.")
    topic_type = state.get("current_topic_type", "N/A")

    lbl = tk.Label(
        root,
        text=f"Argomento attuale: {topic} ({topic_type})\nStrategia: {justification_text}",
        font=("Helvetica", 11, "italic"),
        fg="blue",
        justify=tk.LEFT,
        wraplength=800
    )
    lbl.pack(pady=10)

    txt_area = ScrolledText(
        root, wrap=tk.WORD, width=90, height=22, font=("Helvetica", 11)
    )
    txt_area.insert(tk.INSERT, draft_content)
    txt_area.config(state=tk.DISABLED)
    txt_area.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

    frame_feedback = tk.Frame(root)
    frame_feedback.pack(fill=tk.X, padx=20, pady=5)
    tk.Label(
        frame_feedback,
        text="Opzionale - Spiega cosa modificare:",
        font=("Helvetica", 10),
    ).pack(side=tk.LEFT)
    feedback_entry = tk.Entry(frame_feedback, font=("Helvetica", 11))
    feedback_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

    frame_btn = tk.Frame(root)
    frame_btn.pack(pady=15)

    tk.Button(
        frame_btn,
        text="✅ Approva e Pubblica",
        bg="lightgreen",
        font=("Helvetica", 11, "bold"),
        padx=10,
        pady=5,
        command=on_approve,
    ).pack(side=tk.LEFT, padx=15)
    tk.Button(
        frame_btn,
        text="🔁 Riscrivi (Invia Feedback)",
        bg="gold",
        font=("Helvetica", 11, "bold"),
        padx=10,
        pady=5,
        command=on_rewrite,
    ).pack(side=tk.LEFT, padx=15)
    tk.Button(
        frame_btn,
        text="❌ Scarta Notizia",
        bg="lightcoral",
        font=("Helvetica", 11, "bold"),
        padx=10,
        pady=5,
        command=on_reject,
    ).pack(side=tk.LEFT, padx=15)

    def on_closing():
        if messagebox.askokcancel(
            "Arresta",
            "La chiusura della finestra comporterà lo scarto della notizia. Procedere?",
        ):
            on_reject()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.lift()
    root.attributes("-topmost", True)
    root.after_idle(root.attributes, "-topmost", False)

    root.mainloop()

    if not result:
        rejected = state.get("rejected_topics", []) + [state["current_topic"]]
        result = {
            "status": "rejected_topic",
            "human_feedback": "Notizia scartata (finestra chiusa).",
            "rejected_topics": rejected,
        }

    return result


@tool
def aggiorna_knowledge_graph_db(topic: str, draft: str, source_urls: List[str]) -> str:
    """Aggiorna il K-RAG: Inserisce i vettori in ChromaDB e le triple nel Knowledge Graph Neo4j."""

    count_res = graph_db.query("MATCH (p:Post) RETURN count(p) AS totale")
    post_count = count_res[0]["totale"] if count_res else 0
    post_id = f"Post_{post_count + 1}"

    collection_posts.add(documents=[draft], metadatas=[{"topic": topic}], ids=[post_id])
    print(f"   [ChromaDB] Vettori indicizzati per: {topic}")

    prompt_triple = (
        f"Sei un esperto estrattore di dati per un Knowledge Graph culinario.\n"
        f"Analizza questo testo sulla ricetta '{topic}':\n'{draft}'.\n\n"
        f"Estrai TUTTE le relazioni fondamentali (ingredienti principali e tecniche) in formato tripla.\n"
        f"Usa ESATTAMENTE il formato: Soggetto | RELAZIONE | Oggetto\n\n"
        f"REGOLE TASSATIVE:\n"
        f"1. Il Soggetto deve essere SEMPRE '{topic}'.\n"
        f"2. Per la RELAZIONE, DEVI usare SOLO uno di questi termini pre-approvati (Vietato inventarne altri):\n"
        f"   - USA_INGREDIENTE (Es. Ragù | USA_INGREDIENTE | Carne macinata)\n"
        f"   - USA_TECNICA (Es. Risotto | USA_TECNICA | Mantecatura)\n"
        f"   - TIPO_DI_PIATTO (Es. Tiramisù | TIPO_DI_PIATTO | Dolce)\n"
        f"3. Estrai tutte le triple rilevanti che trovi nel testo (minimo 3, ma estraine quante ne servono per descrivere bene il piatto).\n"
        f"4. Rispondi SOLO con le triple, una per riga. Nessun testo introduttivo, nessun commento, nessun backtick o markdown."
    )
    try:
        claims_response = llm.invoke([HumanMessage(content=prompt_triple)]).content
        triple = [c.strip() for c in claims_response.split("\n") if "|" in c]
    except:
        triple = []

    count_res = graph_db.query("MATCH (p:Post) RETURN count(p) AS totale")
    post_count = count_res[0]["totale"] if count_res else 0
    post_id = f"Post_{post_count + 1}"

    cypher_post = """
    MERGE (t:Topic {name: $topic})
    CREATE (p:Post {id: $post_id})
    MERGE (p)-[:COVERS_TOPIC]->(t)
    """

    graph_db.query(cypher_post, params={"topic": topic, "post_id": post_id})

    for tripla in triple:
        try:
            sog, rel, obj = [item.strip() for item in tripla.split("|")]
            rel_clean = rel.replace(" ", "_").upper()

            cypher_triple = f"""
            MATCH (p:Post {{id: $post_id}})
            MERGE (s:Entity {{name: $sog}})
            MERGE (o:Entity {{name: $obj}})
            MERGE (s)-[:{rel_clean}]->(o)
            MERGE (p)-[:MENTIONS]->(s)
            """
            graph_db.query(
                cypher_triple, params={"post_id": post_id, "sog": sog, "obj": obj}
            )

            print(f"   [Neo4j] Tripla inserita: ({sog}) -[{rel_clean}]-> ({obj})")
        except Exception as e:
            continue

    for url in source_urls:
        cypher_source = """
        MATCH (p:Post {id: $post_id})
        MERGE (s:Source {url: $url})
        MERGE (p)-[:USES_SOURCE]->(s)
        """
        graph_db.query(cypher_source, params={"post_id": post_id, "url": url})

    return "Database Ibrido K-RAG aggiornato con successo!"


def kg_updater(state: AgentState) -> dict:
    print("-> Esecuzione KG Updater...")
    topic = state["current_topic"]
    sources = state.get("verified_resources", [])
    draft = state["draft"]
    trace = state.get("reasoning_trace", [])

    source_urls = [
        res.get("url", f"fonte_sconosciuta_{idx}") for idx, res in enumerate(sources)
    ]

    tools = [aggiorna_knowledge_graph_db]
    db_agent = create_agent(llm, tools)

    prompt = f"""Devi aggiornare il nostro database basato su grafi Neo4j con le informazioni dell'ultimo articolo scritto.
    
    Questi sono i dati che DEVI passare al tool:
    - topic: "{topic}"
    - draft: "{draft}"
    - source_urls: {source_urls}
    
    SPIEGA il tuo ragionamento iniziando con 'Thought: [la tua giustificazione]', poi chiama il tool 'aggiorna_knowledge_graph_db' passandogli in input ESATTAMENTE i tre parametri forniti sopra, dopodichè fermati.
    """

    print("   [KG ReAct] Avvio l'agente per l'inserimento su db...")
    try:
        response = db_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        print("   [KG ReAct] Aggiornamento DB concluso dall'agente!")

        if response:
            messages = response.get("messages", [])
            for msg in messages:
                if (
                    getattr(msg, "type", "") == "ai"
                    and hasattr(msg, "tool_calls")
                    and msg.tool_calls
                ):
                    if msg.content:
                        trace.append(f"Thought: {msg.content.strip()}")
                    for tool_call in msg.tool_calls:
                        trace.append(f"Action: Chiamo tool '{tool_call['name']}'...")
                elif getattr(msg, "type", "") == "tool":
                    trace.append(f"Observation: {msg.content}")

    except Exception as e:
        print(f"   [KG ReAct] Errore nell'esecuzione dell'agente Neo4j: {e}")

    return {"status": "kg_updated", "kg_context": "Neo4j_Active"}


# ==========================================
# 3. DEFINIZIONE DEGLI ARCHI E COMPILAZIONE GRAFO
# ==========================================


def check_resources_quality(
    state: AgentState,
) -> Literal["drafter", "resource_researcher"]:
    if len(state.get("verified_resources", [])) > 0:
        return "drafter"
    return "resource_researcher"


def check_approval(
    state: AgentState,
) -> Literal["kg_updater", "drafter", "topic_planner"]:
    if state["status"] == "approved":
        return "kg_updater"
    elif state["status"] == "rejected_topic":
        return "topic_planner"
    else:
        return "drafter"


workflow = StateGraph(AgentState)

workflow.add_node("topic_planner", topic_planner)
workflow.add_node("resource_researcher", resource_researcher)
workflow.add_node("quality_fact_checker", quality_fact_checker)
workflow.add_node("drafter", drafter)
workflow.add_node("human_review", human_review)
workflow.add_node("kg_updater", kg_updater)

workflow.add_edge(START, "topic_planner")
workflow.add_edge("topic_planner", "resource_researcher")
workflow.add_edge("resource_researcher", "quality_fact_checker")
workflow.add_conditional_edges(
    "quality_fact_checker",
    check_resources_quality,
    {"drafter": "drafter", "resource_researcher": "resource_researcher"},
)
workflow.add_edge("drafter", "human_review")
workflow.add_conditional_edges(
    "human_review",
    check_approval,
    {
        "kg_updater": "kg_updater",
        "drafter": "drafter",
        "topic_planner": "topic_planner",
    },
)
workflow.add_edge("kg_updater", END)

app = workflow.compile()

print("=== GENERAZIONE DEL GRAFO IN CORSO ===")
try:
    img = app.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(img)
except Exception as e:
    print("Errore nella generazione dell'immagine:", e)
print("======================================\n")

# ==========================================
# 4. ESECUZIONE (TEST DELLA PIPE LANGGRAPH)
# ==========================================
print("=== INIZIO PROCESSO LANGGRAPH ===")

initial_state = {
    "user_request": "Scrivi un blog post con la preparazione completa e dettagliata di una famosa ricetta della cucina tradizionale italiana. Voglio imparare ricette autentiche e classiche. Includi solo UNA ricetta per post.",
    "kg_context": None,
    "planned_topics": [],
    "current_topic": "",
    "current_topic_type": "", 
    "editorial_justification": "",
    "raw_resources": [],
    "verified_resources": [],
    "draft": "",
    "human_feedback": "",
    "status": "start",
    "revision_count": 0,
    "rejected_topics": [],
    "reasoning_trace": [],
}

for output in app.stream(initial_state):
    pass
print("\n=== PROCESSO CONCLUSO ===")
