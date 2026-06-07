import json
import os
from typing import List
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import trafilatura
from rank_bm25 import BM25Okapi
from langchain_neo4j import Neo4jGraph

from module_rag import collection_ricette, collection_posts, rrf_fusion, cohere_reranker

KG_FILE = "knowledge_graph.graphml"

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
ddg_search = DuckDuckGoSearchAPIWrapper(max_results=5)

graph_db = Neo4jGraph(
    url=os.environ.get("NEO4J_URI"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD"),
    database=os.environ.get("NEO4J_DATABASE"),
)


@tool
def kg_rag_tool(topic: str) -> str:
    """Usa questo tool per ottenere il contesto completo di una ricetta combinando dati strutturati (Knowledge Graph)
    e testi estesi (Vector Database) con tecniche avanzate di Hybrid Search (Dense + BM25), Fusione RRF e Cohere Reranking.
    """

    print(
        f"      [KG-RAG Tool] Avvio recupero ibrido avanzato per la ricetta: '{topic}'..."
    )

    cypher_query = """
    WITH split(toLower($topic), " ") AS parole
    MATCH (r:Recipe)
    WHERE any(parola IN parole WHERE size(parola) > 3 AND toLower(r.title) CONTAINS parola)
    
    OPTIONAL MATCH (r)-[:USES_INGREDIENT]->(i:Ingredient)
    OPTIONAL MATCH (r)-[:USES_TECHNIQUE]->(t:Technique)
    
    WITH r, i, t, [p IN parole WHERE size(p) > 3 AND toLower(r.title) CONTAINS p] AS matches
    RETURN r.title AS recipe, 
           r.url AS url,
           collect(DISTINCT i.name) AS ingredients, 
           collect(DISTINCT t.name) AS techniques,
           size(matches) AS score
    ORDER BY score DESC
    LIMIT 1
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
        expanded_query = f"{topic} {ingredienti_str} {tecniche_str}".strip()
        print(
            f"      [KG-RAG Tool] Trovate info nel KG. Query espansa: '{expanded_query}'"
        )
    else:
        print(
            "      [KG-RAG Tool] Nessuna informazione strutturata trovata nel KG. Uso la query base."
        )

    id_to_metadata = {}
    try:
        tutti_i_doc = collection_ricette.get()
        all_documents = tutti_i_doc.get("documents", [])
        all_ids = tutti_i_doc.get("ids", [])
        all_metadatas = tutti_i_doc.get("metadatas", []) or []

        for idx, doc_id in enumerate(all_ids):
            if idx < len(all_metadatas) and all_metadatas[idx]:
                id_to_metadata[doc_id] = all_metadatas[idx]
    except Exception as e:
        print(f"Errore recupero documenti complessivi per BM25: {e}")
        all_documents, all_ids = [], []

    testo_recuperato = ""
    documenti_recuperati = []

    if all_documents:
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
                    if idx < len(chroma_metas) and chroma_metas[idx]:
                        id_to_metadata[doc_id] = chroma_metas[idx]
                    dense_docs.append({"id": doc_id, "text": doc_text[:2000]})
        except Exception as e:
            print(f"Errore durante la ricerca Dense in ChromaDB: {e}")

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

        fused_docs = rrf_fusion(
            dense_results=dense_docs, keyword_results=keyword_docs, k=60, top_n=20
        )
        final_docs = cohere_reranker(
            query=expanded_query, documents=fused_docs, top_n=10
        )

        if final_docs:
            documenti_recuperati = [
                {
                    "id": doc.get("id"),
                    "testo": doc.get("text", ""),
                    "indice": idx + 1,
                    "metadata": id_to_metadata.get(doc.get("id"), {}),
                }
                for idx, doc in enumerate(final_docs)
            ]

    document_ids = [doc["id"] for doc in documenti_recuperati if doc.get("id")]

    payload_leggero = {
        "topic": topic,
        "recipe_url": recipe_url,
        "ingredienti": ingredienti_str,
        "tecniche": tecniche_str,
        "document_ids": document_ids,
        "messaggio": (
            "Recupero KG-RAG completato. "
            "Usa valuta_documento_locale passando target_topic, document_ids, "
            "recipe_url, ingredienti e tecniche."
        ),
    }

    return json.dumps(payload_leggero, ensure_ascii=False)


@tool
def cerca_nel_database_ismea(ingredienti_ricetta: list) -> str:
    """
    Cerca i prezzi, le alternative BIO e suggerisce i vini per una ricetta.
    Usa un LLM interno per tradurre i dialetti, fare i calcoli matematici e ragionare sugli abbinamenti.
    """
    print(
        f"      [ISMEA Tool] Avvio LLM per mappare e calcolare: {ingredienti_ricetta}..."
    )

    try:
        with open(
            "dataset_scripts/database_unico_ismea.json", "r", encoding="utf-8"
        ) as f:
            db = json.load(f)
    except Exception as e:
        return f"Errore: Impossibile accedere al database ISMEA. Dettagli: {e}"

    catalogo_standard = {
        k: f"{v['prezzo']} {v['um']}"
        for k, v in db.get("ingredienti_standard", {}).items()
    }
    catalogo_bio = {
        k: f"{v['prezzo']} {v['um']}" for k, v in db.get("ingredienti_bio", {}).items()
    }

    prompt = f"""Sei un autorevole esperto culinario, un sommelier professionista e un matematico finanziario di precisione assoluta.
    Analizza questa lista di ingredienti estratti da una ricetta, che include le quantità specifiche richieste:
    {ingredienti_ricetta}

    CATALOGO REALE ISMEA (NOME: PREZZO BASE E UNITÀ DI MISURA ALL'INGROSSO):
    - Standard: {catalogo_standard}
    - BIO: {catalogo_bio}

    COMPITO 1 (MAPPATURA SEMANTICA E SINONIMI): 
    Il catalogo ISMEA usa termini commerciali all'ingrosso. Devi mappare gli ingredienti della ricetta usando i sinonimi logici del catalogo:
    - "Zucchero" o "Zucchero velo" -> Cerca "Zucchero" o "Saccarosio"
    - "Farina Manitoba" o "Farina" -> Cerca "Frumento tenero" o "Farina di frumento"
    - "Burro" -> Cerca "Burro" o "Latte vaccino (freschi)"
    - "Rum" o alcolici -> Se assenti nel catalogo, non inventare i prezzi, lasciali vuoti.
    Se trovi un match logico, procedi al calcolo. Se è totalmente assente, non inserirlo nel JSON.

    COMPITO 2 (CALCOLO MATEMATICO PROPORZIONALE DELLA PORZIONE):
    Calcola il costo ESATTO della quantità richiesta nella ricetta basandoti sul prezzo all'ingrosso del catalogo.
    ⚠️ REGOLE DI CONVERSIONE TASSATIVE PER IL MATEMATICO:
    - Se l'unità del catalogo è 't' (Tonnellata): dividi il prezzo per 1.000.000 per ottenere il prezzo al grammo, poi moltiplica per i grammi esatti della ricetta. (Es: Frumento 400 €/t -> 400 / 1.000.000 = 0.0004 €/g. Per 300 g di farina: 0.0004 * 300 = 0.12 €).
    - Se l'unità del catalogo è 'q.le' o '100 kg' (Quintale): dividi il prezzo per 100.000 per ottenere il prezzo al grammo, poi moltiplica per i grammi della ricetta.
    - Se l'unità del catalogo è '100 unità' o '100 pezzi': dividi il prezzo per 100 per trovare il prezzo del singolo pezzo, poi moltiplica per il numero di pezzi richiesti (Es: Uova ricetta = 3. Catalogo = 14.60 € / 100 pezzi -> 0.146 € a uovo. Per 3 uova = 0.146 * 3 = 0.44 €).
    - Se la quantità nella ricetta usa frazioni come '½', considerala come metà unità (0.5) o convertila in grammi stimati (es. scorza = 5g).
    - Se la quantità è "q.b.", restituisci il prezzo convertito al chilogrammo finito (es. "Circa 1.20 €/Kg").

    Formatta il campo "prezzo_finale" esattamente così: "Circa X.XX € (per VALORE_QUANTITÀ_RICETTA)" (Es: "Circa 0.12 € (per 300 g)" oppure "Circa 1.32 € (per 3 uova)").

    COMPITO 3 (ABBINAMENTO VINO DA SOMMELIER):
    Seleziona un vino specifico in abbinamento e spiega la scelta.

    Rispondi ESCLUSIVAMENTE con un JSON valido, senza markdown (NO ```json), strutturato così:
    {{
        "ingredienti_trovati": [
            {{"richiesto": "nome_originale", "trovato": "nome_nel_catalogo", "prezzo_finale": "costo calcolato proporzionale"}}
        ],
        "alternative_bio": [
            {{"richiesto": "nome_originale", "trovato": "nome_nel_catalogo_bio", "prezzo_finale": "costo bio calcolato proporzionale"}}
        ],
        "vino": "Nome del vino e motivazione."
    }}
    """

    try:
        risposta_llm = llm.invoke([("human", prompt)]).content
        risposta_pulita = risposta_llm.replace("```json", "").replace("```", "").strip()
        ragionamento = json.loads(risposta_pulita)
    except Exception as e:
        print(f"      [ISMEA Tool - Errore LLM interno] {e}")
        return "Errore durante il ragionamento semantico e calcolo dei prezzi."

    risultati = {"prezzi_trovati": [], "alternative_bio_suggerite": []}
    for item in ragionamento.get("ingredienti_trovati", []):
        risultati["prezzi_trovati"].append(
            f"- {item['richiesto']} ({item['trovato']}): {item['prezzo_finale']}"
        )
    for item in ragionamento.get("alternative_bio", []):
        risultati["alternative_bio_suggerite"].append(
            f"- Variante Naturale: Sostituisci '{item['richiesto']}' con '{item['trovato'].title()} BIO' ({item['prezzo_finale']})"
        )

    output = (
        "RISULTATI RICERCA ISMEA:\nPrezzi da usare:\n"
        + (
            "\n".join(risultati["prezzi_trovati"])
            if risultati["prezzi_trovati"]
            else "Nessuno."
        )
        + "\n\n"
    )
    output += (
        "Alternative BIO:\n"
        + (
            "\n".join(risultati["alternative_bio_suggerite"])
            if risultati["alternative_bio_suggerite"]
            else "Nessuna."
        )
        + "\n\n"
    )
    output += "Vino:\n" + ragionamento.get("vino", "Nessun abbinamento.")
    return output


@tool
def valuta_documento_locale(
    target_topic: str,
    document_ids: List[str],
    recipe_url: str = "Non disponibile",
    ingredienti: str = "",
    tecniche: str = "",
) -> str:
    """Valuta in modo critico se i testi delle ricette recuperate nel database locale sono PERTINENTI rispetto al target_topic richiesto."""

    print(
        f"      [Fact Checker Tool] Valutazione documenti rispetto al target: '{target_topic}'..."
    )

    nuovo_testo = (
        "Nessun documento locale pertinente o parziale trovato dopo la validazione."
    )
    valutazioni = []
    documenti_validi = []
    document_ids = document_ids or []

    if not document_ids:
        return json.dumps(
            {
                "target_topic": target_topic,
                "valutazioni": [],
                "esito_generale": "NO",
                "messaggio": "Nessun document_id ricevuto da valutare.",
            },
            ensure_ascii=False,
        )

    try:
        chroma_result = collection_ricette.get(ids=document_ids)
        docs = chroma_result.get("documents", []) or []
        ids = chroma_result.get("ids", []) or []
        metadatas = chroma_result.get("metadatas", []) or []
        documenti_recuperati = []

        for idx, doc_id in enumerate(ids):
            documenti_recuperati.append(
                {
                    "id": doc_id,
                    "testo": docs[idx] if idx < len(docs) else "",
                    "indice": idx + 1,
                    "metadata": (
                        metadatas[idx]
                        if idx < len(metadatas) and metadatas[idx]
                        else {}
                    ),
                }
            )
    except Exception as e:
        return json.dumps(
            {
                "target_topic": target_topic,
                "valutazioni": [],
                "esito_generale": "NO",
                "messaggio": f"Errore nel recupero documenti da ChromaDB: {e}",
            },
            ensure_ascii=False,
        )

    if not documenti_recuperati:
        return json.dumps(
            {
                "target_topic": target_topic,
                "valutazioni": [],
                "esito_generale": "NO",
                "messaggio": "Nessun documento trovato in ChromaDB per gli ID ricevuti.",
            },
            ensure_ascii=False,
        )

    risultato_kg_rag = {
        "metadata_kg": {
            "topic": target_topic,
            "url": recipe_url,
            "ingredienti": ingredienti,
            "tecniche": tecniche,
        },
        "documenti_recuperati": documenti_recuperati,
    }

    for doc in documenti_recuperati:
        indice = doc.get("indice")
        doc_id = doc.get("id")
        document_text = doc.get("testo", "")
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
            valutazione = f"NO - Errore: {e}"

        val_upper = valutazione.upper()
        is_valid = val_upper.startswith("SI") or val_upper.startswith("PARZIALE")
        if is_valid:
            documenti_validi.append(doc)
            valutazioni.append(
                {
                    "indice": doc["indice"],
                    "id": doc["id"],
                    "titolo_originale": titolo_ricetta_originale,
                    "valutazione": valutazione,
                    "status": "ACCETTATO",
                }
            )

    if documenti_validi:
        nuovo_testo = "\n\n--- FRAMMENTO VETTORIALE ---\n".join(
            [
                f"[DOCUMENTO {d['indice']} - ID: {d['id']} - TITOLO ORIGINALE: {d.get('metadata', {}).get('titolo', 'Sconosciuto')}]\n{d['testo']}"
                for d in documenti_validi
            ]
        )
    else:
        nuovo_testo = (
            "Nessun documento locale pertinente o parziale trovato dopo la validazione."
        )

    risultato_kg_rag["documenti_recuperati"] = documenti_validi
    risultato_kg_rag["testo_recuperato"] = nuovo_testo

    meta = risultato_kg_rag["metadata_kg"]
    risposta_filtrata = f"RISULTATI DEL RECUPERO IBRIDO AVANZATO PER '{meta.get('topic', target_topic)}':\n\n"
    risposta_filtrata += (
        "[1] DATI STRUTTURATI (REGOLE TASSATIVE DAL KNOWLEDGE GRAPH):\n"
    )
    risposta_filtrata += f"- URL Ricetta: {meta.get('url', 'N/A')}\n"
    risposta_filtrata += (
        f"- Ingredienti Obbligatori: {meta.get('ingredienti', 'Nessuno specificato')}\n"
    )
    risposta_filtrata += (
        f"- Tecniche Richieste: {meta.get('tecniche', 'Nessuna specificata')}\n\n"
    )
    risposta_filtrata += (
        f"[2] TESTI DETTAGLIATI (FILTRATI SOLO PER PERTINENZA):\n{nuovo_testo}"
    )
    risultato_kg_rag["risposta_finale"] = risposta_filtrata

    return json.dumps(
        {
            "target_topic": target_topic,
            "valutazioni": valutazioni,
            "documenti_validi": len(documenti_validi),
            "esito_generale": "SI" if documenti_validi else "NO",
            "kg_rag_payload_filtrato": risultato_kg_rag,
        },
        ensure_ascii=False,
    )


@tool
def cerca_e_leggi_sul_web(query: str) -> str:
    """Cerca sul web, estrae il main content e restituisce il testo di PIÙ pagine trovate per permettere il confronto."""
    print(f"      [Web Tool] Eseguo ricerca web per: '{query}'...")

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
                except:
                    pass
            if opzioni_estratte:
                return "".join(opzioni_estratte)
        except Exception as e:
            print(f"      [Web Tool] Errore tentativo {attempt+1}: {e}")
    return "Non è stato possibile estrarre i contenuti delle pagine."


@tool
def aggiorna_knowledge_graph_db(topic: str, draft: str, source_urls: List[str]) -> str:
    """Aggiorna il K-RAG: Inserisce i vettori in ChromaDB e le triple nel Knowledge Graph Neo4j."""

    cypher_query = "MATCH (p:Post) RETURN count(p) AS totale"
    count_res = graph_db.query(cypher_query)
    post_count = count_res[0]["totale"] if count_res else 0
    post_id = f"Post_{post_count + 1}"

    collection_posts.add(documents=[draft], metadatas=[{"topic": topic}], ids=[post_id])
    print(f"   [ChromaDB] Vettori indicizzati per: {topic}")

    prompt = (
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
        claims_response = llm.invoke([HumanMessage(content=prompt)]).content
        triple = [c.strip() for c in claims_response.split("\n") if "|" in c]
    except:
        triple = []

    count_res = graph_db.query(cypher_query)
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
        except:
            continue

    for url in source_urls:
        cypher_source = """
        MATCH (p:Post {id: $post_id})
        MERGE (s:Source {url: $url})
        MERGE (p)-[:USES_SOURCE]->(s)
        """
        graph_db.query(cypher_source, params={"post_id": post_id, "url": url})

    return "Database Ibrido K-RAG aggiornato con successo!"
