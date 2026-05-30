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
from bs4 import BeautifulSoup
import datetime

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
# llm = ChatOllama(model="llama3.1:8b", temperature=0.5)

ddg_search = DuckDuckGoSearchAPIWrapper(max_results=5)

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


# ==========================================
# 1. DEFINIZIONE DELLO STATO
# ==========================================
class AgentState(TypedDict):
    user_request: str
    kg_context: Any
    planned_topics: List[str]
    current_topic: str
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


@tool
def kg_rag_tool(topic: str) -> str:
    """
    Usa questo tool per ottenere il contesto completo di una ricetta combinando dati strutturati (Knowledge Graph) e testi estesi (Vector Database).
    Fornisce gli ingredienti obbligatori, le tecniche e i passaggi di preparazione dettagliati.
    """
    print(f"      [KG-RAG Tool] Avvio recupero ibrido per la ricetta: '{topic}'...")
    
    cypher_query = """
    MATCH (r:Recipe {name: $topic})
    OPTIONAL MATCH (r)-[:USES_INGREDIENT]->(i:Ingredient)
    OPTIONAL MATCH (r)-[:HAS_TECHNIQUE]->(t:Technique)
    RETURN r.name AS recipe, 
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
    expanded_query = topic

    if risultato and risultato[0]['recipe'] is not None:
        dati = risultato[0]
        ingredienti_str = ", ".join(dati['ingredients'])
        tecniche_str = ", ".join(dati['techniques'])

        expanded_query = f"{topic} {ingredienti_str} {tecniche_str}".strip()
        print(f"      [KG-RAG Tool] Trovate info nel KG. Query espansa: '{expanded_query}'")
    else:
        print("      [KG-RAG Tool] Nessuna informazione strutturata trovata nel KG. Uso la query base.")

    try:
        risultati_chroma = collection_ricette.query(query_texts=[expanded_query], n_results=3)
        testo_recuperato = ""
        
        if risultati_chroma and risultati_chroma.get("documents") and risultati_chroma["documents"][0]:
            documenti = risultati_chroma["documents"][0]
            testo_recuperato = "\n\n--- FRAMMENTO VETTORIALE ---\n".join([doc[:2000] for doc in documenti])
        else:
            testo_recuperato = "Nessun documento testuale di supporto trovato nel database locale."
    except Exception as e:
        testo_recuperato = f"Errore durante la ricerca in ChromaDB: {e}"

    risposta_tool = f"RISULTATI DEL RECUPERO KG-RAG PER '{topic}':\n\n"
    risposta_tool += f"[1] DATI STRUTTURATI (REGOLE TASSATIVE DAL KNOWLEDGE GRAPH):\n"
    risposta_tool += f"- Ingredienti Obbligatori: {ingredienti_str if ingredienti_str else 'Nessuno specificato'}\n"
    risposta_tool += f"- Tecniche Richieste: {tecniche_str if tecniche_str else 'Nessuna specificata'}\n\n"
    risposta_tool += f"[2] TESTI DETTAGLIATI (DAL VECTOR DATABASE):\n"
    risposta_tool += testo_recuperato

    return risposta_tool


@tool
def valuta_documento_locale(document_text: str, target_topic: str) -> str:
    """Valuta in modo critico se il testo della ricetta fornito è PERTINENTE rispetto al target_topic richiesto."""
    print(
        f"      [Fact Checker Tool] Valutazione documento rispetto al target: '{target_topic}'..."
    )
    prompt = (
        f"Il topic esatto da trattare è: '{target_topic}'.\n"
        f"Ecco il testo trovato nel database locale:\n{document_text}\n\n"
        f"Il testo contiene le istruzioni esatte per '{target_topic}'? Rispondi in modo secco: 'SI, COPRE INTERAMENTE', oppure 'PARZIALE' (es. manca una specifica variante), oppure 'NO'."
    )
    try:
        return llm.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        return f"Errore valutazione: {e}"


@tool
def cerca_e_leggi_sul_web(query: str) -> str:
    """Cerca sul web (tramite DuckDuckGo), estrae HTML e restituisce direttamente il testo completo delle pagine trovate senza passare URL json."""
    print(f"      [Web Tool] Eseguo ricerca web e Scraping DIRETTO per: '{query}'...")
    try:
        results = ddg_search.results(query, max_results=3)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        testo_estratto = ""
        for r in results:
            url = r["link"]
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(resp.content, "html.parser")

                # Rimuoviamo il rumore di fondo (menu, commenti, ecc.)
                for unwanted in soup(
                    [
                        "nav",
                        "header",
                        "footer",
                        "script",
                        "style",
                        "aside",
                        "form",
                        "meta",
                        "noscript",
                    ]
                ):
                    unwanted.extract()

                # Rimuoviamo blocchi spesso contenti rumore (sidebar, widgets, commenti)
                for unwanted in soup.find_all(
                    lambda tag: tag.has_attr("class")
                    and any(
                        c in str(tag["class"]).lower()
                        for c in [
                            "comment",
                            "sidebar",
                            "widget",
                            "menu",
                            "social",
                            "share",
                        ]
                    )
                ):
                    unwanted.extract()
                for unwanted in soup.find_all(
                    lambda tag: tag.has_attr("id")
                    and any(
                        i in str(tag["id"]).lower()
                        for i in ["comment", "sidebar", "widget", "menu"]
                    )
                ):
                    unwanted.extract()

                # Tentiamo di isolare il contenuto principale
                main_content = (
                    soup.find("article")
                    or soup.find("main")
                    or soup.find(
                        "div", class_=lambda c: c and "content" in str(c).lower()
                    )
                    or soup.find("body")
                    or soup
                )

                # Estraiamo il testo dividendo col newline e ripulendo gli spazi multipli
                testo_sporco = main_content.get_text(separator="\n", strip=True)
                testo_pulito = "\n".join(
                    [line.strip() for line in testo_sporco.splitlines() if line.strip()]
                )

                if len(testo_pulito) > 100:
                    testo_estratto += (
                        f"\n--- FONTE (URL: {url}) ---\n{testo_pulito[:8000]}\n"
                    )
            except Exception:
                pass

        if not testo_estratto:
            return "Non è stato possibile estrarre i contenuti delle pagine."
        return testo_estratto
    except Exception as e:
        print(f"      [Web Tool] Errore: {e}")
        return "Errore nella ricerca."


def topic_planner(state: AgentState) -> dict:
    print("-> Esecuzione Topic Planner...")

    cypher_query = "MATCH (t:Topic) RETURN t.name AS name"
    risultati = graph_db.query(cypher_query)

    past_topics = [res["name"] for res in risultati if res.get("name")]
    rejected_topics = state.get("rejected_topics", [])

    all_avoids = past_topics + rejected_topics

    print(f"   [KG] Argomenti passati trovati: {past_topics}")
    if rejected_topics:
        print(f"   [Feedback] Argomenti scartati in questa sessione: {rejected_topics}")

    prompt = (
        f"La richiesta dell'utente è: '{state['user_request']}'.\n"
        f"KNOWLEDGE GRAPH (RICETTE GIÀ SCRITTE): {all_avoids}.\n"
        f"DEVI scegliere una nuova ricetta della cucina italiana COMPLETAMENTE DIVERSA da quelle scritte.\n"
        f"DIVIETO ASSOLUTO: Non puoi proporre piatti già presenti nella lista dei vietati.\n"
        f"REQUISITO: Identifica un 'Gap in coverage' (vuoto di copertura) esplorando qualcosa di nuovo.\n"
        f"REGOLA DI FORMATTAZIONE TASSATIVA: Rispondi SOLO ED ESCLUSIVAMENTE con il NOME DELLA RICETTA. "
        f"Non aggiungere descrizioni, non aggiungere ingredienti, non scrivere premesse. "
        f"Esempio di risposta corretta: Pappardelle al Cinghiale"
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
        planned = (
            [t.strip() for t in response.split(",")]
            if "," in response
            else [response.strip()]
        )
    except Exception as e:
        print(f"   [Errore LLM] {e}")
        planned = ["Default Topic 1", "Default Topic 2"]

    print(f"   [LLM] Argomenti pianificati: {planned}")
    return {
        "planned_topics": planned,
        "current_topic": planned[0],
        "status": "planning_done",
    }


@tool
def cerca_sul_web(query: str) -> str:
    """Restituisce un JSON sotto forma di stringa contenente i link trovati sul web tramite DuckDuckGo Search per la query specificata."""
    print(f"      [Web Tool] Eseguo ricerca web tramite DuckDuckGo per: '{query}'...")
    try:
        results = ddg_search.results(query, max_results=6)
        raw_results = []
        for r in results:
            raw_results.append(
                {"url": r["link"], "snippet": r["snippet"], "title": r["title"]}
            )
        print(f"      [Web Tool] Trovati {len(raw_results)} risultati raw.")
        return json.dumps(raw_results)
    except Exception as e:
        print(f"      [Web Tool] Errore: {e}")
        return json.dumps([])


def resource_researcher(state: AgentState) -> dict:
    topic = state["current_topic"]
    trace = state.get("reasoning_trace", [])

    print(f"-> Esecuzione Resource Researcher (con paradigma ReAct) per: {topic}")

    tools = [kg_rag_tool, valuta_documento_locale, cerca_e_leggi_sul_web]
    react_agent = create_agent(llm, tools)

    prompt = f"""Devi raccogliere dati completi su come si prepara questa ricetta italiana: '{topic}'.
    Hai a disposizione 2 tool per la ricerca:
    1. 'kg_rag_tool': Cerca la ricetta nel database locale ibrido. Restituisce le regole fisse del Knowledge Graph (ingredienti e tecniche obbligatorie) e il frammento di testo completo della ricetta.
    2. 'cerca_e_leggi_sul_web': Cerca ed estrae integralmente i passaggi da pagine internet. Utilizzalo se la ricetta locale è assente, parziale o per colmare i dettagli mancanti.

    Flusso Operativo Richiesto:
    1. REASONING: Inizia ogni azione con 'Thought: ...'
    2. Usa SEMPRE come prima azione il tool 'kg_rag_tool'.
    3. Valuta tu stesso il risultato ottenuto. Se il testo contiene le istruzioni complete per '{topic}' e rispetta gli ingredienti/tecniche obbligatorie, FERMATI ed esponi tu una sintesi che unisce ingredienti e passaggi di preparazione del piatto in un testo descrittivo.
    4. Se valuti che mancano dati specifici al topic, o se il tool locale non trova nulla, usa 'cerca_e_leggi_sul_web' per ottenere l'intera ricetta dal web in un colpo solo.
    5. Alla fine di tutti i controlli, integra ciò che hai ottenuto di buono sia in locale (se pertinente) che sul web. Come TUA risposta FINALE restituisci un unico riassunto descrittivo che contenga chiaramente la lista degli ingredienti (rispettando rigorosamente i dati del Knowledge Graph) e la preparazione. Non inviare json.
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

        raw = [{"url": "ReAct Synthesis", "content": final_answer, "title": topic}]

    except Exception as e:
        print(f"   [Errore ReAct Agent] {e}")
        raw = []

    return {"raw_resources": raw, "reasoning_trace": trace, "status": "research_done"}


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
    sources = state.get("verified_resources", [])
    topic = state["current_topic"]

    sources_text = "\n".join([s.get("content", "")[:1000] for s in sources])

    prompt = (
        f"Scrivi un coinvolgente articolo di blog su come preparare la ricetta: '{topic}'.\n"
        f"Usa questi frammenti di ricerca come contesto primario:\n{sources_text}\n\n"
        f"KNOWLEDGE GRAPH (AFFERMAZIONI PASSATE DEL BLOG): {past_claims}\n"
        f"REGOLE FONDAMENTALI (PENA IL RIFIUTO DELLO SCRITTO):\n"
        f"1. L'articolo deve parlare ESCLUSIVAMENTE di UNA SOLA RICETTA. Se i frammenti mescolano più ricette, SCEGLINE SOLO UNA (quella inerente a '{topic}') e IGNORA tutto il resto. Vieta categoricamente di mischiare preparazioni diverse.\n"
        f"2. Inizia la tua risposta ESATTAMENTE con 'TITOLO: <il tuo titolo accattivante>' sulla primissima riga.\n"
        f"3. L'articolo deve essere in italiano.\n"
        f"4. Assicurati che l'articolo sia COERENTE (consistency) con le affermazioni passate. Non contraddirle.\n"
        f"5. Se pertinente, CONNETTI (connect with existing topics) il nuovo post a una di queste vecchie informazioni.\n"
        f"6. Scrivi in modo diretto come se fossi uno chef (VIETATO usare scuse, premesse o frasi come 'ecco l'articolo')."
        f"7. L'articolo deve conterenere necessariamente la lista degli ingredienti della ricetta trattata.\n"
        f"8. L'articolo deve contenere necessariamente la preparazione dettagliata della ricette trattata.\n"
    )

    if feedback and previous_draft:
        prompt += f"\n\nEcco la tua BOZZA PRECEDENTE:\n---\n{previous_draft}\n---\n\nCRITICO: L'utente ha rifiutato la bozza precedente con questo feedback: '{feedback}'. DEVI riscrivere pesantemente la bozza. Ricorda SEMPRE di focalizzarti su UNA SOLA RICETTA."

    try:
        response_text = llm.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        print(
            f"\n[ERRORE FATALE] Impossibile generare la bozza a causa di un errore API: {e}"
        )
        raise SystemExit(1)

    new_title = topic
    draft = response_text

    lines = response_text.split("\n")
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("TITOLO:"):
            new_title = (
                line.split(":", 1)[1]
                .strip()
                .replace("**", "")
                .replace('"', "")
                .replace("'", "")
            )
            draft = "\n".join(lines[:i] + lines[i + 1 :]).strip()
            break
        elif line.strip().upper().startswith("**TITOLO:**"):
            new_title = (
                line.split(":", 1)[1]
                .strip()
                .replace("**", "")
                .replace('"', "")
                .replace("'", "")
            )
            draft = "\n".join(lines[:i] + lines[i + 1 :]).strip()
            break

    return {
        "draft": draft,
        "current_topic": new_title,
        "status": "draft_ready",
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

    lbl = tk.Label(
        root,
        text="Ecco la bozza generata. Scegli se approvare, chiedere modifiche o scartare l'argomento:",
        font=("Helvetica", 12, "bold"),
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
    WITH t, p
    MATCH (old:Topic) WHERE old.name <> $topic
    WITH t, p, old ORDER BY old.name DESC LIMIT 1
    MERGE (t)-[:RELATED_TO]->(old)
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
    print("-> Esecuzione KG Updater & Web Publisher (tramite ReAct)...")
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

    html_path = "index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        date_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        formatted_draft = state["draft"].replace("\n", "<br>")

        new_post_html = f"""
        <article class="post-card">
            <h2 class="post-title">👨‍🍳 {topic}</h2>
            <div class="post-meta">Pubblicato il {date_str}</div>
            <div class="post-content">
                {formatted_draft}
            </div>
            <div class="post-tags">
                <span class="tag">AI Generated</span>
                <span class="tag">Cucina Italiana</span>
            </div>
        </article>
        """

        html_content = html_content.replace("<!-- NEW_POSTS_HERE -->", new_post_html)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print("   [Web Publisher] Post aggiunto al file index.html!")

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
