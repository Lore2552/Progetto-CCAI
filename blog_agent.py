import os
import datetime
import json
import chromadb
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.graphs import Neo4jGraph

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
#llm = ChatOllama(model="llama3.1:8b", temperature=0.5)

ddg_search = DuckDuckGoSearchRun()

graph_db = Neo4jGraph(
    url=os.environ.get("NEO4J_URI"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD"),
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection(name="archivio_ricette")

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
def cerca_nei_documenti_locali(query: str) -> str:
    """Cerca nel database vettoriale locale informazioni storiche su ricette passate."""
    print(f"      [RAG Tool] Ricerca in ChromaDB per: '{query}'...")
    risultati = chroma_collection.query(query_texts=[query], n_results=1)
    if risultati and risultati.get('documents') and risultati['documents'][0]:
        documento = risultati['documents'][0][0]

        distanza = risultati['distances'][0][0] if 'distances' in risultati else 0
        print(f"      [RAG Tool] Distanza semantica rilevata: {distanza:.4f}")
        
        if distanza < 1.5:
            print("      [RAG Tool] Documento pertinente! Lo passo all'agente.")
            return f"Documento locale trovato: {documento}"
        else:
            print("      [RAG Tool] Documento trovato ma TROPPO DISTANTE (Low Confidence). Lo scarto.")
            return "Trovato documento ma non pertinente. Cerca sul web."
            
    return "Nessun documento locale trovato. Cerca sul web."


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
        f"DEVI scegliere ricette della cucina italiana COMPLETAMENTE DIVERSE.\n"
        f"DIVIETO ASSOLUTO: Non puoi proporre di nuovo piatti o ricette se sono già presenti nella lista dei vietati.\n"
        f"ATTENZIONE: Scegli UNA SOLA ricetta per argomento.\n"
        f"REQUISITO: Devi identificare i 'Gaps in coverage' (vuoti di copertura) Guardando gli argomenti che abbiamo già trattato, trova sotto-argomenti strettamente correlati ma che non abbiamo mai esplorato.\n"
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
    """Cerca sul web le ultime notizie e informazioni aggiornate."""
    print(f"      [Web Tool] Eseguo ricerca web tramite DuckDuckGo per: '{query}'...")
    try:
        res = ddg_search.invoke(query)
        print(f"      [Web Tool] Ricerca completata. Trovati {len(res)} caratteri.")
        return res
    except Exception as e:
        print(f"      [Web Tool] Errore: {e}")
        return "Errore nella ricerca."


def resource_researcher(state: AgentState) -> dict:
    topic = state["current_topic"]

    trace = state.get("reasoning_trace", [])

    print(f"-> Esecuzione Resource Researcher (con paradigma ReAct) per: {topic}")

    tools = [cerca_sul_web, cerca_nei_documenti_locali]
    react_agent = create_react_agent(llm, tools)

    prompt = f"""Devi raccogliere dati completi su come si prepara questa ricetta italiana: '{topic}'.
    ATTENZIONE: Assicurati di cercare notizie, preparazione, e ingredienti riguardanti UNA SOLA RICETTA.
    
    REGOLE TASSATIVE:
    1. Prima di usare il tool di ricerca, DEVI spiegare il tuo ragionamento iniziando ESATTAMENTE con 'Thought: [la tua giustificazione]'.
    2. NON cercare all'infinito. Usa il tool 'cerca_sul_web' al massimo 1 o 2 volte per raccogliere gli ingredienti.
    3. CONDIZIONE DI USCITA: Appena hai trovato gli ingredienti e i passaggi principali della ricetta, FERMATI IMMEDIATAMENTE. Non chiamare più il tool e scrivi un riassunto finale con i dati che hai trovato.
    """

    print(
        "   [ReAct] Avvio l'agente. Attendi l'elaborazione (potrebbe richiedere un minuto in locale)..."
    )
    try:
        response = react_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        print("   [ReAct] Elaborazione conclusa!")
    except Exception as e:
        print(f"   [Errore ReAct Agent] {e}")
        response = None

    raw = []

    if response:
        messages = response.get("messages", [])

        for msg in messages:

            if (
                getattr(msg, "type", "") == "ai"
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):

                if msg.content:
                    thought_str = f"Thought: {msg.content.strip()}"
                    trace.append(thought_str)
                    print(f"   [ReAct] {thought_str}")

                for tool_call in msg.tool_calls:
                    action_str = f"Action: Chiamo il tool '{tool_call['name']}' con query {tool_call['args']}"
                    trace.append(action_str)
                    print(f"   [ReAct] {action_str}")

            elif getattr(msg, "type", "") == "tool":
                obs_str = f"Observation: Risultato del tool (ID: {msg.tool_call_id}). Letti {len(msg.content)} caratteri."
                trace.append(obs_str)
                print(f"   [ReAct] {obs_str}\n")

                raw.append({"url": "Search Output", "content": msg.content})

        if not raw:
            print("   [Fallback] Nessun tool chiamato o estrazione fallita.")
            search_res = ddg_search.invoke(topic)
            raw.append({"url": f"Search: {topic}", "content": search_res})

    return {"raw_resources": raw, "reasoning_trace": trace, "status": "research_done"}


def quality_fact_checker(state: AgentState) -> dict:
    print("-> Esecuzione Quality & Fact Checker...")
    raw = state.get("raw_resources", [])
    topic = state["current_topic"]
    verified = []

    risultati = graph_db.query("MATCH (t:Topic) RETURN t.name AS name")
    past_topics = [res["name"] for res in risultati if res.get("name")]

    for res in raw:
        content = res.get("content", "")[:1500]
        prompt = (
            f"Sei un Fact Checker. Ricetta richiesta: '{topic}'.\n\n"
            f"KNOWLEDGE GRAPH (RICETTE GIA' SCRITTE): {past_topics}\n"
            f"Frammento di contenuto: '{content}'.\n\n"
            f"RISPONDI 'NO' SE IL CONTENUTO PARLA DI RICETTE GIÀ PRESENTI NEL KNOWLEDGE GRAPH.\n"
            f"Questo contenuto porta informazioni NUOVE, relative alla preparazione di questo piatto, ed è corretto? Rispondi esattamente 'SI' o 'NO'."
        )
        try:
            eval_res = llm.invoke([HumanMessage(content=prompt)]).content.upper()
            if "SI" in eval_res or "SÌ" in eval_res:
                verified.append(res)
                print("   [LLM Evaluator] Fonte approvata.")
            else:
                print(
                    f"   [LLM Evaluator] Fonte scartata (Ripetizione o irrilevante). Risposta: {eval_res}"
                )
        except Exception:
            print("   [LLM Evaluator] Errore API, accetto la fonte per default.")
            verified.append(res)

    return {"verified_resources": verified, "status": "fact_checking_done"}


def drafter(state: AgentState) -> dict:

    print("-> Esecuzione Drafter (Generazione testo in corso)...")

    claims_query = "MATCH (c:Claim) RETURN c.text AS claim LIMIT 5"
    past_claims = [res["claim"] for res in graph_db.query(claims_query)]

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

    chroma_collection.add(
        documents=[draft], 
        metadatas=[{"topic": topic}], 
        ids=[post_id]
    )
    print(f"   [ChromaDB] Vettori indicizzati per: {topic}")

    snippet = draft[:100]

    prompt_triple = (
        f"Analizza questo testo culinario: '{draft}'.\n"
        f"Estrai esattamente 3 relazioni fondamentali in formato tripla.\n"
        f"Usa ESATTAMENTE il formato: Soggetto | RELAZIONE_IN_MAIUSCOLO | Oggetto\n"
        f"Esempio: Ragù alla Bolognese | RICHIEDE | Carne macinata\n"
        f"Rispondi SOLO con le 3 triple, una per riga."
    )
    try:
        claims_response = llm.invoke([HumanMessage(content=prompt_triple)]).content
        triple = [c.strip() for c in claims_response.split('\n') if '|' in c]
    except:
        triple = []

    count_res = graph_db.query("MATCH (p:Post) RETURN count(p) AS totale")
    post_count = count_res[0]["totale"] if count_res else 0
    post_id = f"Post_{post_count + 1}"

    cypher_post = """
    MERGE (t:Topic {name: $topic})
    CREATE (p:Post {id: $post_id, snippet: $snippet})
    MERGE (p)-[:COVERS_TOPIC]->(t)
    WITH t, p
    MATCH (old:Topic) WHERE old.name <> $topic
    WITH t, p, old ORDER BY old.name DESC LIMIT 1
    MERGE (t)-[:RELATED_TO]->(old)
    """

    graph_db.query(
        cypher_post, params={"topic": topic, "post_id": post_id, "snippet": snippet}
    )

    for tripla in triple:
        try:
            sog, rel, obj = [item.strip() for item in tripla.split('|')]
            rel_clean = rel.replace(" ", "_").upper()
            
            cypher_triple = f"""
            MATCH (p:Post {{id: $post_id}})
            MERGE (s:Entity {{name: $sog}})
            MERGE (o:Entity {{name: $obj}})
            MERGE (s)-[:{rel_clean}]->(o)
            MERGE (p)-[:MENTIONS]->(s)
            """
            graph_db.query(cypher_triple, params={"post_id": post_id, "sog": sog, "obj": obj})
        except Exception as e:
            continue

    for url in source_urls:
        cypher_source = """
        MATCH (p:Post {id: $post_id})
        MERGE (s:Source {url: $url})
        MERGE (p)-[:USES_SOURCE]->(s)
        """
        graph_db.query(cypher_source, params={"post_id": post_id, "url": url})

        print(f"   [Neo4j] Tripla inserita: ({sog}) -[{rel_clean}]-> ({obj})")

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
    db_agent = create_react_agent(llm, tools)

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
            <h2 class="post-title">{topic}</h2>
            <div class="post-meta">Pubblicato il {date_str}</div>
            <div class="post-content">
                {formatted_draft}
            </div>
            <div class="post-tags">
                <span class="tag">AI Generated</span>
                
                <span class="tag">K-RAG Verified</span> 
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
