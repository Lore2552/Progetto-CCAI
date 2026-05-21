import os
import datetime
import json
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Literal
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from IPython.display import Image, display, Markdown, HTML
import networkx as nx
from networkx.readwrite import json_graph

load_dotenv()

# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1) 
llm = ChatOllama(model="llama3.1:8b", temperature=0.5)

ddg_search = DuckDuckGoSearchRun()

KG_FILE = "knowledge_graph.graphml"

def load_kg() -> nx.DiGraph:
    if os.path.exists(KG_FILE):
        try:
            return nx.read_graphml(KG_FILE)
        except Exception:
            return nx.DiGraph()
    return nx.DiGraph()

def save_kg(G: nx.DiGraph):
    nx.write_graphml(G, KG_FILE)

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


from langchain_core.tools import tool
from langchain.agents import create_agent

def topic_planner(state: AgentState) -> dict:
    print("-> Esecuzione Topic Planner...")
    
    G = load_kg()

    
    past_topics = [n for n, attributes in G.nodes(data=True) if attributes.get("type") == "Topic"]
    rejected_topics = state.get("rejected_topics", [])

    all_avoids = past_topics + rejected_topics
    
    print(f"   [KG] Argomenti passati trovati: {past_topics}")
    if rejected_topics:
        print(f"   [Feedback] Argomenti scartati in questa sessione: {rejected_topics}")
    
    prompt = (
        f"La richiesta dell'utente è: '{state['user_request']}'.\n"
        f"KNOWLEDGE GRAPH (ARGOMENTI/ATLETI VIETATI GIÀ TRATTATI): {all_avoids}.\n"
        f"DEVI scegliere argomenti COMPLETAMENTE DIVERSI.\n"
        f"DIVIETO ASSOLUTO: Non puoi proporre di nuovo atleti (es. Djokovic, Alcaraz o altri) se sono già presenti nella lista dei vietati.\n"
        f"ATTENZIONE: Scegli UN SOLO sport per argomento. Non creare argomenti che mescolano più sport.\n"
        f"Fornisci esattamente 2 nuovi argomenti interessanti separati da virgola. Rispondi SOLO con gli argomenti, nessun altro testo."
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content
        planned = [t.strip() for t in response.split(",")] if "," in response else [response.strip()]
    except Exception as e:
        print(f"   [Errore LLM] {e}")
        planned = ["Default Topic 1", "Default Topic 2"]
        
    print(f"   [LLM] Argomenti pianificati: {planned}")
    return {"planned_topics": planned, "current_topic": planned[0], "status": "planning_done"}

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
    topic = state['current_topic']

    trace = state.get("reasoning_trace", [])

    print(f"-> Esecuzione Resource Researcher (con paradigma ReAct) per: {topic}")

    tools = [cerca_sul_web]
    react_agent = create_agent(llm, tools)
    
    prompt = f"""Devi raccogliere le ultime notizie su: '{topic}'.
    ATTENZIONE: Assicurati di cercare notizie riguardanti UN SOLO SPORT (quello specificato in '{topic}'). Evita articoli generici che parlano di molteplici sport.
    REQUISITO FONDAMENTALE: Prima di usare il tool di ricerca, DEVI spiegare il tuo ragionamento e giustificare perché lo stai usando.
    Scrivi il tuo ragionamento testuale iniziando con 'Thought: [la tua giustificazione]'.
    Dopodiché, usa il tool 'cerca_sul_web'.
    """
    
    print("   [ReAct] Avvio l'agente. Attendi l'elaborazione (potrebbe richiedere un minuto in locale)...")
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
            
            if getattr(msg, 'type', '') == 'ai' and hasattr(msg, 'tool_calls') and msg.tool_calls:
                
                if msg.content:
                    thought_str = f"Thought: {msg.content.strip()}"
                    trace.append(thought_str)
                    print(f"   [ReAct] {thought_str}")
                
                for tool_call in msg.tool_calls:
                    action_str = f"Action: Chiamo il tool '{tool_call['name']}' con query {tool_call['args']}"
                    trace.append(action_str)
                    print(f"   [ReAct] {action_str}")
            
            elif getattr(msg, 'type', '') == 'tool':
                obs_str = f"Observation: Risultato del tool (ID: {msg.tool_call_id}). Letti {len(msg.content)} caratteri."
                trace.append(obs_str)
                print(f"   [ReAct] {obs_str}\n")
                
                raw.append({"url": "Search Output", "content": msg.content})

        if not raw:
            print("   [Fallback] Nessun tool chiamato o estrazione fallita.")
            search_res = ddg_search.invoke(topic)
            raw.append({"url": f"Search: {topic}", "content": search_res})
        
    return {
        "raw_resources": raw, 
        "reasoning_trace": trace, 
        "status": "research_done"
    }

def quality_fact_checker(state: AgentState) -> dict:
    print("-> Esecuzione Quality & Fact Checker...")
    raw = state.get("raw_resources", [])
    topic = state["current_topic"]
    verified = []
    
    G = load_kg()
    past_topics = [n for n, attributes in G.nodes(data=True) if attributes.get("type") in ["Topic", "Post"]]
    
    for res in raw:
        content = res.get('content', '')[:1500]
        prompt = (
             f"Sei un Fact Checker. Argomento richiesto: '{topic}'.\n\n"
             f"KNOWLEDGE GRAPH (GIA' TRATTATI): {past_topics}\n"
             f"Frammento di contenuto: '{content}'.\n\n"
             f"RISPONDI 'NO' SE IL CONTENUTO PARLA DI ATLETI O EVENTI GIÀ PRESENTI NEL KNOWLEDGE GRAPH (es. se Djokovic è nel KG e l'articolo parla di lui, scarta tutto).\n"
             f"Questo contenuto porta informazioni NUOVE, pertitenti e non ripetitive? Rispondi esattamente 'SI' o 'NO'."
        )
        try:
            eval_res = llm.invoke([HumanMessage(content=prompt)]).content.upper()
            if "SI" in eval_res or "SÌ" in eval_res:
                verified.append(res)
                print("   [LLM Evaluator] Fonte approvata.")
            else:
                print(f"   [LLM Evaluator] Fonte scartata (Ripetizione o irrilevante). Risposta: {eval_res}")
        except Exception:
            print("   [LLM Evaluator] Errore API, accetto la fonte per default.")
            verified.append(res)
            
    return {"verified_resources": verified, "status": "fact_checking_done"}

def drafter(state: AgentState) -> dict:

    print("-> Esecuzione Drafter (Generazione testo in corso)...")
    
    feedback = state.get("human_feedback", "")
    previous_draft = state.get("draft", "")  
    sources = state.get("verified_resources", [])
    topic = state['current_topic']
    
    sources_text = "\n".join([s.get("content", "")[:1000] for s in sources])
    
    prompt = (
        f"Scrivi un breve e coinvolgente articolo di blog sulle ultime notizie riguardo '{topic}'.\n"
        f"Usa questi frammenti di ricerca come contesto primario:\n{sources_text}\n\n"
        f"REGOLE FONDAMENTALI (PENA IL RIFIUTO DELLO SCRITTO):\n"
        f"1. L'articolo deve parlare ESCLUSIVAMENTE di UN SOLO SPORT e di UN SINGOLO EVENTO. Se i frammenti mescolano più sport (es. motori e basket), SCEGLINE SOLO UNO (quello inerente a '{topic}') e IGNORA tutto il resto. Vieta categoricamente di mischiare discipline diverse.\n"
        f"2. Inizia la tua risposta ESATTAMENTE con 'TITOLO: <il tuo titolo accattivante>' sulla primissima riga.\n"
        f"3. L'articolo deve essere in italiano.\n"
        f"4. Scrivi in modo diretto e giornalistico (VIETATO usare scuse, premesse o frasi come 'ecco l'articolo')."
    )
    
    if feedback and previous_draft:
        prompt += f"\n\nEcco la tua BOZZA PRECEDENTE:\n---\n{previous_draft}\n---\n\nCRITICO: L'utente ha rifiutato la bozza precedente con questo feedback: '{feedback}'. DEVI riscrivere pesantemente la bozza. Ricorda SEMPRE di focalizzarti su UN SOLO SPORT."
        
    try:
        response_text = llm.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        print(f"\n[ERRORE FATALE] Impossibile generare la bozza a causa di un errore API: {e}")
        raise SystemExit(1)
        
    new_title = topic
    draft = response_text
    
    lines = response_text.split('\n')
    for i, line in enumerate(lines):
        if line.strip().upper().startswith('TITOLO:'):
            new_title = line.split(':', 1)[1].strip().replace('**', '').replace('"', '').replace("'", "")
            draft = "\n".join(lines[:i] + lines[i+1:]).strip()
            break
        elif line.strip().upper().startswith('**TITOLO:**'):
            new_title = line.split(':', 1)[1].strip().replace('**', '').replace('"', '').replace("'", "")
            draft = "\n".join(lines[:i] + lines[i+1:]).strip()
            break

    return {"draft": draft, "current_topic": new_title, "status": "draft_ready", "revision_count": state.get("revision_count", 0) + 1}

def human_review(state: AgentState) -> dict:
    import tkinter as tk
    from tkinter import messagebox
    from tkinter.scrolledtext import ScrolledText

    print("-> Esecuzione Human Review (Apertura finestra grafica in corso)...")
    
    draft_content = state.get('draft', '')
    topic = state.get('current_topic', 'Senza Titolo')
    
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
            messagebox.showwarning("Attenzione", "Devi inserire un feedback (es. 'Scrivilo più corto' o 'Usa un tono più formale') per poter riscrivere l'articolo.")
            return
        result["status"] = "rewrite"
        result["human_feedback"] = feedback
        root.destroy()

    root = tk.Tk()
    root.title(f"Revisione Bozza: {topic}")
    root.geometry("850x650")
    
    lbl = tk.Label(root, text="Ecco la bozza generata. Scegli se approvare, chiedere modifiche o scartare l'argomento:", font=("Helvetica", 12, "bold"))
    lbl.pack(pady=10)
    
    txt_area = ScrolledText(root, wrap=tk.WORD, width=90, height=22, font=("Helvetica", 11))
    txt_area.insert(tk.INSERT, draft_content)
    txt_area.config(state=tk.DISABLED)
    txt_area.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)
    
    frame_feedback = tk.Frame(root)
    frame_feedback.pack(fill=tk.X, padx=20, pady=5)
    tk.Label(frame_feedback, text="Opzionale - Spiega cosa modificare:", font=("Helvetica", 10)).pack(side=tk.LEFT)
    feedback_entry = tk.Entry(frame_feedback, font=("Helvetica", 11))
    feedback_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
    
    frame_btn = tk.Frame(root)
    frame_btn.pack(pady=15)
    
    tk.Button(frame_btn, text="✅ Approva e Pubblica", bg="lightgreen", font=("Helvetica", 11, "bold"), padx=10, pady=5, command=on_approve).pack(side=tk.LEFT, padx=15)
    tk.Button(frame_btn, text="🔁 Riscrivi (Invia Feedback)", bg="gold", font=("Helvetica", 11, "bold"), padx=10, pady=5, command=on_rewrite).pack(side=tk.LEFT, padx=15)
    tk.Button(frame_btn, text="❌ Scarta Notizia", bg="lightcoral", font=("Helvetica", 11, "bold"), padx=10, pady=5, command=on_reject).pack(side=tk.LEFT, padx=15)
    
    def on_closing():
        if messagebox.askokcancel("Arresta", "La chiusura della finestra comporterà lo scarto della notizia. Procedere?"):
            on_reject()
            
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    
    root.mainloop()
    
    if not result:
        rejected = state.get("rejected_topics", []) + [state["current_topic"]]
        result = {"status": "rejected_topic", "human_feedback": "Notizia scartata (finestra chiusa).", "rejected_topics": rejected}
        
    return result

def kg_updater(state: AgentState) -> dict:
    print("-> Esecuzione KG Updater & Web Publisher...")
    topic = state["current_topic"]
    sources = state.get("verified_resources", [])
    
    G = load_kg()
    
    if not G.has_node(topic):
        G.add_node(topic, type="Topic")
    
    post_count = sum(1 for n, attr in G.nodes(data=True) if attr.get("type") == "Post")
    post_id = f"Post_{post_count + 1}"
    
    G.add_node(post_id, type="Post", snippet=state["draft"][:100]) 
    G.add_edge(post_id, topic, relation="COVERS_TOPIC")
    
    for idx, res in enumerate(sources):
        url = res.get("url", f"fonte_sconosciuta_{idx}")
        if not G.has_node(url):
            G.add_node(url, type="Source")
        G.add_edge(post_id, url, relation="USES_SOURCE")

    print(f"   [KG] Rete aggiornata! Nodi totali: {G.number_of_nodes()}, Archi totali: {G.number_of_edges()}")

    save_kg(G) 
    
    html_path = "index.html"
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        date_str = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
        formatted_draft = state["draft"].replace('\n', '<br>')
        
        new_post_html = f'''
        <article class="post-card">
            <h2 class="post-title">{topic}</h2>
            <div class="post-meta">Pubblicato il {date_str}</div>
            <div class="post-content">
                {formatted_draft}
            </div>
            <div class="post-tags">
                <span class="tag">AI Generated</span>
                <span class="tag">Sport News</span>
            </div>
        </article>
        <!-- NEW_POSTS_HERE -->'''
        
        html_content = html_content.replace("<!-- NEW_POSTS_HERE -->", new_post_html)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print("   [Web Publisher] Post aggiunto al file index.html!")
    
    return {"status": "kg_updated", "kg_context": G}


# ==========================================
# 3. DEFINIZIONE DEGLI ARCHI E COMPILAZIONE GRAFO
# ==========================================

def check_resources_quality(state: AgentState) -> Literal["drafter", "resource_researcher"]:
    if len(state.get("verified_resources", [])) > 0:
        return "drafter"
    return "resource_researcher"

def check_approval(state: AgentState) -> Literal["kg_updater", "drafter", "topic_planner"]:
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
workflow.add_conditional_edges("quality_fact_checker", check_resources_quality, {"drafter": "drafter", "resource_researcher": "resource_researcher"})
workflow.add_edge("drafter", "human_review")
workflow.add_conditional_edges("human_review", check_approval, {"kg_updater": "kg_updater", "drafter": "drafter", "topic_planner": "topic_planner"})
workflow.add_edge("kg_updater", END)

app = workflow.compile()

print("=== GENERAZIONE DEL GRAFO IN CORSO ===")
try:
    img = app.get_graph().draw_mermaid_png()
    display(Image(img))
    with open("graph.png", "wb") as f:
        f.write(img)
except Exception as e:
    print("Errore nella generazione dell'immagine:", e)
print("======================================\n")

# ==========================================
# 4. ESECUZIONE (TEST DELLA PIPE LANGGRAPH)
# ==========================================
print("=== INIZIO PROCESSO LANGGRAPH ===")

persisted_kg = load_kg()

initial_state = {
    "user_request": "Ultimi aggiornamenti, notizie e cronaca riguardanti il mondo dello sport (calcio, tennis, motori, basket, ecc.). Voglio solo notizie, aggiornamenti e cronache per uno sport. Le notizie devono essere recenti dell'anno 2025 e 2026",
    "kg_context": persisted_kg,
    "planned_topics": [],
    "current_topic": "",
    "raw_resources": [],
    "verified_resources": [],
    "draft": "",
    "human_feedback": "",
    "status": "start",
    "revision_count": 0,
    "rejected_topics": []
}

for output in app.stream(initial_state):
    pass
print("\n=== PROCESSO CONCLUSO ===")