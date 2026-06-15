import json
import re
import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from typing import TypedDict, List, Dict, Any, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from module_queue_manager import load_planning_queue, save_planning_queue
from module_tools import (
    llm,
    graph_db,
    kg_rag_tool,
    valuta_documento_locale,
    cerca_e_leggi_sul_web,
    cerca_nel_database_ismea,
    aggiorna_knowledge_graph_db,
)

load_dotenv()


# 1. Definizione dello Stato strutturato del Grafo
class AgentState(TypedDict):
    user_request: str
    kg_context: Any
    kg_rag_result: Dict[str, Any]
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


# 2. Funzioni dei Nodi dell'Agente
def topic_planner(state: AgentState) -> dict:
    print("-> Esecuzione Topic Planner con Coda Persistente (Primo, Secondo, Dolce)...")
    cypher_query = "MATCH (t:Topic) RETURN t.name AS name"
    risultati = graph_db.query(cypher_query)
    past_topics = [res["name"] for res in risultati if res.get("name")]
    rejected_topics = state.get("rejected_topics", [])
    all_avoids = past_topics + rejected_topics

    print(f"   [KG] Argomenti passati da evitare: {past_topics}")
    if rejected_topics:
        print(f"   [Feedback] Argomenti scartati in questa sessione: {rejected_topics}")

    queue = load_planning_queue()
    print(f"   [Planner] Coda attuale letta da file: {queue}")
    current_topic = ""
    current_topic_type = ""
    last_status = state.get("status", "")

    if queue and last_status != "rejected_topic":
        next_item = queue.pop(0)
        current_topic = next_item["ricetta"]
        current_topic_type = next_item["tipo"]
        print(
            f"   [Planner] Consumata ricetta dalla coda: '{current_topic}' ({current_topic_type})"
        )
    elif last_status == "rejected_topic" and rejected_topics:
        print(
            f"   [Planner] Rilevato topic scartato. Rigenerazione della categoria specifica per mantenere l'equilibrio..."
        )

    if current_topic:
        all_avoids.append(current_topic)

    tipi_presenti = [item["tipo"] for item in queue]
    tipi_mancanti = []

    for tipo in ["Primo", "Secondo", "Dolce"]:
        if tipo not in tipi_presenti:
            tipi_mancanti.append(tipo)

    for item in queue:
        all_avoids.append(item["ricetta"])

    justification = state.get(
        "editorial_justification", "Nessuna giustificazione precedente."
    )

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
            f"  ]\n"
            f"CRITICO: Se il nome della ricetta è molto simile ad un altro presente nel knowledge graph (ad esempio, nel knowledge graph c'è scritto la ricetta classica dei tortellini en brodo, e la ricetta da valutare è tortellini in brodo, o tortellini al brodo) non devi includerla ASSOLUTAMENTE."
            f"}}"
        )
        try:
            response = llm.invoke([HumanMessage(content=prompt)]).content.strip()
            if response.startswith("```"):
                response = re.sub(
                    r"^```[a-zA-Z]*\n|```$", "", response, flags=re.M
                ).strip()
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            json_clean = json_match.group(0) if json_match else response
            data_pianificata = json.loads(json_clean)

            justification = data_pianificata.get(
                "giustificazione_editoriale",
                "Pianificazione strategica per la diversità del menu.",
            )
            nuovi_piatti = data_pianificata.get("sequenza_piano", [])

            if isinstance(nuovi_piatti, list):
                if not current_topic and nuovi_piatti:
                    primo_estratto = nuovi_piatti.pop(0)
                    current_topic = primo_estratto["ricetta"]
                    current_topic_type = primo_estratto["tipo"]
                    print(
                        f"   [Planner] Primo avvio: '{current_topic}' impostato come topic corrente."
                    )

                for piatto in nuovi_piatti:
                    ricetta_nome = piatto.get("ricetta", "").strip()
                    tipo_nome = piatto.get("tipo", "").strip().capitalize()

                    if ricetta_nome and tipo_nome:
                        ricetta_lower = ricetta_nome.lower()
                        is_duplicate = False
                        for avoid_item in all_avoids:
                            avoid_lower = avoid_item.lower()
                            if (
                                ricetta_lower in avoid_lower
                                or avoid_lower in ricetta_lower
                            ):
                                is_duplicate = True
                                break

                        if not is_duplicate:
                            queue.append({"tipo": tipo_nome, "ricetta": ricetta_nome})
                            all_avoids.append(ricetta_nome)
                            if tipo_nome in tipi_mancanti:
                                tipi_mancanti.remove(tipo_nome)
                        else:
                            print(
                                f"   [Planner BUG-SHIELD] Scartato duplicato semantico: '{ricetta_nome}'"
                            )

        except Exception as e:
            print(
                f"   [Planner Errore LLM/Parsing JSON] {e}. Imposto fallback d'emergenza."
            )

            if not current_topic:
                current_topic = "Spaghetti alla Carbonara"
            if not queue:
                queue = [
                    {"tipo": "Secondo", "ricetta": "Cotoletta alla Milanese"},
                    {"tipo": "Dolce", "ricetta": "Panna Cotta"},
                ]
            justification = "Fallback applicato per mantenere la continuità editoriale."

    save_planning_queue(queue)

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


def resource_researcher(state: AgentState) -> dict:
    topic = state["current_topic"]
    trace = state.get("reasoning_trace", [])
    print(f"-> Esecuzione Resource Researcher (con paradigma ReAct) per: {topic}")

    tools = [kg_rag_tool, valuta_documento_locale, cerca_e_leggi_sul_web]
    react_agent = create_agent(llm, tools)

    prompt = f"""Devi raccogliere dati completi su come si prepara questa ricetta: '{topic}'.
    Hai a disposizione 3 tool per la ricerca:
    1. 'kg_rag_tool': Cerca la ricetta nel database locale ibrido. Restituisce le regole fisse del Knowledge Graph (ingredienti e tecniche obbligatorie) e il frammento di testo completo della ricetta.
    2. 'valuta_documento_locale': Usa questo tool fornendo 'target_topic', 'document_ids', 'recipe_url', 'ingredienti' e 'tecniche' restituiti da 'kg_rag_tool', per far valutare oggettivamente i documenti appena recuperati dal DB locale ed escludere automaticamente quelli irrilevanti.
    3. 'cerca_e_leggi_sul_web': Cerca ed estrae integralmente i passaggi da pagine internet. Utilizzalo se la ricetta locale è assente, parziale o per colmare i dettagli mancanti.

    Flusso Operativo Richiesto:
    1. REASONING: Inizia ogni azione con 'Thought: ...'
    2. Usa SEMPRE come prima azione il tool 'kg_rag_tool'.
    3. Usa SUBITO il tool 'valuta_documento_locale' passandogli target_topic, document_ids, recipe_url, ingredienti e tecniche restituiti dal tool 'kg_rag_tool'. Se la risposta dice che copre interamente la variante, FERMATI ed esponi tu una sintesi che unisce ingredienti e passaggi di preparazione del piatto in un testo descrittivo.
    4. Se l'esito della validazione del database locale è 'PARZIALE' perché manca un ingrediente o un dettaglio specifico rispetto al topic, la ricerca sul web deve essere MIRATA a prendere le informazioni mancanti o la ricetta COMPLETA per integrarle.
    5. Se il tool locale non trova nulla o sei insoddisfatto, usa 'cerca_e_leggi_sul_web' per ottenere l'intera ricetta. Questo tool andrà a restituire diverse fonti web, devi prendere la più inerente al nostro topic (anche basandoti sul nome della ricetta e gli ingredienti). 
    6. Se cerca_e_leggi_sul_web restituisce siti web non italiani (ad esempio spagnoli,inglesi,ecc) devi ripetere la ricerca magari cambiando qualche parola chiave per cercare di ottenere risultati più pertinenti alla cucina italiana. Puoi fare fino a 3 tentativi di ricerca web modificando leggermente la query.
    7. FALLBACK: Se anche dopo le stampe e le ricerche sul server web non si trova nulla di perfettamente completo, procedi prendendo assieme le informazioni e frammenti 'parziali' trovati fino ad ora, assemblando il draft con ciò che hai.
    8. Alla fine di tutti i controlli, unisci ciò che hai ottenuto di buono sia in locale che sul web. Come TUA risposta FINALE restituisci un unico riassunto descrittivo.

    REGOLA FONDAMENTALE PER LA RISPOSTA FINALE:
    Se durante l'esecuzione hai usato il tool 'cerca_e_leggi_sul_web', inserisci alla fine del tuo riassunto una riga esatta con l'URL scelto:
    URL_SELEZIONATO: <url_della_fonte_web>
    Se NON hai usato il tool web perché il database locale era già sufficiente, scrivi semplicemente:
    URL_SELEZIONATO: DB_Locale
    """
    print("   [ReAct] Avvio l'agente. Attendi l'elaborazione...")

    try:
        response = react_agent.invoke({"messages": [HumanMessage(content=prompt)]})
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
        kg_rag_payload = {}
        kg_rag_payload_filtrato = {}

        for msg in response.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.get("name") == "cerca_e_leggi_sul_web":
                        ha_usato_il_web = True

            if getattr(msg, "type", "") == "tool":
                try:
                    parsed_tool_content = json.loads(msg.content)

                    if isinstance(parsed_tool_content, dict):
                        if "document_ids" in parsed_tool_content:
                            kg_rag_payload = {
                                "metadata_kg": {
                                    "topic": parsed_tool_content.get("topic", topic),
                                    "url": parsed_tool_content.get(
                                        "recipe_url", "Non disponibile"
                                    ),
                                    "ingredienti": parsed_tool_content.get(
                                        "ingredienti", ""
                                    ),
                                    "tecniche": parsed_tool_content.get("tecniche", ""),
                                },
                                "document_ids": parsed_tool_content.get(
                                    "document_ids", []
                                ),
                            }

                        if "kg_rag_payload_filtrato" in parsed_tool_content:
                            kg_rag_payload_filtrato = parsed_tool_content.get(
                                "kg_rag_payload_filtrato", {}
                            )

                except Exception:
                    pass

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

        kg_rag_effettivo = kg_rag_payload_filtrato or kg_rag_payload

        contesto_rag_effettivo = kg_rag_effettivo.get(
            "risposta_finale",
            "Nessun contesto DB locale o non attivato in modo sufficiente.",
        )

    except Exception as e:
        print(f"   [Errore ReAct Agent] {e}")
        raw = []
        contesto_rag_effettivo = None
        kg_rag_effettivo = {}

    return {
        "raw_resources": raw,
        "kg_context": contesto_rag_effettivo,
        "kg_rag_result": kg_rag_effettivo,
        "reasoning_trace": trace,
        "status": "research_done",
    }


def quality_fact_checker(state: AgentState) -> dict:
    print("-> Esecuzione Quality & Fact Checker...")
    raw = state.get("raw_resources", [])
    topic = state["current_topic"]
    valid_links = []

    cypher_query = "MATCH (t:Topic) RETURN t.name AS name"
    risultati = graph_db.query(cypher_query)
    past_topics = [res["name"] for res in risultati if res.get("name")]

    for res in raw:
        content = res.get("content", "")[:3500]
        prompt = (
            f"Sei un Fact Checker. Ricetta sintetica presentata: '{topic}'.\n\n"
            f"KNOWLEDGE GRAPH (RICETTE GIA' SCRITTE): {past_topics}\n"
            f"Testo della ricetta da validare:\n'{content}'.\n\n"
            f"La ricetta è già inclusa nel knowledge graph? Se si, rispondi 'NO'.\n"
            f"Il procedimento espone la preparazione di questo piatto ed è valido? Rispondi esattamente 'SI' o 'NO'."
            f"CRITICO: Se il nome della ricetta è molto simile ad un altro presente nel knoledge graph (ad esempio, nel knowledge graph c'è scritto la ricetta classica dei tortellini en brodo, e la ricetta da valutare è tortellini in brodo, o tortellini al brodo) devi rispondere 'SI' perché è una variante molto comune e accettabile."
        )
        try:
            eval_res = llm.invoke([HumanMessage(content=prompt)]).content.upper()
            print(f"EVAL_RES: {eval_res}")
            if "SI" in eval_res or "SÌ" in eval_res:
                valid_links.append(res)
                print("   [LLM Evaluator] Output di ricerca approvato.")
            else:
                print(f"   [LLM Evaluator] Scartato o non valido. Resp: {eval_res}")
        except Exception:
            valid_links.append(res)

    return {"verified_resources": valid_links, "status": "fact_checking_done"}


def drafter(state: AgentState) -> dict:
    print("-> Esecuzione Drafter (Generazione Post in corso)...")

    cypher_query = """
    MATCH (s:Entity)-[r]->(o:Entity)
    RETURN s.name + ' ' + type(r) + ' ' + o.name AS claim 
    LIMIT 10
    """
    try:
        risultati_grafo = graph_db.query(cypher_query)
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
    sources = state.get("verified_resources") or state.get("raw_resources") or []
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
        f"⚠️ OBBLIGATORIO: Ogni ingrediente in lista DEVE avere la sua quantità e unità di misura esatta e realistica (es. 'Vitello (girello) 800 g', 'Tonno sott'olio 160 g', 'Acciughe 4 filetti', 'Aglio 1 spicchio'). È severamente vietato scrivere solo il nome dell'ingrediente senza il suo peso o dosaggio.\n"
        f"8. L'articolo deve contenere necessariamente la preparazione dettagliata della ricetta trattata.\n"
        f"9. CITAZIONI IN LINEA OBBLIGATORIE (REQUISITO CRITICO): Il tuo testo finale deve dimostrare chiaramente l'uso combinato delle fonti. Applica queste citazioni nel testo:\n"
        f"   - Quando inserisci un ingrediente obbligatorio o una tecnica del database locale, usa: [Fonte: Knowledge Graph].\n"
        f"   - Quando descrivi passaggi pratici, tempi o dettagli presi dal web, usa tassativamente l'URL: [Fonte: {url_fonte}].\n"
        f"   Esempio: 'Aggiungete il guanciale [Fonte: Knowledge Graph] e fatelo rosolare a fuoco lento per circa 10 minuti [Fonte: {url_fonte}].'\n"
        f"10. DICITURA FINALE OBBLIGATORIA: Alla fine dell'articolo, lascia una riga vuota e inserisci la fonte globale. "
        f"11. DIVIETO SUI PREZZI: È SEVERAMENTE VIETATO inserire costi, prezzi o valute (es. €) accanto agli ingredienti nella bozza. Scrivi solo il nome dell'ingrediente e la quantità (es. 'Uova (3 medie)'). Ai prezzi ci penserà il reparto contabilità successivamente."
        f"12. LUNGHEZZA E STILE EDITORIALE: Mantieni l'articolo entro circa 650-900 parole. "
    )

    if url_fonte == "DB_Locale" or url_fonte == "Nessun URL trovato":
        prompt += "Scrivi ESATTAMENTE: 'Fonte: Ricetta del database locale'."
    else:
        prompt += f"Scrivi ESATTAMENTE: 'Fonte web: {url_fonte}'."

    if feedback and previous_draft:
        prompt += (
            f"\n\nEcco la tua BOZZA PRECEDENTE:\n---\n{previous_draft}\n---\n\n"
            f"CRITICO: L'utente ha rifiutato la bozza precedente con questo feedback: '{feedback}'. "
            f"DEVI riscrivere la bozza seguendo il feedback. "
            f"Ricorda la Regola 11: durante la riscrittura, ELIMINA qualsiasi prezzo in Euro eventualmente presente. Lascia la lista degli ingredienti pulita."
        )
    try:
        drafter_llm = llm.bind(max_tokens=1800)
        response_text = drafter_llm.invoke([("human", prompt)]).content
    except Exception as e:
        print(
            f"\n[ERRORE FATALE] Impossibile generare la bozza a causa di un errore API: {e}"
        )
        raise SystemExit(1)

    new_title = topic
    draft = response_text

    lines = response_text.split("\n")
    for i, line in enumerate(lines):
        clean_line = line.strip().upper()
        if clean_line.startswith("TITOLO:") or clean_line.startswith("**TITOLO:**"):
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
        "verified_resources": sources,
        "revision_count": state.get("revision_count", 0) + 1,
    }


def recipe_enricher(state: AgentState) -> dict:
    print("-> Esecuzione Recipe Enricher...")
    draft = state.get("draft", "")
    if not draft:
        return {"status": "enrichment_failed"}

    tools = [cerca_nel_database_ismea]
    enricher_agent = create_agent(llm, tools)
    prompt = f"""Sei l'Editor Centrale di un blog di cucina. Il tuo compito è arricchire la bozza della ricetta usando ESCLUSIVAMENTE i dati reali estratti dal tool 'cerca_nel_database_ismea'.

    BOZZA ATTUALE:
    ---
    {draft}
    ---

    FLUSSO OPERATIVO TASSATIVO:
    1. Estrai gli ingredienti principali dalla bozza.
    2. CHIAMA IL TOOL 'cerca_nel_database_ismea' passandogli la lista.
    3. Riscrivi e restituisci l'INTERA bozza originale (mantenendo intatte la preparazione e le fonti [Fonte: ...]), applicando queste rigide modifiche:

     1. PREZZI (OBBLIGATORIO): 
    Modifica la lista degli ingredienti presente nella bozza. Per ogni ingrediente, se il tool ti ha restituito un prezzo, scrivilo accanto tra parentesi tonde. 
    Esempio: "Burro 100 g (Circa 10.00 €/Kg) [Fonte: Knowledge Graph]"

     2. SEZIONE BIO / INTOLLERANZE (CONDIZIONALE):
    SE E SOLO SE il tool restituisce delle Alternative BIO che hanno senso (ad esempio, non ha senso sostituire un ingrediente con un altro se non è effettivamente una variante BIO o ad esempio consigliare il Sale BIO non ha senso), crea in fondo al testo la sezione "💡 I Consigli dello Chef" e inseriscile. Se la ricetta contiene burro/latte/formaggio, aggiungi di tua iniziativa un consiglio sulle varianti senza lattosio.
    ATTENZIONE: Se il tool risponde che NON ci sono alternative BIO, NON creare questa sezione, NON nominarle e NON scusarti. Ignora l'argomento.

     3. SEZIONE VINO (CONDIZIONALE):
    SE E SOLO SE il tool restituisce un suggerimento per il vino, crea la sezione finale "🍷 L'Abbinamento Perfetto" e ricopia il suggerimento. 
    ATTENZIONE: Se il tool risponde "Nessun abbinamento specifico trovato", è SEVERAMENTE VIETATO creare questa sezione. NON inventare abbinamenti tuoi e NON scrivere "Purtroppo non ho trovato vini". Semplicemente, omettila.

    Restituisci direttamente l'articolo finale formattato.
    """

    print("   [Enricher] Chiamo l'agente per arricchire i dati...")

    try:
        response = enricher_agent.invoke({"messages": [("human", prompt)]})
        final_answer = draft
        for msg in response.get("messages", []):
            if getattr(msg, "type", "") == "ai" and msg.content:
                final_answer = msg.content
    except Exception as e:
        print(f"   [Errore Enricher] {e}")
        final_answer = draft

    return {"draft": final_answer, "status": "enrichment_done"}


def human_review(state: AgentState) -> dict:
    print("-> Esecuzione Human Review (Apertura Finestra Grafica)...")
    draft_content = state.get("draft", "")
    topic = state.get("current_topic", "Senza Titolo")
    result = {}

    def on_approve():
        result["status"] = "approved"
        result["human_feedback"] = ""
        root.destroy()

    def on_reject():
        queue = load_planning_queue()
        if queue:
            queue.pop()
        queue.insert(
            0,
            {
                "tipo": state.get("current_topic_type", "Piatto"),
                "ricetta": state["current_topic"],
            },
        )
        save_planning_queue(queue)

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
    root.title(f"Revisione: {topic}")
    root.geometry("850x650")

    justification_text = state.get(
        "editorial_justification", "Nessuna giustificazione fornita."
    )
    topic_type = state.get("current_topic_type", "N/A")

    lbl = tk.Label(
        root,
        text=f"Argomento attuale: {topic} ({topic_type})\nStrategia: {justification_text}",
        font=("Helvetica", 11, "italic"),
        fg="blue",
        justify=tk.LEFT,
        wraplength=800,
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


# 3. Regole di Flusso Condizionale e Compilazione Workflow
def check_resources_quality(
    state: AgentState,
) -> Literal["drafter", "resource_researcher"]:
    if len(state.get("verified_resources", [])) > 0:
        return "drafter"
    return "resource_researcher"


def check_approval(state: AgentState) -> str:
    status = state.get("status")
    if status == "approved":
        return "kg_updater"
    elif status == "rewrite":
        return "drafter"
    else:
        return "topic_planner"


workflow = StateGraph(AgentState)

workflow.add_node("topic_planner", topic_planner)
workflow.add_node("resource_researcher", resource_researcher)
workflow.add_node("quality_fact_checker", quality_fact_checker)
workflow.add_node("drafter", drafter)
workflow.add_node("enricher", recipe_enricher)
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

workflow.add_edge("drafter", "enricher")
workflow.add_edge("enricher", "human_review")

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
