import os
import json
import chromadb
import re
import random
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Carichiamo le variabili dal file .env (GROQ_API_KEY)
load_dotenv()

# 1. Inizializziamo il modello cloud da 70B per estrarre gli esempi perfetti (Ground Truth)
llm_generatore = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

# 2. Connessione al database ChromaDB locale
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_ricette = chroma_client.get_or_create_collection(
    name="ricette_giallozafferano"
)

OUTPUT_FILE = "dataset_triple_culinarie.jsonl"

print("📥 Recupero di tutti i frammenti grezzi da ChromaDB...")
try:
    # Recuperiamo TUTTI i documenti, metadati e ID per poterli riorganizzare
    risultati = collection_ricette.get()
    documenti = risultati.get("documents", [])
    metadati = risultati.get("metadatas", []) or []
    ids = risultati.get("ids", [])
    print(f"📚 Trovati {len(documenti)} frammenti complessivi nel database locale.")
except Exception as e:
    print(f"❌ Errore nel recupero dati da ChromaDB: {e}")
    documenti = []

if not documenti:
    print("⚠️ Il database ChromaDB è vuoto o non accessibile. Impossibile procedere.")
    exit()

# =========================================================================
# SOLUZIONE ALLA STRUTTURA DEI CHUNK: RAGGRUPPIAMO I CHUNK PER RICETTA
# =========================================================================
print("🧩 Riorganizzazione e ricomposizione dei chunk in ricette intere...")
ricette_mappate = {}

for doc_id, testo_chunk, meta in zip(ids, documenti, metadati):
    titolo_ricetta = meta.get("titolo", "Ricetta Sconosciuta").strip()
    if titolo_ricetta == "Ricetta Sconosciuta":
        continue

    if titolo_ricetta not in ricette_mappate:
        ricette_mappate[titolo_ricetta] = []

    # Salviamo l'ID e il testo del chunk per poterli ordinare cronologicamente
    ricette_mappate[titolo_ricetta].append((doc_id, testo_chunk))

# Ordiniamo i chunk di ogni singola ricetta in base all'ID (garantisce: Ingredienti -> Preparazione)
for titolo in ricette_mappate:
    ricette_mappate[titolo].sort(key=lambda x: x[0])

elenco_ricette_uniche = list(ricette_mappate.keys())
totale_ricette_trovate = len(elenco_ricette_uniche)
print(f"🔍 Identificate {totale_ricette_trovate} ricette uniche e distinte.")

# =========================================================================
# SOLUZIONE ALL'ESTRAZIONE CASUALE E LIMITATA (Target: 350 Ricette)
# =========================================================================
TARGET_ESTRAZIONE = min(
    totale_ricette_trovate, 350
)  # Estrae esattamente 350 ricette (centro del target 300/400)
print(
    f"🎲 Selezione CASUALE di {TARGET_ESTRAZIONE} ricette per rompere l'ordinamento alfabetico..."
)

# Impostiamo un seed opzionale se vuoi riproducibilità, altrimenti lascialo puramente casuale
random.seed(42)
ricette_selezionate_a_caso = random.sample(elenco_ricette_uniche, TARGET_ESTRAZIONE)

print(f"🚀 Avvio generazione del dataset sintetico su file: {OUTPUT_FILE}")
count = 0

# Apriamo il file .jsonl (JSON Lines) in modalità scrittura
with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    for idx, titolo_ricetta in enumerate(ricette_selezionate_a_caso):
        print(
            f" ⏳ [{idx+1}/{TARGET_ESTRAZIONE}] Unione chunk ed estrazione triple per: {titolo_ricetta}..."
        )

        # Uniamo tutti i chunk ordinati della ricetta in un unico testo coeso per dare pieno contesto all'LLM
        testo_completo_ricetta = "\n\n".join(
            [chunk[1] for chunk in ricette_mappate[titolo_ricetta]]
        )

        prompt_estrazione = f"""Sei un motore deterministico di estrazione dati per grafi di conoscenza culinari.
        Analizza questo testo completo (ingredienti e passaggi combinati) legato alla ricetta '{titolo_ricetta}':
        ---
        {testo_completo_ricetta[:5000]}
        ---
        
        COMPITO TASSATIVO: Estrai TUTTE le relazioni fondamentali (ingredienti principali e tecniche) presenti ESCLUSIVAMENTE nel testo fornito.
        USA RIGIDAMENTE il formato: Soggetto | RELAZIONE | Oggetto
        
        REGOLE COMPORTAMENTALI CRITICHE:
        1. Il Soggetto deve essere SEMPRE l'entità principale della ricetta (es. "{titolo_ricetta}").
        2. Per la RELAZIONE, puoi utilizzare SOLO uno di questi tre termini pre-approvati (VIETATO inventarne altri):
           - USA_INGREDIENTE
           - USA_TECNICA
           - TIPO_DI_PIATTO
        3. Rispondi SOLO con le triple, una per riga, separate dal carattere '|'. Non aggiungere numeri, introduzioni, commenti o markdown.
        
        Esempio di output pulito richiesto:
        {titolo_ricetta} | TIPO_DI_PIATTO | Primo
        {titolo_ricetta} | USA_INGREDIENTE | Farina
        {titolo_ricetta} | USA_TECNICA | Impastare
        """

        try:
            # Chiamata a Groq usando il modello 70B per estrarre la Ground Truth perfetta
            risposta = llm_generatore.invoke(
                [HumanMessage(content=prompt_estrazione)]
            ).content.strip()

            # Isoliamo le linee pulite per evitare scorie di testo libero
            triple_pulite = [
                line.strip() for line in risposta.split("\n") if "|" in line
            ]
            output_triple = "\n".join(triple_pulite)

            if triple_pulite:
                # Struttura JSON standard per il Fine-Tuning Supervisionato (SFT)
                json_line = {
                    "instruction": "Analizza il testo della ricetta ed estrai le relazioni nel formato rigido: Soggetto | RELAZIONE | Oggetto. Usa solo i termini approvati: USA_INGREDIENTE, USA_TECNICA, TIPO_DI_PIATTO.",
                    "input": testo_completo_ricetta[
                        :2500
                    ].strip(),  # Input limitato per mantenere bilanciata la lunghezza dei contesti nel training LoRA
                    "output": output_triple,
                }

                # Scriviamo il record (una riga per ogni ricetta intera selezionata)
                f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                count += 1
        except Exception as e:
            print(
                f"   ⚠️ Errore durante l'elaborazione della ricetta {titolo_ricetta}: {e}"
            )
            continue

print(
    f"\n🎉 Operazione conclusa! Generati con successo {count} esempi di ricette intere e casuali in '{OUTPUT_FILE}'."
)
print("Il dataset è matematicamente perfetto per essere caricato su Kaggle.")
