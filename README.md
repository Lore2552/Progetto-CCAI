# Progetto-CCAI

## Introduzione
Progetto per generare Post (Ricette) di un Blog di cucina il cui draft finale viene mostrato in un'interfaccia grafica che permette all'utente di approvare, modificare o rifiutare e rigenerare il post.

## Prerequisiti
- Python 3.8 o superiore
- Dipendenze: esegui `pip install -r requirements.txt`
- Cartella database locale: `chroma_db/`

## Configurazione (.env)
Creare un file `.env` nella root del progetto che contenta le seguenti variabili:

- `GROQ_API_KEY` — API key per `ChatGroq` 
- `NEO4J_URI` — URI di Neo4j (es. `neo4j+s://...`)
- `NEO4J_USERNAME` — Username Neo4j
- `NEO4J_PASSWORD` — Password Neo4j
- `NEO4J_DATABASE` — Nome del database Neo4j (di solito `neo4j`)
- `COHERE_API_KEY` — API key per Cohere 
- `LANGCHAIN_TRACING_V2` — abilita tracing LangChain 
- `LANGCHAIN_ENDPOINT` — endpoint LangChain
- `LANGCHAIN_API_KEY` — API key per LangChain
- `LANGCHAIN_PROJECT` — nome progetto LangChain
- `HF_TOKEN` — token Hugging Face 

Note: i moduli caricano le variabili con `load_dotenv()` quindi il file `.env` deve essere nella root dove esegui `python main.py`.

## Avvio
1. Installa dipendenze:

```bash
pip install -r requirements.txt
```

2. Avvia l'app:

```bash
python main.py
```

Output e comportamento:
- Il flusso operativo è tracciato via stampe in console 
- L'interfaccia grafica mostra il post generato e permette all'utente di:
  - approvare 
  - modificare
  - rifiutare e rigenerare 

