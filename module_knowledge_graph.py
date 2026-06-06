import json
import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

PLANNING_FILE = "planning_queue.json"
KG_FILE = "knowledge_graph.graphml"

graph_db = Neo4jGraph(
    url=os.environ.get("NEO4J_URI"),
    username=os.environ.get("NEO4J_USERNAME"),
    password=os.environ.get("NEO4J_PASSWORD"),
    database=os.environ.get("NEO4J_DATABASE"),
)


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
        print("   [Planner] Coda di pianificazione aggiornata e salvata su file.")
    except Exception as e:
        print(f"   [Planner] Errore nel salvataggio della coda: {e}")
