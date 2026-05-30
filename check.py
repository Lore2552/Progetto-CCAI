import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

load_dotenv()

graph_db = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
)

queries = {
    "Recipe": "MATCH (n:Recipe) RETURN count(n) AS count",
    "Ingredient": "MATCH (n:Ingredient) RETURN count(n) AS count",
    "Technique": "MATCH (n:Technique) RETURN count(n) AS count",
    "Url": "MATCH (n:Url) RETURN count(n) AS count",
}

print("\n===== RECIPE KG STATUS =====\n")

for label, query in queries.items():
    try:
        result = graph_db.query(query)
        count = result[0]["count"]
        print(f"{label:<12} {count}")
    except Exception as e:
        print(f"{label:<12} ERRORE: {e}")
