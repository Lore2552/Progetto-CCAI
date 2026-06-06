from module_graph import app

if __name__ == "__main__":
    print("=== GENERAZIONE DEL GRAFO IN CORSO ===")
    try:
        img = app.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(img)
        print("Mappa dei nodi esportata con successo in 'graph.png'")
    except Exception as e:
        print("Avviso: Impossibile generare l'immagine del grafico:", e)
    print("======================================\n")

    print("=== INIZIO PROCESSO LANGGRAPH ===")
    initial_state = {
        "user_request": (
            "Scrivi un blog post con la preparazione completa e dettagliata di una "
            "famosa ricetta della cucina tradizionale italiana. Voglio imparare ricette "
            "autentiche e classiche. Includi solo UNA ricetta per post."
        ),
        "kg_context": None,
        "kg_rag_result": {},
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

    print("\n=== PROCESSO CONCLUSO CORRETTAMENTE ===")
