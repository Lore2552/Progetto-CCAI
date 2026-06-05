import json

def integra_database_ismea(file_json="database_unico_ismea.json"):
    print(f"Lettura del database esistente: {file_json}...")
    
    try:
        with open(file_json, "r", encoding="utf-8") as f:
            db = json.load(f)
    except FileNotFoundError:
        print("Errore: File JSON non trovato. Assicurati che il nome e il percorso siano corretti.")
        return

    # --- LISTINO DEGLI INGREDIENTI (Prezzi medi al dettaglio stimati in €/Kg o €/Litro) ---
    integrazione_standard = {
        "burro": {"prezzo": 11.50, "um": "€/Kg"},
        "parmigiano reggiano": {"prezzo": 22.00, "um": "€/Kg"},
        "pecorino": {"prezzo": 19.50, "um": "€/Kg"},
        "grana padano": {"prezzo": 18.00, "um": "€/Kg"},
        "panna fresca": {"prezzo": 6.50, "um": "€/Litro"},
        "prosciutto crudo": {"prezzo": 28.00, "um": "€/Kg"},
        "prosciutto cotto": {"prezzo": 18.00, "um": "€/Kg"},
        "guanciale": {"prezzo": 16.00, "um": "€/Kg"},
        "pancetta": {"prezzo": 14.00, "um": "€/Kg"},
        "mortadella": {"prezzo": 13.50, "um": "€/Kg"},
        "salame": {"prezzo": 20.00, "um": "€/Kg"},
        "sale fino": {"prezzo": 0.80, "um": "€/Kg"},
        "sale grosso": {"prezzo": 0.80, "um": "€/Kg"},
        "pepe nero": {"prezzo": 35.00, "um": "€/Kg"},
        "noce moscata": {"prezzo": 55.00, "um": "€/Kg"},
        "salvia": {"prezzo": 15.00, "um": "€/Kg"},
        "basilico": {"prezzo": 18.00, "um": "€/Kg"},
        "rosmarino": {"prezzo": 12.00, "um": "€/Kg"},
        "prezzemolo": {"prezzo": 10.00, "um": "€/Kg"},
        "pinoli": {"prezzo": 75.00, "um": "€/Kg"},
        "trofie": {"prezzo": 3.80, "um": "€/Kg"},
        "pasta di semola": {"prezzo": 1.60, "um": "€/Kg"},
        "farina 00": {"prezzo": 1.30, "um": "€/Kg"},
        "lievito di birra fresco": {"prezzo": 10.00, "um": "€/Kg"},
        "zucchero": {"prezzo": 1.40, "um": "€/Kg"}
    }

    integrazione_bio = {
        "burro": {"prezzo": 15.00, "um": "€/Kg"},
        "parmigiano reggiano": {"prezzo": 28.00, "um": "€/Kg"},
        "farina 00": {"prezzo": 2.20, "um": "€/Kg"},
        "zucchero": {"prezzo": 2.80, "um": "€/Kg"},
        "pasta di semola": {"prezzo": 2.50, "um": "€/Kg"},
        "pinoli": {"prezzo": 90.00, "um": "€/Kg"}
    }

    integrazione_vini_bianchi = {
        "vino bianco da cucina": {"prezzo": 2.50, "um": "€/Litro"},
        "sauvignon blanc": {"prezzo": 8.50, "um": "€/Bottiglia"},
        "pinot grigio": {"prezzo": 7.00, "um": "€/Bottiglia"}
    }
    
    integrazione_vini_rossi = {
        "vino rosso da cucina": {"prezzo": 2.50, "um": "€/Litro"},
        "chianti classico": {"prezzo": 9.50, "um": "€/Bottiglia"},
        "lambrusco": {"prezzo": 5.50, "um": "€/Bottiglia"}
    }

    # Funzione per evitare duplicati semantici
    def ingrediente_gia_presente(nome_nuovo, chiavi_esistenti):
        # Puliamo la stringa da apostrofi e parole di collegamento per un confronto più "crudo"
        pulito_nuovo = nome_nuovo.lower().replace("d'", "di ").replace("'", " ")
        
        for chiave_es in chiavi_esistenti:
            pulito_es = chiave_es.lower().replace("d'", "di ").replace("'", " ")
            # Se uno è contenuto nell'altro (es. "zucchero" dentro "zucchero raffinato") evitiamo duplicati
            if pulito_nuovo in pulito_es or pulito_es in pulito_nuovo:
                return True
        return False

    # --- INSERIMENTO CONTROLLATO ---
    print("\nVerifica e inserimento Ingredienti Standard...")
    aggiunti_std = 0
    chiavi_std_attuali = list(db["ingredienti_standard"].keys())
    
    for nome, dati in integrazione_standard.items():
        if not ingrediente_gia_presente(nome, chiavi_std_attuali):
            db["ingredienti_standard"][nome] = dati
            chiavi_std_attuali.append(nome) # Lo aggiungiamo per i controlli successivi
            aggiunti_std += 1
        else:
            print(f" - Saltato: '{nome}' (Esiste già una variante simile nel DB ISMEA)")

    print("\nVerifica e inserimento Ingredienti BIO...")
    aggiunti_bio = 0
    chiavi_bio_attuali = list(db["ingredienti_bio"].keys())
    
    for nome, dati in integrazione_bio.items():
        if not ingrediente_gia_presente(nome, chiavi_bio_attuali):
            db["ingredienti_bio"][nome] = dati
            chiavi_bio_attuali.append(nome)
            aggiunti_bio += 1
        else:
            print(f" - Saltato: '{nome}' BIO (Esiste già una variante simile nel DB ISMEA)")

    # Integrazione vini (questi li aggiungiamo in blocco perché ISMEA non li ha tracciati nel tuo DB)
    if "vini" not in db:
        db["vini"] = {"bianco": {}, "rosso": {}, "bollicine": {}}
        
    for nome, dati in integrazione_vini_bianchi.items():
        if nome not in db["vini"]["bianco"]:
            db["vini"]["bianco"][nome] = dati
            
    for nome, dati in integrazione_vini_rossi.items():
        if nome not in db["vini"]["rosso"]:
            db["vini"]["rosso"][nome] = dati

    # Salvataggio del nuovo file
    with open(file_json, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4, ensure_ascii=False)

    print(f"\nAggiornamento completato con successo!")
    print(f"- Effettivamente aggiunti: {aggiunti_std} nuovi ingredienti standard.")
    print(f"- Effettivamente aggiunti: {aggiunti_bio} nuovi ingredienti BIO.")

if __name__ == "__main__":
    integra_database_ismea()