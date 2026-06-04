import pandas as pd
import json
import os
import glob


def crea_database_unico(
    cartella_excel="prezzicibo", output_json="database_unico_ismea.json"
):
    print(f"Avvio fusione dei file Excel dalla cartella: '{cartella_excel}'...")

    file_excel = glob.glob(os.path.join(cartella_excel, "*.xlsx"))
    file_excel.extend(glob.glob(os.path.join(cartella_excel, "*.xls")))

    if not file_excel:
        print("Attenzione: Nessun file Excel trovato nella cartella specificata!")
        return

    database = {
        "ingredienti_standard": {},
        "ingredienti_bio": {},
        "vini": {"rosso": {}, "bianco": {}, "bollicine": {}},
    }

    # Proviamo ad usare una lista più ampia o un controllo dinamico delle colonne
    mesi_disponibili = [
        "2026-05",
        "2026-04",
        "2026-03",
        "2026-02",
        "2026-01",
        "2025-12",
        "2025-11",
        "2025-10",
        "2025-09",
        "2025-08",
        "2025-07",
        "2025-06",
    ]

    for file in file_excel:
        nome_file = os.path.basename(file).lower()
        print(f"\n--- Elaborazione file: {nome_file} ---")

        try:
            df = pd.read_excel(file)

            # STAMPA DEBUG 1: Vediamo come si chiamano DAVVERO le colonne
            print(f"Colonne trovate in Excel: {df.columns.tolist()}")

        except Exception as e:
            print(f"Errore nella lettura del file {nome_file}: {e}")
            continue

        prodotti_letti = 0
        prodotti_salvati = 0

        for index, row in df.iterrows():
            # STAMPA DEBUG 2: Controlliamo se la colonna Prodotti esiste
            if "Prodotti" not in df.columns and "Prodotto" not in df.columns:
                print("ERRORE CRITICO: Non trovo la colonna 'Prodotti' o 'Prodotto'.")
                break

            nome_colonna_prod = "Prodotti" if "Prodotti" in df.columns else "Prodotto"

            if pd.isna(row.get(nome_colonna_prod)):
                continue

            prodotti_letti += 1
            prodotto_grezzo = str(row[nome_colonna_prod]).lower().strip()
            nome_pulito = prodotto_grezzo.split("-")[0].strip()

            # Trova l'ultimo prezzo valido (Adesso trasformiamo i nomi colonne in stringhe per sicurezza)
            prezzo_ingrosso = 0.0
            colonne_str = [str(c) for c in df.columns]

            for mese in mesi_disponibili:
                if mese in colonne_str:
                    # Troviamo il vero nome della colonna in df (che potrebbe essere un oggetto DateTime)
                    vera_colonna = df.columns[colonne_str.index(mese)]
                    valore_cella = row[vera_colonna]

                    if (
                        pd.notna(valore_cella)
                        and isinstance(valore_cella, (int, float))
                        and valore_cella > 0.0
                    ):
                        prezzo_ingrosso = float(valore_cella)
                        break

            if prezzo_ingrosso > 0.0:
                prodotti_salvati += 1
                prezzo_dettaglio = round(prezzo_ingrosso * 1.8, 2)

                # Gestione colonna Unità di Misura (spesso si chiama diversamente)
                colonna_um = "Valuta/UM" if "Valuta/UM" in df.columns else df.columns[1]
                um_letta = row.get(colonna_um)

                um_pulita = str(um_letta).strip() if pd.notna(um_letta) else "N/A"
                um_pulita = um_pulita.replace("/peso vivo", "")

                dati = {"prezzo": prezzo_dettaglio, "um": um_pulita}

                if "bio" in prodotto_grezzo or "biologico" in prodotto_grezzo:
                    database["ingredienti_bio"][
                        nome_pulito.replace(" bio", "").strip()
                    ] = dati
                elif "vino" in nome_file or "vini" in nome_file:
                    if any(
                        x in prodotto_grezzo
                        for x in ["rosso", "chianti", "barolo", "merlot", "cabernet"]
                    ):
                        database["vini"]["rosso"][nome_pulito] = dati
                    elif any(
                        x in prodotto_grezzo
                        for x in ["bianco", "chardonnay", "pinot", "sauvignon"]
                    ):
                        database["vini"]["bianco"][nome_pulito] = dati
                    else:
                        database["vini"]["bollicine"][nome_pulito] = dati
                else:
                    database["ingredienti_standard"][nome_pulito] = dati

        print(
            f"Letti {prodotti_letti} righe prodotto -> Salvati {prodotti_salvati} con prezzo valido."
        )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=4, ensure_ascii=False)

    print(f"\nDatabase creato con successo in {output_json}!")


if __name__ == "__main__":
    crea_database_unico(cartella_excel="prezzicibo")
