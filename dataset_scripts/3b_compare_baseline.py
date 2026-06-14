import os
import json
import time
from tqdm import tqdm
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
# ==========================================
# 1. INIZIALIZZAZIONE API CLIENT (HUGGING FACE)
# ==========================================

hf_token = os.getenv("HF_TOKEN")

client = InferenceClient(token=hf_token)

PATH_DATASET_TEST = os.path.join("dataset_scripts", "local_test_finale.jsonl")
RELAZIONI_VALIDE = {"USA_INGREDIENTE", "USA_TECNICA", "TIPO_DI_PIATTO"}
N_CAMPIONI = 50

BASELINES = {
    "Baseline-Qwen-3B": {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "output_file": "baseline_qwen3b_risultati.json",
    },
}


# ==========================================
# 2. METRICHE LOGICHE (IDENTICHE AL FINE-TUNE)
# ==========================================
def parse_triple(testo):
    triple = set()
    for riga in testo.strip().split("\n"):
        parti = [p.strip() for p in riga.split("|")]
        if len(parti) == 3 and parti[1] in RELAZIONI_VALIDE:
            triple.add(tuple(parti))
    return triple


def estrai_entita(triple):
    entita = set()
    for sogg, rel, ogg in triple:
        entita.add((sogg, "Ricetta"))
        if rel == "USA_INGREDIENTE":
            entita.add((ogg, "Ingrediente"))
        elif rel == "USA_TECNICA":
            entita.add((ogg, "Tecnica"))
        elif rel == "TIPO_DI_PIATTO":
            entita.add((ogg, "Tipo_Piatto"))
        else:
            entita.add((ogg, "Altro"))
    return entita


def calcola_precision_recall_f1(pred_set, vero_set):
    if not pred_set and not vero_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not vero_set:
        return 0.0, 0.0, 0.0
    veri_positivi = len(pred_set & vero_set)
    precision = veri_positivi / len(pred_set)
    recall = veri_positivi / len(vero_set)
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1_score


def calcola_metriche_globali(predette_str, attese_str):
    pred_triple = parse_triple(predette_str)
    vere_triple = parse_triple(attese_str)

    pred_entita = estrai_entita(pred_triple)
    vere_entita = estrai_entita(vere_triple)

    st_p, st_r, st_f1 = calcola_precision_recall_f1(pred_triple, vere_triple)
    ner_p, ner_r, ner_f1 = calcola_precision_recall_f1(pred_entita, vere_entita)

    return {
        "strict_triplet_precision": st_p,
        "strict_triplet_recall": st_r,
        "strict_triplet_f1": st_f1,
        "ner_precision": ner_p,
        "ner_recall": ner_r,
        "ner_f1": ner_f1,
    }


# ==========================================
# 3. CARICAMENTO DATASET LOCAL_TEST_FINALE
# ==========================================
print(f"📥 Caricamento del test set da: {PATH_DATASET_TEST}")
if not os.path.exists(PATH_DATASET_TEST):
    raise FileNotFoundError(f"❌ Impossibile trovare il file {PATH_DATASET_TEST}.")

dataset_test = []
with open(PATH_DATASET_TEST, "r", encoding="utf-8") as f:
    for line in f:
        dataset_test.append(json.loads(line))

campione_test = dataset_test[: min(N_CAMPIONI, len(dataset_test))]
print(f"✅ File caricato. Trovati {len(campione_test)} campioni per lo Zero-Shot.")


# ==========================================
# 4. LOOP DI VALUTAZIONE
# ==========================================
for nome_baseline, info in BASELINES.items():
    print(
        f"\n🚀 Esecuzione {nome_baseline} su Hugging Face via Hub Client ({info['model_name']})..."
    )
    risultati_esempi = []

    for idx, esempio in enumerate(tqdm(campione_test, desc=f"In corso")):
        try:
            # Sfruttiamo il metodo chat_completion nativo di huggingface_hub
            chat_completion = client.chat_completion(
                model=info["model_name"],
                messages=[
                    {"role": "system", "content": esempio["instruction"]},
                    {"role": "user", "content": esempio["input"]},
                ],
                temperature=0.1,
                max_tokens=512,
            )

            risposta_predetta = chat_completion.choices[0].message.content.strip()
            metriche_singole = calcola_metriche_globali(
                risposta_predetta, esempio["output"]
            )
            risultati_esempi.append(metriche_singole)

            # Sonnellino per evitare di saturare i rate-limit free
            time.sleep(0.4)

        except Exception as e:
            print(f"\n⚠️ Salto riga {idx} per errore API: {e}")
            continue

    if not risultati_esempi:
        print(
            f"❌ Nessun dato estratto per {nome_baseline}. Il modello potrebbe essere temporaneamente offline o il token non è valido."
        )
        continue

    # Media aritmetica
    medie = {
        k: sum(r[k] for r in risultati_esempi) / len(risultati_esempi)
        for k in risultati_esempi[0]
    }

    report_json = {
        "baseline_name": nome_baseline,
        "model_id": info["model_name"],
        "metrics": {
            "ner_precision": round(medie["ner_precision"], 3),
            "ner_recall": round(medie["ner_recall"], 3),
            "ner_f1": round(medie["ner_f1"], 3),
            "strict_triplet_precision": round(medie["strict_triplet_precision"], 3),
            "strict_triplet_recall": round(medie["strict_triplet_recall"], 3),
            "strict_triplet_f1": round(medie["strict_triplet_f1"], 3),
            "perplexity": None,
        },
    }

    with open(info["output_file"], "w", encoding="utf-8") as out_f:
        json.dump(report_json, out_f, ensure_ascii=False, indent=2)

    print(f"💾 Risultati estratti salvati in '{info['output_file']}'")

print("\n🎯 Pipeline completata.")
