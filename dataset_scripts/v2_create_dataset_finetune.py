import os
import random
from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd

# ==========================================
# CONFIGURAZIONE CONFIG
# ==========================================

FILE_CULINARIE = "dataset_triple_culinarie.jsonl"
FILE_OOD = "dataset_test_ood_200.jsonl"

OUTPUT_TRAIN = "local_train_misto.jsonl"
OUTPUT_TEST = "local_test_finale.jsonl"

SEED = 42

def carica_dataset_safe(file_path):
    """Carica un file JSONL con fallback su Pandas in caso di errori"""
    print(f"📦 Caricamento di {file_path}...")
    try:
        return load_dataset("json", data_files=file_path, split="train")
    except Exception as e:
        print(f"⚠️ Errore con HF datasets ({e}). Provo il fallback con Pandas...")
        df = pd.read_json(file_path, lines=True)
        return Dataset.from_pandas(df)

def main():
    # 1. Caricamento dataset
    ds_culinarie = carica_dataset_safe(FILE_CULINARIE)
    ds_ood = carica_dataset_safe(FILE_OOD)
    
    print(f"Dimensione originale Culinarie: {len(ds_culinarie)}")
    print(f"Dimensione originale OOD: {len(ds_ood)}")
    
    # ==========================================
    # 2. CAMPIONAMENTO E SPLIT
    # ==========================================
    print("\n✂️ Estrazione delle quote richieste...")
    

    ds_culinarie_shuffled = ds_culinarie.shuffle(seed=SEED)
    treno_culinarie = ds_culinarie_shuffled.select(range(min(1000, len(ds_culinarie))))
    

    ds_ood_shuffled = ds_ood.shuffle(seed=SEED)
    treno_ood = ds_ood_shuffled.select(range(150))
    test_finale_ood = ds_ood_shuffled.select(range(150, min(200, len(ds_ood))))
    
    # ==========================================
    # 3. FUSIONE E SHUFFLE DEL TRAIN
    # ==========================================
    print("🔀 Fusione e shuffle del nuovo Train Set...")
    train_misto = concatenate_datasets([treno_culinarie, treno_ood])
    train_misto = train_misto.shuffle(seed=SEED)
    
    # ==========================================
    # 4. SALVATAGGIO DEI NUOVI FILE
    # ==========================================
    print("\n💾 Salvataggio dei file locali...")
    
    if "__index_level_0__" in train_misto.column_names:
        train_misto = train_misto.remove_columns(["__index_level_0__"])
    if "__index_level_0__" in test_finale_ood.column_names:
        test_finale_ood = test_finale_ood.remove_columns(["__index_level_0__"])
        
    train_misto.to_json(OUTPUT_TRAIN, orient="records", lines=True)
    test_finale_ood.to_json(OUTPUT_TEST, orient="records", lines=True)
    
    print("-" * 40)
    print(f"✅ Fatto! Creati i seguenti file:")
    print(f"📝 {OUTPUT_TRAIN} -> {len(train_misto)} elementi (1000 Culinari + 150 OOD shufflati)")
    print(f"📝 {OUTPUT_TEST} -> {len(test_finale_ood)} elementi (50 OOD puri)")
    print("-" * 40)

if __name__ == "__main__":
    main()