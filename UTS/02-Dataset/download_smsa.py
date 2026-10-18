"""Unduh dataset SmSA (IndoNLU) dan muat sebagai DataFrame.

Sumber : https://github.com/IndoNLP/indonlu/tree/master/dataset/smsa_doc-sentiment-prosa
Format : TSV tanpa header, dua kolom: teks <TAB> label (positive / neutral / negative)
Jalankan: python download_smsa.py
"""
import pandas as pd

BASE = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa"
FILES = {"train": "train_preprocess.tsv", "valid": "valid_preprocess.tsv", "test": "test_preprocess.tsv"}


def load_smsa():
    data = {}
    for split, name in FILES.items():
        # quoting=3 (QUOTE_NONE) agar tanda kutip di dalam teks tidak merusak pembacaan
        data[split] = pd.read_csv(f"{BASE}/{name}", sep="\t", header=None, names=["text", "label"], quoting=3)
    return data


if __name__ == "__main__":
    d = load_smsa()
    for split, df in d.items():
        print(split, len(df), df.label.value_counts().to_dict())
    for split, df in d.items():
        df.to_csv(f"smsa_{split}.csv", index=False)
    print("Tersimpan: smsa_train.csv, smsa_valid.csv, smsa_test.csv")
