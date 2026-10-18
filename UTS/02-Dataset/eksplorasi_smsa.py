"""Eksplorasi data SmSA: ukuran, distribusi kelas, panjang teks, duplikat, dan tingkat keinformalan."""
import pandas as pd
from download_smsa import load_smsa

INFORMAL = {"enggak", "nggak", "gue", "lo", "aja", "udah", "kayak", "gimana", "gak", "ga", "tp", "yg", "dgn"}

d = load_smsa()
for split, df in d.items():
    words = df.text.str.split().str.len()
    informal = df.text.apply(lambda t: any(w in INFORMAL for w in t.split()))
    print(f"[{split}] n={len(df)}")
    print("  kelas      :", df.label.value_counts().to_dict(), df.label.value_counts(normalize=True).round(3).to_dict())
    print(f"  panjang    : rata-rata {words.mean():.1f} kata, median {int(words.median())}, p95 {int(words.quantile(.95))}, maks {words.max()}")
    print(f"  informal   : {informal.mean():.1%} teks memuat kata informal")
print("duplikat latih          :", d["train"].text.duplicated().sum())
print("irisan latih-validasi   :", len(set(d["train"].text) & set(d["valid"].text)))
print("irisan latih-uji        :", len(set(d["train"].text) & set(d["test"].text)))
