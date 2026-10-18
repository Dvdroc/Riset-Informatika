# 05 - Rencana Evaluation Metric

## 1. Jenis tugas penelitian

**Klasifikasi teks (sentimen 3 kelas: positif, netral, negatif)**, sehingga kelompok metrik yang dipakai adalah metrik klasifikasi. Kelompok lain ditampilkan hanya sebagai pembanding agar jelas mengapa tidak dipakai.

| Jenis tugas | Metrik | Dipakai? | Alasan |
|---|---|---|---|
| **Klasifikasi** | Accuracy, precision, recall, F1-score, confusion matrix | **Ya** | Label berupa kelas diskret |
| Regresi | MAE, RMSE, R² | Tidak | Keluaran bukan nilai kontinu |
| Clustering | Silhouette score, Davies-Bouldin | Tidak | Data berlabel dan tugas terawasi |
| NLP / generasi teks | BLEU, ROUGE, perplexity | Tidak | Model tidak menghasilkan teks |

## 2. Metrik yang digunakan

Untuk setiap kelas *k*: TP, FP, FN, TN dihitung dari confusion matrix 3 x 3.

| Metrik | Rumus | Peran |
|---|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Gambaran umum; **tidak** dijadikan metrik utama |
| Precision | TP / (TP + FP) | Ketepatan prediksi per kelas |
| Recall | TP / (TP + FN) | Kemampuan menemukan seluruh data per kelas |
| F1-score | 2 x P x R / (P + R) | Keseimbangan precision dan recall |
| **Macro F1** | rata-rata F1 semua kelas | **Metrik utama** |
| Weighted F1 | F1 dibobot jumlah data per kelas | Pembanding |
| Confusion matrix | tabel label sebenarnya vs prediksi | Pola kesalahan, terutama kelas netral |

### Mengapa macro F1 sebagai metrik utama

Hasil eksplorasi SmSA menunjukkan distribusi kelas yang timpang dan **berbeda antara data latih dan data uji**:

| Pembagian | Positif | Negatif | Netral |
|---|---|---|---|
| Latih | 58,3% | 31,2% | 10,4% |
| Uji | 41,6% | 40,8% | 17,6% |

Accuracy dapat menyesatkan pada kondisi ini (model yang cenderung menebak kelas mayoritas tetap tampak akurat). Macro F1 memberi bobot sama pada setiap kelas sehingga kelas netral tidak tertutupi. Precision, recall, dan F1 per kelas dilaporkan terpisah.

## 3. Metrik pelengkap

| Metrik | Cara pengukuran | Menjawab |
|---|---|---|
| Waktu pelatihan | Waktu nyata (wall-clock) hingga model terbaik | Rumusan masalah 3 |
| Waktu inferensi | Waktu memprediksi 1.000 teks uji | Rumusan masalah 3 |
| Ukuran model dan memori GPU | MB dan GB | Rumusan masalah 3 |
| Uji McNemar | Perbandingan berpasangan dua model pada data uji yang sama | Hipotesis H1 |
| Simpangan baku antarseed | 3 seed pada IndoBERT | Stabilitas hasil |
| Penurunan macro F1 pada uji ketahanan | Selisih terhadap data uji asli (E5 dan E6) | Hipotesis H2 |

## 4. Protokol evaluasi

1. Pembagian data resmi SmSA: 11.000 latih, 1.260 validasi, 500 uji.
2. Hyperparameter dan checkpoint dipilih dengan **macro F1 pada data validasi**. Data uji dipakai sekali untuk pelaporan akhir.
3. TF-IDF dipelajari hanya dari data latih (mencegah kebocoran data).
4. IndoBERT dijalankan dengan 3 random seed; laporkan rata-rata dan simpangan baku.
5. Perbedaan dua model diuji dengan uji McNemar (taraf 5%) dan dilaporkan bersama besar selisih.

## 5. Skenario eksperimen yang dievaluasi

| Kode | Model | Prapemrosesan |
|---|---|---|
| E1 | TF-IDF + SVM | Skenario A (minimal) |
| E2 | TF-IDF + SVM | Skenario B (normalisasi) |
| E3 | IndoBERT | Skenario A |
| E4 | IndoBERT | Skenario B |
| E5 | E1-E4 | Evaluasi pada subset uji yang informal |
| E6 | E1-E4 | Evaluasi pada uji dengan gangguan sintetis |

## 6. Templat tabel hasil

| Kode | Accuracy | Macro F1 | F1 positif | F1 netral | F1 negatif | Waktu latih | Waktu inferensi |
|---|---|---|---|---|---|---|---|
| E1 | | | | | | | |
| E2 | | | | | | | |
| E3 (rata-rata ± SD) | | | | | | | |
| E4 (rata-rata ± SD) | | | | | | | |

## 7. Contoh kode (scikit-learn)

```python
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             f1_score, classification_report, confusion_matrix)

labels = ["positive", "neutral", "negative"]

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, digits=4)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "report": report, "confusion_matrix": cm}

# Uji McNemar (statsmodels)
# from statsmodels.stats.contingency_tables import mcnemar
# b = ((pred_a == y) & (pred_b != y)).sum()
# c = ((pred_a != y) & (pred_b == y)).sum()
# print(mcnemar([[0, b], [c, 0]], exact=False, correction=True))
```

## 8. Referensi

[16] M. Sokolova and G. Lapalme, "A systematic analysis of performance measures for classification tasks," *Inf. Process. Manage.*, vol. 45, pp. 427-437, 2009, doi: 10.1016/j.ipm.2009.03.002.
