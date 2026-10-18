# 02 - Dataset Rencana Riset Kecil

## Dataset yang dipakai: SmSA (IndoNLU)

| Item | Keterangan |
|---|---|
| Nama | SmSA, bagian dari benchmark IndoNLU (nama folder: `smsa_doc-sentiment-prosa`) |
| Sumber data | Komentar dan ulasan berbahasa Indonesia dari berbagai platform daring, dikumpulkan oleh Purwarianti dan Crisdayanti (2019) |
| Tugas | Klasifikasi sentimen 3 kelas: `positive`, `neutral`, `negative` |
| Ukuran | Latih 11.000, validasi 1.260, uji 500 (total 12.760 teks) |
| Format | TSV tanpa header: `teks <TAB> label` |
| Tautan data | https://github.com/IndoNLP/indonlu/tree/master/dataset/smsa_doc-sentiment-prosa |
| Tautan benchmark | https://github.com/IndoNLP/indonlu |
| Makalah | B. Wilie dkk., "IndoNLU", AACL-IJCNLP 2020, doi: 10.18653/v1/2020.aacl-main.85 |

Berkas mentah yang langsung bisa diunduh:

- https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv
- https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/valid_preprocess.tsv
- https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/test_preprocess.tsv

> Periksa ketentuan lisensi dataset pada repositori IndoNLU sebelum mengunggah ulang seluruh data. Karena itu folder ini hanya memuat **sampel 100 baris** dan skrip pengunduh.

## Isi folder

| Berkas | Fungsi |
|---|---|
| `sample_smsa_100.csv` | Sampel 100 teks dari data latih (58 positif, 31 negatif, 11 netral; seed 42) |
| `download_smsa.py` | Mengunduh dan memuat SmSA sebagai DataFrame |
| `eksplorasi_smsa.py` | Skrip eksplorasi data (ukuran, kelas, panjang, duplikat, keinformalan) |
| `distribusi_kelas_smsa.png` | Grafik distribusi kelas per pembagian data |
| `hasil_eksplorasi.md` | Ringkasan hasil eksplorasi |

## Alasan dataset ini cocok

1. Dataset publik berbahasa Indonesia dan sudah dipakai sebagai benchmark, sehingga hasil dapat dibandingkan dengan literatur.
2. Sudah terbagi menjadi data latih, validasi, dan uji, sehingga protokol evaluasi jelas.
3. Ukurannya sedang (puluhan ribu teks), cukup untuk TF-IDF + SVM dan fine-tuning IndoBERT pada GPU gratis Google Colab.
4. Memiliki tiga kelas dengan kelas netral yang minoritas, sehingga macro F1 bermakna sebagai metrik utama.

## Cara menjalankan

```bash
pip install pandas
python download_smsa.py       # unduh dan simpan smsa_train/valid/test.csv
python eksplorasi_smsa.py     # cetak statistik dataset
```
