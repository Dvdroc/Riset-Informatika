# 02 - Dataset Rencana Riset Kecil

## Dataset yang Dipakai: SmSA (IndoNLU)

Penelitian ini menggunakan **SmSA (*Sentence-level Sentiment Analysis*)**, yaitu dataset analisis sentimen berbahasa Indonesia yang menjadi bagian dari benchmark IndoNLU. Dataset ini digunakan untuk membandingkan performa metode **TF-IDF + SVM** dan **IndoBERT** dalam mengklasifikasikan sentimen teks ke dalam tiga kelas, yaitu positif, netral, dan negatif.

## Informasi Dataset

| Item | Keterangan |
|---|---|
| Nama dataset | SmSA (*Sentence-level Sentiment Analysis*) |
| Benchmark | IndoNLU |
| Nama folder sumber | `smsa_doc-sentiment-prosa` |
| Sumber data | Komentar dan ulasan berbahasa Indonesia dari berbagai platform daring |
| Referensi dataset | Purwarianti dan Crisdayanti (2019) |
| Tugas | Klasifikasi sentimen tiga kelas |
| Label | `positive`, `neutral`, `negative` |
| Data latih | 11.000 teks |
| Data validasi | 1.260 teks |
| Data uji | 500 teks |
| Total | 12.760 teks |
| Format sumber | TSV tanpa header, dengan dua kolom: teks dan label |
| Bahasa | Bahasa Indonesia |

## Sumber dan Referensi

**Repositori dataset IndoNLU:**

https://github.com/IndoNLP/indonlu

**Folder dataset SmSA:**

https://github.com/IndoNLP/indonlu/tree/master/dataset/smsa_doc-sentiment-prosa

**Berkas dataset yang dapat diunduh langsung:**

- Data latih: https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv
- Data validasi: https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/valid_preprocess.tsv
- Data uji: https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/test_preprocess.tsv

**Publikasi benchmark IndoNLU:**

Wilie, B., dkk. (2020). *IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding*. Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing (AACL-IJCNLP 2020), halaman 843–857.

DOI: https://doi.org/10.18653/v1/2020.aacl-main.85

**Referensi penelitian SmSA:**

Purwarianti, A., dan Crisdayanti, I. A. P. (2019). *Improving Bi-LSTM Performance for Indonesian Sentiment Analysis Using Paragraph Vector*. 2019 International Conference of Advanced Informatics: Concepts, Theory and Applications (ICAICTA).

DOI: https://doi.org/10.1109/ICAICTA.2019.8904199

## Lisensi dan Atribusi Dataset

Metadata dataset SmSA pada implementasi NusaCrowd mencantumkan lisensi **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.

Referensi lisensi:

- Metadata dataset: https://github.com/IndoNLP/nusa-crowd/blob/master/nusacrowd/nusa_datasets/smsa/smsa.py
- Teks lisensi CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/
- Repositori sumber: https://github.com/IndoNLP/indonlu

Informasi lisensi tersebut dicantumkan sebagai rujukan dokumentasi dataset. Cakupan lisensi dan izin untuk mendistribusikan ulang seluruh teks perlu dipastikan berdasarkan ketentuan yang berlaku dari pengelola dataset atau pemegang hak yang berwenang, terutama karena data berasal dari berbagai platform daring.

Penelitian ini menggunakan dataset untuk keperluan eksperimen klasifikasi sentimen. Sumber dan publikasi terkait dicantumkan sebagai bentuk atribusi akademik.

**Catatan distribusi:** dataset lengkap digunakan secara lokal untuk eksperimen penelitian dan tidak disertakan dalam repositori ini pada tahap dokumentasi dan verifikasi lisensi. Repositori menyediakan skrip pengunduhan dari sumber dataset, dokumentasi, serta sampel terbatas untuk memperlihatkan format data.

## Isi Folder

| Berkas | Fungsi |
|---|---|
| `README.md` | Dokumentasi dataset, sumber, lisensi, struktur folder, dan cara penggunaan |
| `sample_smsa_100.csv` | Sampel 100 teks dari data latih untuk demonstrasi format dataset |
| `download_smsa.py` | Mengunduh dan memuat dataset SmSA sebagai DataFrame, kemudian menyimpannya ke CSV |
| `eksplorasi_smsa.py` | Melakukan eksplorasi data, meliputi ukuran dataset, distribusi kelas, panjang teks, duplikasi, dan keinformalan bahasa |
| `distribusi_kelas_smsa.png` | Visualisasi distribusi kelas sentimen pada data latih, validasi, dan uji |
| `hasil_eksplorasi.md` | Ringkasan hasil eksplorasi dataset |

Dataset lengkap yang telah diunduh disimpan secara lokal dan tidak termasuk dalam daftar berkas repositori publik di atas.

## Alasan Pemilihan Dataset

1. **Berbahasa Indonesia.** Dataset sesuai dengan fokus penelitian klasifikasi sentimen teks berbahasa Indonesia.
2. **Memiliki referensi penelitian.** SmSA merupakan bagian dari benchmark IndoNLU sehingga dapat digunakan sebagai dasar eksperimen dan perbandingan dengan penelitian terkait.
3. **Memiliki pembagian data yang jelas.** Dataset menyediakan data latih, validasi, dan uji untuk mendukung evaluasi model yang terstruktur.
4. **Mendukung perbandingan dua pendekatan.** Data dapat digunakan untuk membandingkan metode pembelajaran mesin klasik, yaitu TF-IDF + SVM, dengan pendekatan berbasis transformer, yaitu IndoBERT.
5. **Mendukung evaluasi multikelas.** Dataset memiliki tiga label sentimen, sehingga penelitian dapat mengukur performa model pada masing-masing kelas.
6. **Memungkinkan evaluasi dengan macro F1.** Metrik ini digunakan sebagai metrik utama untuk menilai performa pada seluruh kelas, khususnya ketika distribusi kelas tidak seimbang.

## Pembagian Data dan Protokol Eksperimen

Dataset digunakan sesuai pembagian yang telah disediakan oleh sumber.

| Pembagian | Jumlah | Penggunaan |
|---|---:|---|
| Train | 11.000 | Melatih model |
| Validation | 1.260 | Memilih model dan parameter |
| Test | 500 | Evaluasi akhir |

Kedua metode, TF-IDF + SVM dan IndoBERT, menggunakan pembagian data yang sama agar hasil perbandingan lebih adil.

Data uji tidak digunakan untuk melatih model atau memilih parameter. Data tersebut hanya digunakan untuk evaluasi akhir setelah konfigurasi model ditetapkan.

Metrik evaluasi yang digunakan meliputi:

- Accuracy
- Precision
- Recall
- Macro F1-score

Macro F1 digunakan sebagai metrik utama karena menghitung F1 setiap kelas secara seimbang, tanpa memberikan bobot lebih besar kepada kelas yang jumlah datanya dominan.

## Format Dataset

Berkas sumber menggunakan format TSV (*Tab-Separated Values*) tanpa header. Data terdiri atas dua kolom:

| Kolom | Keterangan |
|---|---|
| `text` | Kalimat atau teks yang dianalisis |
| `label` | Label sentimen: `positive`, `neutral`, atau `negative` |

Skrip `download_smsa.py` membaca berkas TSV tersebut menggunakan pandas, kemudian menyimpan hasilnya dalam format CSV dengan nama kolom `text` dan `label`.

## Persiapan Lingkungan

Pastikan Python telah terpasang. Pustaka dasar yang dibutuhkan oleh skrip pengunduhan adalah pandas.

Instalasi:

```bash
pip install pandas
```

## Cara Mengunduh Dataset

Buka terminal pada folder `UTS/02-Dataset`, kemudian jalankan:

```bash
python download_smsa.py
```

Skrip akan mengunduh dan memuat tiga bagian dataset dari sumber IndoNLU, lalu menyimpan hasilnya sebagai:

- `smsa_train.csv`
- `smsa_valid.csv`
- `smsa_test.csv`

Berkas tersebut disimpan pada direktori kerja saat perintah dijalankan. Pastikan terminal berada pada folder yang sesuai jika ingin menyimpan hasil unduhan di lokasi tersebut.

Jumlah data yang diharapkan:

- `smsa_train.csv`: 11.000 baris
- `smsa_valid.csv`: 1.260 baris
- `smsa_test.csv`: 500 baris

Total yang diharapkan adalah 12.760 teks.

## Cara Menjalankan Eksplorasi

Setelah dataset berhasil diunduh, jalankan:

```bash
python eksplorasi_smsa.py
```

Skrip eksplorasi digunakan untuk memeriksa karakteristik dataset, termasuk:

- Jumlah data pada setiap pembagian.
- Distribusi label sentimen.
- Panjang teks.
- Duplikasi data.
- Penggunaan kata informal.
- Potensi irisan teks antarpembagian data.

Hasil eksplorasi digunakan untuk mendukung pemahaman karakteristik data sebelum preprocessing dan pelatihan model.

## Alur Penggunaan Dataset

1. Mengunduh dataset SmSA dari sumber IndoNLU.
2. Memvalidasi struktur dan jumlah data.
3. Melakukan eksplorasi karakteristik dataset.
4. Melakukan preprocessing sesuai kebutuhan masing-masing metode.
5. Melatih model TF-IDF + SVM menggunakan data latih.
6. Melakukan fine-tuning IndoBERT menggunakan data latih dan validasi.
7. Menetapkan konfigurasi model berdasarkan hasil validasi.
8. Mengevaluasi kedua model menggunakan data uji.
9. Membandingkan hasil berdasarkan accuracy, precision, recall, dan macro F1-score.
10. Mendokumentasikan kode, konfigurasi, dan hasil eksperimen untuk mendukung reproduksibilitas penelitian.

## Catatan Reproduksibilitas

Skrip pengunduhan mengambil data dari URL sumber IndoNLU. Hasil pengunduhan dapat bergantung pada ketersediaan sumber dan perubahan pada repositori asal. Untuk reproduksibilitas, catat sumber dataset, tanggal pengunduhan, konfigurasi eksperimen, serta versi pustaka yang digunakan.

Dataset lengkap disimpan secara lokal untuk keperluan eksperimen. Repositori ini menyediakan skrip pengunduhan dan dokumentasi agar pengguna dapat memperoleh dataset dari sumber asal sesuai ketentuan yang berlaku.
