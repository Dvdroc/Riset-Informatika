# 03 - Research Gap

**Topik:** Perbandingan TF-IDF + SVM dan IndoBERT untuk klasifikasi sentimen berbahasa Indonesia (dataset SmSA), dengan dan tanpa normalisasi kata tidak baku.

## 1. Tabel penelitian terkait

| No | Referensi | Tugas dan data | Metode | Temuan utama | Keterbatasan relevan |
|---|---|---|---|---|---|
| [4] | Devlin dkk. (2019) | Pre-training bahasa umum | BERT | Dasar arsitektur pre-training dua arah | Terutama bahasa Inggris, bukan bahasa Indonesia |
| [5] | Wilie dkk. (2020) | Benchmark IndoNLU, termasuk SmSA | IndoBERT dan baseline lain | Model pre-trained unggul dibanding baseline | Cakupan luas; tidak menganalisis efek normalisasi dan biaya komputasi secara rinci |
| [6] | Koto dkk. (2020) | Benchmark IndoLEM | IndoBERT (IndoLEM) | Model pre-trained efektif untuk NLP Indonesia | Evaluasi lintas tugas; kurang fokus pada pembanding klasik pada sumber daya terbatas |
| [7] | Koto dkk. (2021) | Teks Twitter Indonesia | IndoBERTweet | Adaptasi domain Twitter | Domain media sosial, belum tentu optimal untuk ulasan |
| [8] | Pang dkk. (2002) | Ulasan film (Inggris) | Naive Bayes, MaxEnt, SVM | Metode klasik efektif, sentimen lebih sulit dari topik | Bahasa Inggris; belum ada model Transformer |
| [9] | Saputri dkk. (2018) | Emosi tweet Indonesia | Rekayasa fitur + classifier klasik | F1-score 69,73% dengan kombinasi fitur | Tugas emosi, bukan sentimen; belum membandingkan dengan model pre-trained |
| [10] | Nugroho dkk. (2022) | Emosi tweet Indonesia | Variasi pembobotan term | TF-ICF dan TF-RF terbaik | Fokus pembobotan fitur; tidak membandingkan dengan model pre-trained |
| [11] | Hilmiaji dkk. (2021) | Emosi tweet Indonesia | CNN | F1-score 90,2% pada dataset yang sama | Tugas emosi; tidak ada baseline klasik terkontrol |
| [12] | Purwarianti dan Crisdayanti (2019) | Sentimen bahasa Indonesia | Bi-LSTM + paragraph vector | Peningkatan kinerja signifikan | Pra-BERT; tidak ada model pre-trained |
| [13] | Murfi dkk. (2022) | Sentimen bahasa Indonesia | BERT + hibrida CNN/RNN | LSTM-CNN sedikit lebih baik dari hibrida lain | Fokus arsitektur hibrida, bukan perbandingan dengan baseline klasik dan biaya komputasi |

Nomor mengikuti daftar pustaka pada proposal. Tambahkan 1-2 paper hasil pencarianmu sendiri (Google Scholar, kata kunci: `Indonesian sentiment analysis IndoBERT SVM`) agar tabel ini mencerminkan bacaanmu.

## 2. Celah penelitian

1. **Perbandingan terkontrol.** Model pre-trained dan metode klasik masing-masing banyak dilaporkan, tetapi pada dataset, prapemrosesan, dan protokol evaluasi yang berbeda-beda sehingga selisih kinerja sulit ditafsirkan.
2. **Pengaruh normalisasi teks.** Normalisasi kata tidak baku lazim dilakukan, tetapi pengaruhnya jarang diuji terpisah dan dibandingkan antara model berbasis kata (TF-IDF) dan berbasis subword (IndoBERT).
3. **Biaya komputasi dan reprodusibilitas.** Kinerja jarang dilaporkan bersama waktu latih, waktu inferensi, dan konfigurasi lengkap.

## 3. Kesimpulan gap (satu paragraf)

Penelitian sebelumnya telah menunjukkan keunggulan model pre-trained bahasa Indonesia dan penggunaan model klasik pada dataset tertentu, tetapi belum banyak yang membandingkan keduanya secara langsung pada dataset benchmark yang sama dengan kontrol prapemrosesan (termasuk normalisasi dan uji ketahanan terhadap teks informal), uji statistik, serta pelaporan biaya komputasi. Penelitian ini mengisi celah tersebut.

## 4. Pertanyaan penelitian yang muncul dari gap

1. Seberapa besar selisih macro F1 antara TF-IDF + SVM dan IndoBERT pada SmSA?
2. Apakah normalisasi kata tidak baku mengubah kinerja, dan apakah efeknya berbeda antarmodel?
3. Berapa biaya waktu dan memori tambahan IndoBERT dibandingkan SVM?

## 5. Daftar pustaka (IEEE)

[1] G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Inf. Process. Manage.*, vol. 24, no. 5, pp. 513-523, 1988, doi: 10.1016/0306-4573(88)90021-0.

[2] C. Cortes and V. Vapnik, "Support-vector networks," *Mach. Learn.*, vol. 20, pp. 273-297, Sep. 1995, doi: 10.1007/BF00994018.

[3] A. Vaswani *et al.*, "Attention is all you need," in *Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 30, 2017, pp. 5998-6008. [Online]. Available: https://arxiv.org/abs/1706.03762

[4] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, vol. 1, Minneapolis, MN, USA, Jun. 2019, pp. 4171-4186, doi: 10.18653/v1/N19-1423.

[5] B. Wilie *et al.*, "IndoNLU: Benchmark and resources for evaluating Indonesian natural language understanding," in *Proc. AACL-IJCNLP*, Suzhou, China, Dec. 2020, pp. 843-857, doi: 10.18653/v1/2020.aacl-main.85.

[6] F. Koto, A. Rahimi, J. H. Lau, and T. Baldwin, "IndoLEM and IndoBERT: A benchmark dataset and pre-trained language model for Indonesian NLP," in *Proc. COLING*, Barcelona, Spain (Online), Dec. 2020, pp. 757-770, doi: 10.18653/v1/2020.coling-main.66.

[7] F. Koto, J. H. Lau, and T. Baldwin, "IndoBERTweet: A pretrained language model for Indonesian Twitter with effective domain-specific vocabulary initialization," in *Proc. EMNLP*, Nov. 2021, pp. 10660-10668, doi: 10.18653/v1/2021.emnlp-main.833.

[8] B. Pang, L. Lee, and S. Vaithyanathan, "Thumbs up? Sentiment classification using machine learning techniques," in *Proc. EMNLP*, Jul. 2002, pp. 79-86, doi: 10.3115/1118693.1118704.

[9] M. S. Saputri, R. Mahendra, and M. Adriani, "Emotion classification on Indonesian Twitter dataset," in *Proc. IALP*, Bandung, Indonesia, Nov. 2018, doi: 10.1109/IALP.2018.8629262.

[10] K. S. Nugroho, F. A. Bachtiar, and W. F. Mahmudy, "Detecting emotion in Indonesian tweets: A term-weighting scheme study," *J. Inf. Syst. Eng. Bus. Intell.*, vol. 8, no. 1, pp. 61-70, Apr. 2022, doi: 10.20473/jisebi.8.1.61-70.

[11] N. Hilmiaji, K. M. Lhaksmana, and M. D. Purbolaksono, "Identifying emotion on Indonesian tweets using convolutional neural networks," *J. RESTI*, vol. 5, no. 3, pp. 584-593, Jun. 2021, doi: 10.29207/resti.v5i3.3137.

[12] A. Purwarianti and I. A. P. A. Crisdayanti, "Improving Bi-LSTM performance for Indonesian sentiment analysis using paragraph vector," in *Proc. ICAICTA*, 2019, doi: 10.1109/ICAICTA.2019.8904199.

[13] H. Murfi *et al.*, "BERT-based combination of convolutional and recurrent neural network for Indonesian sentiment analysis," arXiv:2211.05273, 2022, doi: 10.48550/arXiv.2211.05273.

[14] T. Wolf *et al.*, "Transformers: State-of-the-art natural language processing," in *Proc. EMNLP: System Demonstrations*, Oct. 2020, pp. 38-45, doi: 10.18653/v1/2020.emnlp-demos.6.

[15] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825-2830, Oct. 2011. [Online]. Available: https://jmlr.org/papers/v12/pedregosa11a.html

[16] M. Sokolova and G. Lapalme, "A systematic analysis of performance measures for classification tasks," *Inf. Process. Manage.*, vol. 45, pp. 427-437, 2009, doi: 10.1016/j.ipm.2009.03.002.
