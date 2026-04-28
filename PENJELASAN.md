# Panduan Memahami Proyek FTA LLM
**Diyouva Christa Novith** | Machine Learning Foundation with Python | Carnegie Mellon University — Spring 2026

---

> **Catatan dokumen (April 2026):** file ini adalah panduan konseptual. Untuk
> langkah eksekusi terbaru, gunakan `README.md` dan `run_pipeline.py`. Setelah
> ada perubahan kode pada extraction / sampling / validation / comparison,
> artefak JSON lama harus dibuat ulang sebelum angka hasil di sini dianggap
> mutakhir.

## Daftar Isi

1. [Gambaran Besar — Proyek Ini Tentang Apa?](#1-gambaran-besar)
2. [Istilah-Istilah Penting](#2-istilah-istilah-penting)
3. [Pipeline — Alur Kerja Step by Step](#3-pipeline--alur-kerja-step-by-step)
4. [Hasil dan Temuan Utama](#4-hasil-dan-temuan-utama)
5. [Cara Membaca Dashboard](#5-cara-membaca-dashboard)
6. [Kesimpulan](#6-kesimpulan)

---

## 1. Gambaran Besar

### Masalahnya Apa?

Bayangkan kamu adalah **analis kebijakan perdagangan** di sebuah kementerian. Tugasmu adalah membandingkan isi tiga perjanjian perdagangan bebas (Free Trade Agreement / FTA) berikut:

| Kode | Nama Lengkap | Pihak | Ditandatangani |
|------|-------------|-------|----------------|
| **RCEP** | Regional Comprehensive Economic Partnership | 15 negara ASEAN+ | 2020 |
| **AHKFTA** | ASEAN–Hong Kong Free Trade Agreement | 10 ASEAN + HK | 2017 |
| **AANZFTA** | ASEAN–Australia–New Zealand FTA | 10 ASEAN + AU + NZ | 2009 |

Masalahnya: masing-masing perjanjian itu **tebal sekali** — ribuan halaman teks hukum. Kalau dibandingkan secara manual, bisa butuh **berminggu-minggu** waktu analis.

### Solusinya?

Gunakan **AI (Large Language Model / LLM)** untuk secara otomatis:
1. **Membaca** seluruh dokumen PDF perjanjian
2. **Mengklasifikasikan** setiap ketentuan ke dalam kategori kebijakan
3. **Membandingkan** isi antar perjanjian secara lintas-dokumen

Proyek ini membangun **pipeline otomatis** dari PDF mentah hingga analisis komparatif lengkap — yang bisa dijalankan siapa saja hanya dengan API key gratis dari Groq.

### Tiga Pertanyaan Riset

> **RQ1:** Bisakah LLM secara andal mengklasifikasikan ketentuan FTA ke dalam kategori kebijakan standar? Seberapa akurat, dan apakah strategi prompting berpengaruh?

> **RQ2:** Bagaimana ketiga perjanjian berbeda dalam fitur desain kebijakan mereka — seperti kriteria rules of origin, fleksibilitas hukum, dan struktur komitmen?

> **RQ3:** Apakah ketiga perjanjian menunjukkan pola konvergensi (semakin mirip) atau fragmentasi (tetap berbeda) dalam desain hukumnya?

---

## 2. Istilah-Istilah Penting

### Istilah Umum

| Istilah | Penjelasan |
|---------|------------|
| **FTA** | Free Trade Agreement — perjanjian perdagangan bebas antar negara/kawasan |
| **Provision** | Satu ketentuan atau klausul hukum dalam perjanjian (setara satu paragraf bermakna) |
| **Corpus** | Kumpulan seluruh provisions dari semua perjanjian (total: 4.059 provisions) |
| **Gold Set** | 50 provisions yang sudah diberi label benar secara manual oleh manusia — dipakai untuk menguji akurasi AI |
| **Pipeline** | Alur kerja otomatis dari input (PDF) hingga output (analisis dan grafik) |

### Istilah AI / Machine Learning

| Istilah | Penjelasan |
|---------|------------|
| **LLM** | Large Language Model — model AI yang memahami dan menghasilkan teks, seperti ChatGPT |
| **LLaMA 3.3 70B** | Model AI buatan Meta (Facebook), 70 miliar parameter, diakses gratis via Groq |
| **Qwen 3 32B** | Model AI buatan Alibaba Cloud, 32 miliar parameter, juga diakses gratis via Groq |
| **Groq** | Platform cloud yang menyediakan akses gratis ke LLaMA dan Qwen dengan kuota harian |
| **Token** | Satuan teks yang diproses AI — kira-kira 1 token ≈ ¾ kata dalam bahasa Inggris |
| **API Key** | "Kunci" untuk mengakses layanan AI — seperti password untuk masuk ke sistem |
| **Embedding** | Proses mengubah teks menjadi deretan angka yang merepresentasikan **makna** teks tersebut |
| **Vektor** | Deretan angka hasil embedding — provisions yang maknanya mirip akan punya vektor yang berdekatan |
| **ChromaDB** | Database khusus untuk menyimpan dan mencari vektor (makna teks), bukan kata-katanya |
| **Semantic Search** | Pencarian berdasarkan **makna**, bukan kesamaan kata — "mobil" dan "kendaraan" dianggap mirip |
| **RAG** | Retrieval-Augmented Generation — AI diberi konteks tambahan (provisions relevan) sebelum menjawab |

### Istilah Strategi Prompting

| Strategi | Cara Kerja | Analogi |
|----------|-----------|---------|
| **Zero-shot** | AI langsung menjawab tanpa diberi contoh sama sekali | Ujian mendadak tanpa belajar |
| **Few-shot** | AI diberi 2 contoh berlabel sebelum menjawab | Dikasih 2 soal contoh + jawaban dulu |
| **CoT (Chain-of-Thought)** | AI diperintah untuk "berpikir langkah demi langkah" sebelum memberi jawaban | Disuruh tunjukkan cara pengerjaannya |

### Istilah Metrik

| Metrik | Penjelasan | Range |
|--------|-----------|-------|
| **Accuracy** | Persentase jawaban AI yang benar dari 50 provisions | 0% – 100% |
| **Macro-F1** | Rata-rata F1 per kategori — **tidak bias** ke kategori yang banyak; menghukum jika ada kategori yang diabaikan | 0.0 – 1.0 |
| **Cohen's κ (kappa)** | Seberapa konsisten dua model dalam memberi label yang sama, **dikurangi** kebetulan | −1 s/d 1 |
| **Entropy** | Ukuran "keacakan" distribusi — dipakai untuk mengukur konvergensi/fragmentasi antar perjanjian | ≥ 0 |

> **Cara baca Cohen's κ:**
> | Nilai κ | Interpretasi |
> |---------|-------------|
> | < 0.20 | Hampir tidak ada kesepakatan |
> | 0.21 – 0.40 | Kesepakatan lemah |
> | 0.41 – 0.60 | Kesepakatan sedang |
> | 0.61 – 0.80 | Kesepakatan substansial |
> | > 0.80 | Kesepakatan hampir sempurna |

### Istilah Kebijakan Perdagangan

| Istilah | Penjelasan |
|---------|------------|
| **Rules of Origin (RoO)** | Aturan yang menentukan suatu produk "buatan negara mana" — penting untuk menentukan apakah bisa dapat tarif preferensial |
| **RVC (Regional Value Content)** | Persentase nilai produk yang harus berasal dari dalam kawasan — misal RVC 40% artinya minimal 40% nilai produk harus dari negara anggota |
| **CTC (Change in Tariff Classification)** | Cara buktikan origin: bahan baku impor harus berubah klasifikasi HS code-nya setelah diproses |
| **CC (Change in Chapter)** | CTC level ketat — bahan harus berubah **chapter** (2 digit pertama HS code) |
| **CTH (Change in Tariff Heading)** | CTC level menengah — bahan cukup berubah **heading** (4 digit pertama HS code), lebih longgar dari CC |
| **CTSH (Change in Tariff Sub-Heading)** | CTC level paling longgar — cukup berubah **sub-heading** (6 digit HS code) |
| **HS Code** | Harmonized System Code — kode internasional untuk mengklasifikasikan produk perdagangan (misal: 8471 = komputer) |
| **De Minimis** | Batas toleransi — bahan yang tidak memenuhi syarat CTC masih boleh dipakai asal di bawah persentase tertentu dari nilai produk |
| **Tariff Commitment** | Jadwal komitmen pengurangan/penghapusan bea masuk — kapan tarif diturunkan dan seberapa banyak |
| **Non-Tariff Measures** | Hambatan perdagangan selain bea cukai — misal lisensi impor, kuota, standar teknis |
| **ISDS** | Investor-State Dispute Settlement — mekanisme investor asing menggugat pemerintah ke arbitrase internasional |
| **Mode 1–4** | Empat cara perdagangan jasa internasional (lintas batas, konsumsi luar negeri, kehadiran komersial, pergerakan orang) |

---

## 3. Pipeline — Alur Kerja Step by Step

```
PDFs (7 dokumen dari 3 FTA)
        │
        ▼
┌─────────────────────┐
│   STEP 1: Ekstraksi │  PDF → teks → dipotong jadi provisions
│   (extraction.py)   │  pdfplumber → PyMuPDF → Tesseract OCR
└─────────────────────┘
        │  all_provisions.json (4.059 provisions)
        ▼
┌─────────────────────┐
│   STEP 2: Embedding │  Teks → vektor makna → disimpan di ChromaDB
│   (embedding.py)    │  Model: all-MiniLM-L6-v2
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  STEP 3: Klasifikasi│  AI baca provision → pilih 1 dari 11 kategori
│  (classification.py)│  2 model × 3 strategi = 6 kombinasi run
└─────────────────────┘
        │  classified_*.json (6 file hasil)
        ▼
┌──────────────┐   ┌─────────────────────┐   ┌──────────────────┐
│  STEP 4:     │   │  STEP 5:            │   │  STEP 6:         │
│  Validasi    │   │  Perbandingan RAG   │   │  Atribut         │
│  (50 gold)   │   │  (comparison.py)    │   │  (RVC%, CTC, HS) │
└──────────────┘   └─────────────────────┘   └──────────────────┘
        │
        ▼
┌─────────────────────┐
│  STEP 7: Analisis   │  Cohen's κ, entropy, convergence signal
│  (analysis.py)      │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  STEP 8: Visualisasi│  8 grafik PNG untuk laporan
│  (visualize.py)     │
└─────────────────────┘
```

---

### Step 1 — Ekstraksi PDF

**Tujuan:** Mengubah PDF dokumen hukum menjadi 4.059 provisions terstruktur.

**Cara kerja:**
- Coba baca PDF dengan **pdfplumber** (library Python untuk PDF teks)
- Kalau gagal, coba **PyMuPDF** (library alternatif)
- Kalau PDF-nya hasil scan (gambar), pakai **Tesseract OCR** (AI pembaca gambar → teks)
- Provisions yang terlalu pendek (< 80 karakter) atau terlalu panjang (> 1.500 karakter) dibuang

**Setiap provision disimpan dengan:**
```
id, agreement, doc_type, chapter, article, paragraph_idx, text, char_count
```

**Kenapa AHKFTA butuh OCR?** Dokumen AHKFTA discan dari kertas fisik, jadi isinya adalah gambar — bukan teks. Tesseract membaca gambar tersebut dan mengubahnya jadi teks.

**Distribusi provisions hasil ekstraksi:**

| Perjanjian | Jumlah Provisions | Persentase |
|-----------|------------------|-----------|
| RCEP | 2.171 | 53.5% |
| AANZFTA | 1.526 | 37.6% |
| AHKFTA | 362 | 8.9% |
| **Total** | **4.059** | **100%** |

> **Masalah:** RCEP mendominasi 53.5% corpus. Kalau analisis tidak disesuaikan, hasil akan bias ke RCEP. Solusinya: gunakan **stratified sample** — ambil 100 provisions per perjanjian untuk analisis komparatif.

---

### Step 2 — Embedding dan Vector Store

**Tujuan:** Membuat sistem pencarian berdasarkan **makna**, bukan kata-kata.

**Cara kerja:**
1. Setiap provision dimasukkan ke model `all-MiniLM-L6-v2` (model kecil dan cepat dari sentence-transformers)
2. Model menghasilkan **vektor** — 384 angka yang merepresentasikan makna teks
3. Vektor disimpan di **ChromaDB** (database vektor)

**Kenapa perlu ini?**
Untuk membandingkan provisions lintas perjanjian, kita perlu menemukan ketentuan yang **topiknya sama** meski kata-katanya berbeda. Semantic search bisa menemukan:
- RCEP: *"The regional value content shall not be less than 40%"*
- AHKFTA: *"...provided that the RVC is at least forty percent (40%)..."*

Keduanya bermakna sama meski susunan katanya beda.

---

### Step 3 — Klasifikasi LLM

**Tujuan:** Memberi label kategori pada setiap provision secara otomatis.

**11 Kategori Kebijakan:**

| # | Kategori | Contoh Provision |
|---|---------|-----------------|
| 1 | **Tariff Commitments** | Jadwal pengurangan bea cukai, timeline penghapusan tarif |
| 2 | **Rules of Origin** | Threshold RVC, aturan CTC, kriteria substantial transformation |
| 3 | **Non-Tariff Measures** | Lisensi impor, standar teknis, kuota |
| 4 | **Trade in Services** | Komitmen liberalisasi jasa Mode 1–4 |
| 5 | **Investment** | ISDS, national treatment untuk investor |
| 6 | **Dispute Settlement** | Prosedur panel, timeline konsultasi |
| 7 | **Customs Procedures** | Dokumentasi, advance ruling, single window |
| 8 | **Sanitary & Phytosanitary** | Standar keamanan pangan, kesehatan hewan/tanaman |
| 9 | **Intellectual Property** | Hak cipta, merek dagang, perlindungan GI |
| 10 | **General Provisions** | Definisi, ruang lingkup, pengecualian umum |
| 11 | **Other** | Tidak masuk kategori manapun |

**Contoh prompt zero-shot:**
```
You are a trade policy analyst. Classify the following FTA provision into exactly 
one of these categories: [daftar 11 kategori]

Provision: "Each Party shall ensure that its customs procedures are applied in a 
manner that is predictable, consistent and transparent."

Category:
```
AI akan menjawab: `Customs Procedures`

**Contoh prompt CoT:**
```
...Think step by step about what policy area this provision addresses, then 
state your final category on the last line.

Provision: "..."

Let me analyze this step by step:
[AI berpikir dulu...]
Final category: Customs Procedures
```

---

### Step 4 — Validasi

**Tujuan:** Mengukur seberapa akurat AI dibandingkan penilaian manusia.

**Cara kerja:**
1. Buat **validation sample** 50 provisions dari cohort yang seimbang
2. Penulis proyek memberi label manual — ini disebut **gold set**
3. Export cohort yang sama ke `validation_provisions.json`
4. Semua 6 kombinasi model dijalankan pada 50 provisions yang sama
5. Hasil AI dibandingkan dengan label manusia

**Perbedaan Accuracy vs Macro-F1:**
- **Accuracy**: dari 50 provisions, berapa yang benar? → bisa menyesatkan kalau satu kategori mendominasi
- **Macro-F1**: rata-rata performa di semua kategori — lebih adil, menghukum model yang mengabaikan kategori langka

---

### Step 5 — Perbandingan RAG

**Tujuan:** Menghasilkan narasi analisis komparatif lintas perjanjian per kategori.

**Cara kerja:**
1. Untuk setiap kategori (misal: Rules of Origin), ambil **3 provisions paling relevan** dari masing-masing perjanjian via ChromaDB
2. Provisions tersebut dikirim ke AI sebagai konteks
3. AI diminta menulis analisis: persamaan, perbedaan, fleksibilitas, konvergensi, dan implikasi kebijakan

**Hasilnya:** Narasi komparatif seperti:
> *"Ketiga perjanjian menetapkan threshold RVC 40%, namun berbeda dalam persyaratan transformasi substantifnya. AHKFTA menggunakan CC (chapter-level CTC) yang lebih ketat, sementara RCEP menggunakan CTH (heading-level) yang lebih longgar..."*

---

### Step 6 — Ekstraksi Atribut

**Tujuan:** Mengekstrak nilai numerik dan flag kategorikal dari provisions Rules of Origin dan Tariff Commitments.

**Dua pendekatan:**

| Pendekatan | Dipakai Untuk | Contoh |
|-----------|--------------|--------|
| **Regex** (pencarian pola teks) | Angka-angka spesifik | RVC 40%, de minimis 10%, phase-out 10 years |
| **LLM** | Klasifikasi kategorikal | Jenis CTC (CC/CTH/CTSH), scope HS code, staging category |

**Temuan kunci dari ekstraksi atribut:**
- Semua 3 perjanjian menggunakan threshold **RVC 40%**
- Tapi AHKFTA mewajibkan **CC** (ganti chapter) vs RCEP yang hanya perlu **CTH** (ganti heading)
- AHKFTA lebih **ketat** dalam persyaratan transformasi → produsen harus melakukan perubahan yang lebih besar pada bahan bakunya

---

### Step 7 — Analisis Statistik

**Cohen's κ — Konsistensi Antar Model:**

| Pasangan | κ | Interpretasi |
|---------|---|-------------|
| LLaMA zero-shot vs LLaMA few-shot | 0.668 | Substantial — cukup konsisten |
| Qwen zero-shot vs Qwen few-shot | 0.689 | Substantial — cukup konsisten |
| LLaMA few-shot vs Qwen few-shot | 0.582 | Moderat — beda, tapi tidak liar |

> **Artinya:** Meski dua model punya akurasi yang mirip, mereka sering **tidak setuju pada provisions yang sama**. Ini penting — tidak berarti kalau model A benar, model B juga benar di provision yang sama.

**Convergence Signal — Entropy:**
- Dihitung per kategori: seberapa seragam distribusi provisions di ketiga perjanjian?
- **Entropy rendah** (konvergen) = ketiga perjanjian punya proporsi provisions yang mirip di kategori ini
- **Entropy tinggi** (fragmentasi) = distribusinya sangat berbeda antar perjanjian

---

### Step 8 — Visualisasi

8 grafik dihasilkan otomatis oleh `src/visualize.py`:

| File | Isi |
|------|-----|
| `fig_corpus_overview.png` | Ukuran corpus per perjanjian dan tipe dokumen |
| `fig_category_heatmap.png` | Distribusi kategori (%) di semua 6 model-strategi run |
| `fig_kappa_matrix.png` | Matrix Cohen's κ antar semua pasangan run |
| `fig_category_x_agreement.png` | Heatmap jumlah provisions: kategori × perjanjian |
| `fig_strategy_effect_llama.png` | Bagaimana strategi prompting mengubah distribusi LLaMA |
| `fig_strategy_effect_qwen.png` | Bagaimana strategi prompting mengubah distribusi Qwen |
| `fig_convergence.png` | Signal konvergensi berbasis entropy per kategori |
| `fig_validation_accuracy.png` | Akurasi dan Macro-F1 per model + strategi |

---

## 4. Hasil dan Temuan Utama

### RQ1 — Klasifikasi yang Andal?

| Model + Strategi | Accuracy | Macro-F1 | n |
|-----------------|----------|----------|---|
| **LLaMA 3.3 70B — Zero-shot** | **48.0%** | 0.431 | 50 |
| **Qwen 3 32B — CoT** | 46.0% | **0.442** | 50 |
| Qwen 3 32B — Zero-shot | 38.0% | 0.424 | 50 |
| Qwen 3 32B — Few-shot | 38.0% | 0.373 | 50 |
| LLaMA 3.3 70B — Few-shot | 34.0% | 0.336 | 50 |
| LLaMA 3.3 70B — CoT | 32.0% | 0.327 | 50 |

**Interpretasi yang lebih jujur setelah rerun:**
- Tidak ada konfigurasi yang menembus 50% accuracy
- LLaMA zero-shot paling tinggi di accuracy
- Qwen CoT paling tinggi di macro-F1, tapi selisihnya kecil
- Jadi output klasifikasi lebih cocok untuk **triage analis** daripada label final

---

### RQ2 — Perbedaan Desain Kebijakan

| Dimensi | RCEP | AHKFTA | AANZFTA |
|---------|------|--------|---------|
| Cakupan | Komprehensif (barang, jasa, investasi) | Hanya barang | Komprehensif |
| Rules of Origin | CTH (heading-level), RVC 40% | CC (chapter-level), RVC 40% | CTH/CTSH, delegasi ke jadwal |
| Keketatan RoO | Moderat | Tinggi (CC lebih ketat) | Moderat |
| Tariff Scheduling | Di jadwal annex | Di jadwal annex | Di jadwal annex (banyak delegasi) |
| Bab Jasa | Ada | Tidak ada | Ada |
| Bab Investasi | Ada | Tidak ada | Ada |

---

### RQ3 — Konvergensi atau Fragmentasi?

| Kategori | Status | Penjelasan |
|---------|--------|-----------|
| General Provisions | **Konvergen** | Template regional mulai terbentuk — ketiga perjanjian punya struktur dan bahasa yang mirip |
| Dispute Settlement | **Konvergen** | Prosedur konsultasi dan panel sudah hampir seragam |
| Tariff Commitments | **Fragmentasi** | Setiap perjanjian punya pendekatan jadwal yang sangat berbeda |
| Rules of Origin | **Fragmentasi** | CC vs CTH vs CTSH — tiap perjanjian punya desain sendiri |

---

## 5. Cara Membaca Dashboard

Dashboard HTML (`index.html`) bisa dibuka langsung di browser — tidak perlu internet atau server khusus.

---

### Halaman 1: Overview

```
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  4,059       │ │  0.442       │ │  6           │ │  11          │
│  PROVISIONS  │ │  BEST F1     │ │  LLM RUNS    │ │  CATEGORIES  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘

        ┌─────────────────┐    ┌──────────────────────────┐
        │  Corpus Donut   │    │  Research Questions       │
        │  RCEP: 53.5%    │    │  RQ1 ✓ triage only       │
        │  AANZFTA: 37.6% │    │  RQ2 ✓ CC vs CTH         │
        │  AHKFTA: 8.9%   │    │  RQ3 ✓ GP convergent     │
        └─────────────────┘    └──────────────────────────┘

        ┌────────────────────────────────────────────────────┐
        │  Pipeline: PDF → Extraction → Embedding →          │
        │           Classification → RAG → Validation        │
        └────────────────────────────────────────────────────┘
```

**Yang perlu diperhatikan:**
- Donut chart: RCEP sangat mendominasi — itulah kenapa stratified sample (100/perjanjian) dipakai
- Research Questions: setiap RQ sudah dijawab dengan temuan singkatnya

---

### Halaman 2: Validation

Bar chart menampilkan **Accuracy** dan **Macro-F1** untuk semua 6 kombinasi.

**Cara baca:**
- Bar **biru** = Accuracy (berapa % yang benar dari 50)
- Bar **hijau/oranye** = Macro-F1 (akurasi yang sudah disesuaikan untuk keadilan antar kategori)
- **Qwen CoT** = bar tertinggi → konfigurasi terbaik
- **LLaMA CoT** = bar terendah → CoT justru merugikan LLaMA

**Insight penting:** metrik sekarang jauh lebih rendah dari draft awal repo. LLaMA zero-shot unggul di accuracy, tetapi Qwen CoT sedikit unggul di macro-F1. Itu berarti pilihan model tergantung apakah kamu lebih peduli pada hit rate total atau pemerataan performa antar kategori.

---

### Halaman 3: Inter-Run Agreement

Tabel Cohen's κ — baca seperti tabel jarak/korelasi:
- **Diagonal** = model vs dirinya sendiri (selalu 1.00)
- **Nilai mendekati 0** = dua model hampir tidak setuju
- **Nilai negatif** = lebih buruk dari kalau jawab acak!

**Yang menarik:** setelah cohort disejajarkan dengan benar, κ tidak lagi mendekati nol. Masalah utamanya sekarang bukan chaos antar model, tetapi akurasi absolut yang masih rendah terhadap gold labels.

---

### Halaman 4: Provision Distribution

Heatmap **kategori × perjanjian** (dari stratified sample 100/perjanjian):

- **Warna lebih gelap** = proporsi lebih tinggi
- AHKFTA berwarna gelap di Rules of Origin (28%) — perjanjian ini sangat fokus pada aturan asal barang
- RCEP lebih merata — mencakup lebih banyak topik kebijakan
- AANZFTA seimbang antara rules dan jasa/investasi

---

### Halaman 5: Convergence Analysis

Grafik entropy menunjukkan **seberapa seragam** distribusi kategori antar perjanjian:

- **Bar hijau pendek** = entropy rendah = **konvergen** (ketiga perjanjian mirip di kategori ini)
- **Bar oranye panjang** = entropy tinggi = **fragmentasi** (ketiga perjanjian sangat berbeda)

**Baca dari kiri ke kanan:**
- Kiri (pendek/hijau) = kategori yang sudah mulai punya template regional seragam
- Kanan (panjang/oranye) = kategori yang tiap perjanjian masih punya desain sendiri-sendiri

---

### Halaman 6: Policy Design Matrix

Tabel fitur desain per perjanjian — mirip seperti tabel spesifikasi produk di toko elektronik.

**Yang perlu diperhatikan:**
- Baris **CTC Rule** → AHKFTA: CC (ketat), RCEP: CTH (longgar)
- Baris **RVC Threshold** → semua 40%, tapi efeknya beda karena CTC rule berbeda
- Highlight kuning = perbedaan signifikan yang perlu diperhatikan negosiator

---

### Halaman 7: RAG Comparisons

Accordion berisi narasi komparatif AI per kategori kebijakan.

**Cara pakai:**
- Klik nama kategori (misal "Rules of Origin") untuk buka narasi
- Narasi mencakup: persamaan, perbedaan, tingkat fleksibilitas, sinyal konvergensi, dan implikasi kebijakan
- Ini adalah output dari RAG pipeline — AI sudah membaca provisions relevan dari ketiga perjanjian sebelum menulis analisis

**Catatan:** Narasi ini **buatan AI** — dengan akurasi validasi saat ini yang masih rendah, detail penting tetap harus diverifikasi ke teks asli.

---

### Halaman 8: Figures

Galeri 8 grafik PNG yang dihasilkan otomatis oleh pipeline Python.

**Cara baca setiap grafik:**

| Grafik | Yang Perlu Dilihat |
|--------|-------------------|
| `fig_corpus_overview` | Proporsi tiap perjanjian — kenapa perlu stratified sample |
| `fig_category_heatmap` | Apakah distribusi kategori konsisten antar run? Kalau tidak, model tidak stabil |
| `fig_kappa_matrix` | Kotak hitam/gelap di diagonal = normal; warna terang di luar diagonal = kesepakatan tinggi |
| `fig_category_x_agreement` | Dimana perbedaan terbesar antar perjanjian? |
| `fig_strategy_effect_*` | Bagaimana mengubah strategi prompting menggeser distribusi kategori? |
| `fig_convergence` | Kategori mana yang sudah konvergen, mana yang masih fragmentasi? |
| `fig_validation_accuracy` | Grafik perbandingan performa — konfirmasi Qwen CoT sebagai konfigurasi terbaik |

---

## 6. Kesimpulan

### Jawaban Tiga Pertanyaan Riset

**RQ1 — Klasifikasi andal?**
Ya, tapi hanya sebagai alat bantu awal. Artefak repo saat ini menunjukkan **48% accuracy terbaik** dan **0.442 macro-F1 terbaik**, jadi sistem ini cocok untuk triage awal oleh analis, bukan keputusan hukum final.

**RQ2 — Perbedaan desain kebijakan?**
AHKFTA adalah perjanjian khusus **barang** dengan Rules of Origin paling ketat (CC, chapter-level CTC). RCEP adalah perjanjian paling komprehensif dengan RoO lebih longgar (CTH, heading-level). Keduanya menetapkan threshold RVC 40%, tapi intensitas persyaratan transformasinya berbeda signifikan.

**RQ3 — Konvergensi atau fragmentasi?**
**Keduanya.** General Provisions dan Dispute Settlement sudah mulai **konvergen** — template regional Asia-Pasifik mulai terbentuk. Tapi Tariff Commitments dan Rules of Origin masih sangat **terfragmentasi** — tiap perjanjian tetap mempertahankan desainnya sendiri.

### Batas Kemampuan Sistem Ini

| Keterbatasan | Artinya |
|-------------|---------|
| Akurasi 32–48% | Cocok hanya untuk penyaringan awal; harus diverifikasi ke sumber asli untuk keputusan penting |
| Annex tidak tercover | Jadwal tarif numerik (yang ada di lampiran) tidak ikut dianalisis |
| Tiga perjanjian saja | Temuan sugestif, belum bisa digeneralisasi ke semua FTA Asia-Pasifik |
| Bahasa Inggris only | Dokumen terjemahan tidak tercakup |
| Kuota API gratis | Groq free tier membatasi jumlah provisions yang bisa diproses per hari |

### Kontribusi Utama

Pipeline ini mengubah **ratusan halaman teks hukum** menjadi **data terstruktur yang bisa dibandingkan** — dalam waktu kurang dari satu jam. Bagi analis kebijakan perdagangan, ini mengurangi waktu analisis dari berminggu-minggu menjadi beberapa jam, sambil tetap menyediakan referensi ke teks asli untuk verifikasi.

---

*Dokumen ini ditulis dalam Bahasa Indonesia sebagai panduan memahami proyek. Untuk detail teknis lengkap, lihat `README.md`, `REPORT.md`, dan `REPORT_DRAFT.md`.*
