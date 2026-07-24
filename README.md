# Complaint Intelligence 🔍

Citation-backed question answering over the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).

Ask a plain-English question about consumer complaints and get an answer grounded in — and traceable to — specific complaint narratives.

---

## What it does

- Ingests real consumer complaints from the CFPB public API
- Explores and validates the corpus (volume trends, category mix, data quality, duplicates) in a dedicated EDA notebook
- Cleans, deduplicates, and prepares the corpus for modeling
- Compares retrieval methods (BM25, TF-IDF, two embedding models, hybrid fusion) with significance testing
- Answers natural-language questions with bracketed citations pointing at the complaints that support each claim
- Flags any citation that falls outside the retrieved evidence set

---

## Pipeline

```
CFPB API (official v1 endpoint)
  ↓  ingest.py
data/complaints.parquet
  ↓  01_copilot_eda.ipynb
EDA findings (data quality, category mix, length distribution)
  ↓  02_copilot_data_prep.ipynb
data/complaints_model_ready.parquet   (cleaned, deduplicated)
  ↓  04_copilot_modeling.py
Retrieval + classification evaluation → data/modeling_*.csv
  ↓  retrieval.py                      (shared retrieval layer)
  ↓
03_copilot_rag.py  (CLI)   ·   app.py  (Streamlit artifact)
```

`retrieval.py` is the single source of truth for how complaints are embedded and
searched. Both the CLI and the app import it, so the deployed tool always runs
the configuration that was evaluated.

---

## Key results

Evaluated on 26 analyst-style queries, relevance proxied by CFPB issue label.

| Retrieval system | P@5 | P@10 | MRR |
|---|---|---|---|
| BM25 (lexical baseline) | 0.431 | 0.435 | 0.613 |
| TF-IDF cosine | 0.315 | 0.312 | 0.457 |
| Dense — MiniLM-L6 | 0.531 | 0.542 | 0.711 |
| **Dense — BGE-small** | **0.592** | 0.550 | 0.715 |
| Hybrid RRF (BM25 + BGE) | 0.554 | **0.558** | **0.741** |

Semantic retrieval significantly outperforms the lexical baseline on precision
(P@5 +0.162, p = 0.010; P@10 +0.115, p = 0.001; paired bootstrap over 26
queries). The two do not differ significantly on MRR, and hybrid fusion shows no
significant gain over dense retrieval alone — so the simpler dense-only
architecture is what ships.

On a secondary classification diagnostic the ordering reverses: sparse TF-IDF
features beat dense embeddings (macro-F1 0.754 vs 0.642), a gap that persisted
across three pooling strategies and reflects representational capacity rather
than a correctable implementation choice.

---

## Run locally

```bash
git clone https://github.com/mvillanueva00/ADS-599-Capstone-Project
cd ADS-599-Capstone-Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`data/complaints_model_ready.parquet` is committed, so you can skip ingest and
the notebooks and go straight to modeling:

```bash
# Modeling and evaluation (~15-25 min first run, then embeddings are cached)
python 04_copilot_modeling.py

# Citation-backed answers from the command line
export GEMINI_API_KEY=...
python 03_copilot_rag.py

# Interactive dashboard
streamlit run app.py
```

To rebuild the corpus from scratch:

```bash
python ingest.py --rows 10000
jupyter notebook 01_copilot_eda.ipynb
jupyter notebook 02_copilot_data_prep.ipynb
```

**Note on embeddings.** Cached vectors (`data/*.npy`) are gitignored — they
exceed GitHub's 100 MB file limit. The first run of any script regenerates and
caches them.

**Note on answer generation.** Without `GEMINI_API_KEY` the app performs
semantic search and displays source complaints but produces no written answer.
It deliberately does not fabricate a summary, since an ungrounded summary shown
beside real citations would be misleading.

---

## Data source

CFPB Consumer Complaint Database — official v1 API:
https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/

Updated daily. No API key required.

---

## Tech stack

| Layer | Tool |
|---|---|
| Data source | CFPB public API |
| Storage | Parquet |
| EDA | pandas, seaborn, plotly, statsmodels, wordcloud |
| Data preparation | pandas |
| Embeddings | sentence-transformers (`BAAI/bge-small-en-v1.5`) |
| Lexical baselines | rank-bm25, scikit-learn TF-IDF |
| Evaluation | scikit-learn, paired bootstrap significance tests |
| Answer generation | Google Gemini (`google-genai`) |
| Dashboard | Streamlit |
| Language | Python 3.10+ |

---

## Repository layout

| File | Role |
|---|---|
| `ingest.py` | Pull complaints from the CFPB API |
| `01_copilot_eda.ipynb` | Exploratory analysis and data-quality checks |
| `02_copilot_data_prep.ipynb` | Cleaning, deduplication, model-ready corpus |
| `04_copilot_modeling.py` | Retrieval + classification evaluation, ablations, significance tests |
| `retrieval.py` | Shared retrieval layer (corpus, model, search) |
| `03_copilot_rag.py` | Citation-backed question answering, CLI |
| `app.py` | Streamlit dashboard |
