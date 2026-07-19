# Complaint Intelligence 🔍

AI-powered intelligence over the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).

**Live demo:** https://complaint-intelligence.streamlit.app

---

## What it does

- Ingests real consumer complaints from the CFPB public API
- Explores and validates the corpus (volume trends, category mix, data quality, duplicates) via a dedicated EDA notebook before any modeling begins
- Cleans, deduplicates, and prepares the corpus for modeling (leakage checks, transformations) in a dedicated data preparation notebook
- Embeds narratives using `sentence-transformers` for semantic search
- Answers natural-language questions over the complaint corpus via RAG
- Surfaces top products, issues, and companies in an interactive dashboard

---

## Architecture
```
CFPB API (official v1 endpoint)
↓  ingest.py
data/complaints.parquet
↓  01_copilot_eda.ipynb
EDA findings (data quality, category mix, chunking strategy)
↓  02_copilot_data_prep.ipynb
Cleaned, deduplicated, model-ready corpus
↓  03_copilot_rag.py
Embeddings (all-MiniLM-L6-v2)
↓
RAG query engine
↓
Streamlit dashboard
```

---

## Run locally

```bash
git clone https://github.com/mvillanueva00/ADS-599-Capstone-Project
cd ADS-599-Capstone-Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pull complaints from the CFPB API
python ingest.py --rows 10000

# Explore the data before modeling
jupyter notebook 01_copilot_eda.ipynb

# Clean and prepare the data for modeling
jupyter notebook 02_copilot_data_prep.ipynb

# Generate embeddings + test RAG
python 03_copilot_rag.py
streamlit run app.py
```

---

## Data source

CFPB Consumer Complaint Database — official v1 API:
https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/

Updated daily. 4M+ complaints. No API key required.

---

## Tech stack

| Layer | Tool |
|---|---|
| Data source | CFPB public API |
| Storage | Parquet |
| EDA | pandas, seaborn, plotly, statsmodels |
| Data preparation | pandas, scikit-learn |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| RAG | Custom retrieval pipeline (scikit-learn cosine similarity) |
| Dashboard | Streamlit |
| Language | Python 3.10+ |
