# Complaint Intelligence 🔍

Citation-backed question answering over the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/).

Ask a plain-English question about consumer complaints and get an answer grounded in — and traceable to — specific complaint narratives.

**Live demo:** https://complaint-intelligence.streamlit.app

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
↓ ingest.py
data/complaints.parquet
↓ 01_copilot_eda.ipynb
EDA findings (data quality, category mix, length distribution)
↓ 02_copilot_data_prep.ipynb
data/complaints_model_ready.parquet (cleaned, deduplicated)
↓ 04_copilot_modeling.py
Retrieval + classification evaluation → data/modeling_*.csv
↓ retrieval.py (shared retrieval layer)
↓
03_copilot_rag.py (CLI) · app.py (Streamlit artifact)
```

`retrieval.py` is the single source of truth for how complaints are embedded and
searched. Both the CLI and the app import it, so the deployed tool always runs
the configuration that was evaluated.

Answer generation runs through `gemini_client.py`, a shared layer used by
`03_copilot_rag.py`, `app.py`, and `test_pipeline.py` alike — so the model name,
retry policy, and citation-parsing logic can't drift between the CLI, the
deployed app, and the tests.

---

## Key results

Evaluated on 26 analyst-style queries, relevance proxied by CFPB issue label.

| Retrieval system | P@5 | P@10 | MRR |
|---|---|---|---|
| BM25 (lexical baseline) | 0.485 | 0.481 | 0.683 |
| TF-IDF cosine | 0.369 | 0.362 | 0.498 |
| Dense — MiniLM-L6 | 0.585 | 0.588 | 0.765 |
| **Dense — BGE-small** | **0.646** | 0.596 | **0.785** |
| Hybrid RRF (BM25 + BGE) | 0.600 | **0.600** | 0.775 |

Semantic retrieval significantly outperforms the lexical baseline on precision
(P@5 +0.162, p = 0.010; P@10 +0.115, p = 0.001; paired bootstrap over 26
queries). The two do not differ significantly on MRR, and hybrid fusion shows no
significant gain over dense retrieval alone — so the simpler dense-only
architecture is what ships.

> **Note:** these figures use relevance labels corrected by a pooled audit
> across all three retrievers (see `audit_labels.py`). The correction raised
> every system by ~0.054 and left the paired-test conclusions unchanged. Full
> numbers are in `data/modeling_retrieval.csv`.

On a secondary classification diagnostic the ordering reverses: sparse TF-IDF
features beat dense embeddings (macro-F1 0.754 vs 0.642), a gap that persisted
across three pooling strategies and reflects representational capacity rather
than a correctable implementation choice.

**Label leakage, caught and fixed.** An early version of the classification
pipeline reported a macro-F1 of 0.971 — implausibly high for an eleven-class
problem on noisy consumer text. The cause was `embedding_input`, a convenience
field built for the generation prompt that was prefixed with the complaint's
own product and issue labels (e.g. `"Product: Credit card | Issue: Fees or
interest."`). Embedding that field meant the model was reading the answer, not
the complaint. Every model now reads label-free `narrative_clean` text only,
and `assert_no_label_leakage()` in `04_copilot_modeling.py` raises an error if
a label-bearing column ever reaches the encoder again. All figures reported
above reflect the corrected, label-free run.

---

## Known limitations

- **Citation instability.** The generator tends to over-cite (citing all five
  retrieved complaints on nearly every claim) or, occasionally, under-cite
  (producing an answer with no citation at all). The app's citation-validation
  layer catches both failure modes rather than letting them pass silently, but
  citation behavior should be read as uncalibrated rather than reliably
  precise. Calibrating this at generation time — rather than only validating
  it after the fact — is the top item in future work.
- **Six-month analytical window (Jan–Jun 2026).** Scoped deliberately after
  EDA revealed a structural break in complaint volume and composition tied to
  a documented CFPB policy change (see `01_copilot_eda.ipynb`). Blending
  pre/post-policy data would conflate a regulatory artifact with a genuine
  behavioral trend.
- **Off-the-shelf embeddings, no domain adaptation.** Both embedding models
  used here are pretrained and not fine-tuned on complaint-specific language.
  The classification gap in favor of sparse features is consistent with the
  literature's predicted penalty under these conditions.
- **26-query evaluation set.** Substantially expanded from the 8 queries used
  early in the project, but still a modest sample; confidence intervals on
  retrieval metrics are correspondingly wide. Relevance is also proxied by
  CFPB issue label rather than direct human judgment.
- **Live app serves a stratified demo subset**, not the full corpus, due to
  hosting-tier memory/startup constraints. All performance numbers reported
  above were computed against the full 44,234-complaint corpus.

---

## Run locally

```bash
git clone https://github.com/mvillanueva00/ADS-599-Capstone-Project
cd ADS-599-Capstone-Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # full toolkit: modeling, notebooks, app
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
python ingest.py --rows 10000   # small sample for a quick local test run;
                                 # results reported above used the full
                                 # Jan-Jun 2026 extract, ~48,700 rows pre-cleaning
jupyter notebook 01_copilot_eda.ipynb
jupyter notebook 02_copilot_data_prep.ipynb
```

**Two requirements files.** `requirements.txt` is the minimal app runtime (what
Streamlit Cloud installs). `requirements-dev.txt` installs everything —
modeling, evaluation, and the notebooks — and is what you want for
reproducing results locally.

**Note on embeddings.** Cached vectors (`data/*.npy`) are gitignored — they
exceed GitHub's 100 MB file limit. The first run of any script regenerates and
caches them.

**Note on answer generation.** Without `GEMINI_API_KEY` the app performs
semantic search and displays source complaints but produces no written answer.
It deliberately does not fabricate a summary, since an ungrounded summary shown
beside real citations would be misleading.

**Notebooks vs. scripts.** `03_copilot_rag.ipynb` and `04_copilot_modeling.ipynb`
mirror `03_copilot_rag.py` and `04_copilot_modeling.py` respectively — same
logic, notebook form for interactive/exploratory use. The `.py` scripts are
what the app and test suite actually import and run.

---

## Testing

`test_pipeline.py` runs three tiers of automated checks:

- **Tier 1 — Setup and smoke.** No API key required. Catches broken imports,
  missing columns, and the label-leakage failure mode directly (checks that no
  narrative starts with its own product label).
- **Tier 2 — Retrieval quality regression.** Re-runs the evaluation queries and
  asserts precision stays above a floor, catching silent breakage (wrong text
  column, unnormalized vectors, a swapped model) that wouldn't raise an error
  but would quietly wreck results.
- **Tier 3 — Generation and faithfulness.** Requires `GEMINI_API_KEY`. Checks
  every citation resolves to a retrieved complaint, that the system refuses
  out-of-scope questions rather than hallucinating, and runs an LLM-as-judge
  faithfulness check in the style of RAGAS (Es et al., 2024).

```bash
python test_pipeline.py            # tiers 1-2, free, ~2 min
python test_pipeline.py --full     # adds tier 3, uses API quota, ~3 min
```

`diagnose_query.py` is a companion tool for interpreting a Tier 2 failure —
when a query scores zero, it distinguishes a genuine retrieval miss from a
label-mapping artifact (since relevance is proxied by CFPB issue label, and
consumers frequently mislabel their own complaints).

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
| `04_copilot_modeling.py` / `.ipynb` | Retrieval + classification evaluation, ablations, significance tests |
| `retrieval.py` | Shared retrieval layer (corpus, model, search) |
| `03_copilot_rag.py` / `.ipynb` | Citation-backed question answering, CLI |
| `gemini_client.py` | Shared answer-generation layer: model selection, retry/fallback logic, citation parsing |
| `app.py` | Streamlit dashboard |
| `audit_labels.py` | Pooled relevance-label audit across all three retrievers |
| `diagnose_query.py` | Diagnostic tool for inspecting zero-scoring evaluation queries |
| `test_pipeline.py` | Three-tier automated test suite (setup/smoke, retrieval regression, generation/faithfulness) |

---

## Team

Alexander Zhuk · Mark Villanueva · Michael Ha  
Master of Science in Applied Data Science  
Shiley Marcos School of Engineering / University of San Diego
