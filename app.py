"""
app.py — Streamlit dashboard for Complaint Copilot.
Run: streamlit run app.py
Requires: data/complaints.parquet (from ingest.py)
          data/embeddings.npy     (from 02_copilot_rag.py, or generated on first load)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Complaint Copilot",
    page_icon="🔍",
    layout="wide"
)

DATA_DIR = Path("data")
NARRATIVE_COL = "consumer_complaint_narrative"

# ── Load data & embeddings ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_DIR / "complaints.parquet")
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce", utc=True)
    return df

@st.cache_data
def load_embeddings(_df):
    emb_path = DATA_DIR / "embeddings.npy"
    if emb_path.exists():
        return np.load(emb_path)
    st.info("Generating embeddings on first load (~1 min)...")
    model = load_model()
    texts = _df[NARRATIVE_COL].fillna("").tolist()
    embs = model.encode(texts, batch_size=128, show_progress_bar=False,
                        convert_to_numpy=True)
    np.save(emb_path, embs)
    return embs

model      = load_model()
df         = load_data()
embeddings = load_embeddings(df)

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = 5, product_filter=None) -> pd.DataFrame:
    q_emb = model.encode([query], convert_to_numpy=True)
    scores = embeddings @ q_emb.T / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) *
        np.linalg.norm(q_emb) + 1e-9
    )
    scores = scores.flatten()
    if product_filter and product_filter != "All":
        mask = (df["product"] == product_filter).values
        scores = np.where(mask, scores, -1)
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = df.iloc[top_idx].copy()
    results["similarity"] = scores[top_idx]
    return results[results["similarity"] > 0]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Complaint Copilot")
    st.caption(f"**{len(df):,}** complaints loaded")
    st.divider()
    product_filter = st.selectbox(
        "Filter by product",
        ["All"] + sorted(df["product"].dropna().unique().tolist())
    )
    top_k = st.slider("Results to retrieve", 3, 10, 5)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🔍 Complaint Copilot")
st.caption("Semantic search over the CFPB Consumer Complaint Database")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Complaints", f"{len(df):,}")
col2.metric("Products", df["product"].nunique())
col3.metric("Companies", df["company"].nunique())
col4.metric("States", df["state"].nunique())

st.divider()

# ── Query box ─────────────────────────────────────────────────────────────────
st.subheader("Ask anything about the complaints")

examples = [
    "What are the most common issues with student loans?",
    "Problems with debt collectors harassing consumers",
    "Credit card billing disputes and unauthorized charges",
    "Mortgage companies failing to process payments correctly",
    "What companies have the most complaints about fees?",
]

cols = st.columns(len(examples))
query = ""
for i, (col, ex) in enumerate(zip(cols, examples)):
    if col.button(ex[:35] + "…", key=f"ex_{i}"):
        query = ex

query = st.text_input("Or type your own question:", value=query, placeholder="e.g. What are the biggest issues with credit reporting?")

if query:
    with st.spinner("Searching complaints..."):
        retrieved = retrieve(query, top_k=top_k, product_filter=product_filter)

    if len(retrieved) == 0:
        st.warning("No matching complaints found. Try rephrasing your question.")
    else:
        top_products = ", ".join(retrieved["product"].value_counts().head(3).index.tolist())
        st.markdown("### 💡 Summary")
        st.info(f"Found {len(retrieved)} complaints most similar to your question. "
                f"They mostly involve: {top_products}. See the matching complaints below.")

        st.markdown(f"### 📄 Source Complaints ({len(retrieved)} retrieved)")
        for i, (_, row) in enumerate(retrieved.iterrows()):
            with st.expander(f"[{i+1}] {row['company']} — {row['product']} — {row['issue']} (similarity: {row['similarity']:.2f})"):
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"**State:** {row.get('state','N/A')}")
                col2.markdown(f"**Response:** {row.get('company_response','N/A')}")
                col3.markdown(f"**Timely:** {row.get('timely','N/A')}")
                st.markdown("**Narrative:**")
                st.write(str(row[NARRATIVE_COL])[:1000])

st.divider()

# ── Trends ────────────────────────────────────────────────────────────────────
st.subheader("📊 Complaint Trends")

tcol1, tcol2 = st.columns(2)

with tcol1:
    st.markdown("**Top Products**")
    top_products = df["product"].value_counts().head(8)
    st.bar_chart(top_products)

with tcol2:
    st.markdown("**Top Issues**")
    top_issues = df["issue"].value_counts().head(8)
    st.bar_chart(top_issues)

st.markdown("**Top Companies by Complaint Volume**")
top_companies = df["company"].value_counts().head(10)
st.bar_chart(top_companies)

if "date_received" in df.columns:
    st.markdown("**Complaints Over Time**")
    timeline = df.set_index("date_received").resample("W").size()
    st.line_chart(timeline)
