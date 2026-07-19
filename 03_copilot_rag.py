"""
02_copilot_rag.py — RAG query engine over CFPB complaints.
Run after ingest.py. 
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading complaints...")
df = pd.read_parquet(DATA_DIR / "complaints.parquet")
print(f"  {len(df):,} rows, columns: {df.columns.tolist()}")

NARRATIVE_COL = "consumer_complaint_narrative"
assert NARRATIVE_COL in df.columns, f"Missing column: {NARRATIVE_COL}"

texts = df[NARRATIVE_COL].fillna("").tolist()
print(f"  Narratives ready: {len(texts):,}")

# ── Embeddings ────────────────────────────────────────────────────────────────
print("\nGenerating embeddings (first run downloads ~90MB model)...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, batch_size=128, show_progress_bar=True,
                          convert_to_numpy=True)
np.save(DATA_DIR / "embeddings.npy", embeddings)
print(f"  Embeddings shape: {embeddings.shape}")

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int = 5) -> pd.DataFrame:
    q_emb = model.encode([query], convert_to_numpy=True)
    scores = embeddings @ q_emb.T / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) *
        np.linalg.norm(q_emb) + 1e-9
    )
    scores = scores.flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = df.iloc[top_idx].copy()
    results["similarity"] = scores[top_idx]
    return results

# ── Answer ────────────────────────────────────────────────────────────────────
USE_LLM = bool(os.environ.get("ANTHROPIC_API_KEY"))
print(f"\nLLM mode: {'ON (Anthropic)' if USE_LLM else 'OFF (keyword fallback)'}")

def answer(query: str, top_k: int = 5) -> dict:
    retrieved = retrieve(query, top_k=top_k)
    context = "\n\n".join([
        f"[{i+1}] Product: {row['product']} | Issue: {row['issue']}\n{row[NARRATIVE_COL][:500]}"
        for i, (_, row) in enumerate(retrieved.iterrows())
    ])

    if USE_LLM:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    f"You are a consumer complaint analyst. Answer the question using ONLY "
                    f"the complaints below. Be concise and cite complaint numbers.\n\n"
                    f"Complaints:\n{context}\n\nQuestion: {query}"
                )
            }]
        )
        answer_text = resp.content[0].text
    else:
        # Keyword fallback
        answer_text = (
            f"Top {top_k} complaints retrieved for: '{query}'\n"
            f"Set ANTHROPIC_API_KEY for LLM-synthesized answers.\n\n"
            + "\n".join([f"[{i+1}] {row['product']} — {row['issue']}"
                         for i, (_, row) in enumerate(retrieved.iterrows())])
        )

    return {
        "query": query,
        "answer": answer_text,
        "sources": retrieved[["complaint_id", "product", "issue", "company",
                               "state", "similarity", NARRATIVE_COL]].to_dict("records")
    }

# ── Test queries ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
test_queries = [
    "What are the most common issues with student loans?",
    "Problems with debt collectors harassing consumers",
    "Credit card billing disputes and unauthorized charges",
]

for q in test_queries:
    print(f"\nQ: {q}")
    result = answer(q, top_k=3)
    print(f"A: {result['answer'][:300]}")
    print(f"   Sources: {[s['company'] for s in result['sources']]}")

print("\n✅ RAG pipeline working. Run streamlit run app.py for the dashboard.")
