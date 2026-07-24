"""
make_demo_subset.py — Build a small, committable corpus for Streamlit Cloud.

WHY
    The full index is 44,234 complaints and its embedding cache is ~65 MB, which
    is gitignored. On Streamlit Cloud that means the app tries to encode the
    whole corpus on first boot — roughly 11 minutes on a laptop, and longer on
    Cloud's shared CPU, where it will usually hit the boot timeout or exhaust
    the ~1 GB memory limit first. Worse, the container resets, so it rebuilds
    every time.

    A 10,000-complaint subset with its embeddings committed alongside is about
    23 MB total. Cloud loads it instantly with no build step and no memory
    pressure. All reported evaluation numbers still come from the full corpus;
    this affects only what the public demo searches over.

SAMPLING
    Stratified by product so every category stays represented in proportion,
    with a fixed seed so the subset is reproducible. Complaints shorter than
    20 words are excluded — they make poor demo citations.

Usage:
    python make_demo_subset.py
    python make_demo_subset.py --rows 8000
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
FULL_CORPUS = DATA_DIR / "complaints_model_ready.parquet"
DEMO_CORPUS = DATA_DIR / "complaints_demo.parquet"
DEMO_EMB = DATA_DIR / "embeddings_demo.npy"

TEXT_COL = "narrative_clean"
LABEL_COL = "product"
MIN_WORDS = 20
RANDOM_STATE = 42

# Only the columns the app displays, to keep the file small.
KEEP_COLS = ["complaint_id", "date_received", "product", "sub_product", "issue",
             "sub_issue", "company", "state", "company_response", "timely",
             TEXT_COL]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000)
    args = ap.parse_args()

    if not FULL_CORPUS.exists():
        raise SystemExit(f"{FULL_CORPUS} not found — run the data prep notebook.")

    print("Loading full corpus...")
    df = pd.read_parquet(FULL_CORPUS)
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    print(f"  {len(df):,} complaints")

    # Drop very short narratives — they retrieve poorly and read badly as demo
    # citations.
    long_enough = df[TEXT_COL].astype(str).str.split().str.len() >= MIN_WORDS
    df = df[long_enough].reset_index(drop=True)
    print(f"  {len(df):,} after dropping narratives under {MIN_WORDS} words")

    n = min(args.rows, len(df))
    frac = n / len(df)

    # Stratified by product: every category keeps its share, and small
    # categories keep at least a handful of examples.
    parts = []
    for product, group in df.groupby(LABEL_COL, sort=False):
        take = max(int(round(len(group) * frac)), min(25, len(group)))
        parts.append(group.sample(n=min(take, len(group)),
                                  random_state=RANDOM_STATE))
    demo = (pd.concat(parts)
              .sample(frac=1.0, random_state=RANDOM_STATE)   # shuffle
              .reset_index(drop=True))

    cols = [c for c in KEEP_COLS if c in demo.columns]
    demo = demo[cols]

    print(f"\nDemo subset: {len(demo):,} complaints, "
          f"{demo[LABEL_COL].nunique()} products")
    print(demo[LABEL_COL].value_counts().to_string())

    demo.to_parquet(DEMO_CORPUS, index=False)
    size_mb = DEMO_CORPUS.stat().st_size / 1e6
    print(f"\nWrote {DEMO_CORPUS} ({size_mb:.1f} MB)")

    print("\nEncoding demo embeddings...")
    import retrieval
    model = retrieval.load_model()
    emb = model.encode(demo[TEXT_COL].astype(str).tolist(), batch_size=128,
                       show_progress_bar=True, convert_to_numpy=True,
                       normalize_embeddings=True)
    np.save(DEMO_EMB, emb)
    emb_mb = DEMO_EMB.stat().st_size / 1e6
    print(f"Wrote {DEMO_EMB} {emb.shape} ({emb_mb:.1f} MB)")

    total = size_mb + emb_mb
    print(f"\nTotal to commit: {total:.1f} MB")
    if total > 90:
        print("  WARNING: close to GitHub's 100 MB per-file limit — "
              "re-run with a smaller --rows")

    print("\nNext:")
    print("  git add -f data/complaints_demo.parquet data/embeddings_demo.npy")
    print("  git commit -m 'Add demo corpus for Streamlit Cloud'")
    print("\nThe app uses these automatically when the full embedding cache is "
          "absent, so local runs still use all 44,234 complaints.")


if __name__ == "__main__":
    main()
