"""
diagnose_query.py — Show per-query retrieval performance and inspect failures.

Run this when test_pipeline.py reports a query scoring zero, to tell apart two
very different situations:

  (a) genuine retrieval failure — the top results are off-topic
  (b) label-mapping artifact — the results are topically right but carry a
      different CFPB issue label than the one the test treats as "relevant"

(b) is not a bug. Relevance here is proxied by issue label, and consumers
mislabel their own complaints (Bastani et al., 2019), so a perfectly good
result can score zero. Look at what actually came back before changing anything.

Usage:
    python diagnose_query.py
"""

import retrieval
from test_pipeline import EVAL_QUERIES

print("Loading corpus and embeddings (cached)...")
df = retrieval.load_corpus()
model = retrieval.load_model()
emb = retrieval.build_or_load_embeddings(df, model, progress=False)

print(f"\n{'P@5':>6}  {'relevant in corpus':>18}  query")
print("-" * 92)

failures = []
for q, issues in EVAL_QUERIES:
    rel = df["issue"].isin(issues).values
    r = retrieval.retrieve(q, df, emb, model, top_k=5)
    idx = df.index.get_indexer(r.index)
    p5 = rel[idx].sum() / 5
    print(f"{p5:>6.2f}  {rel.sum():>18,}  {q[:60]}")
    if p5 == 0:
        failures.append((q, issues, r, rel.sum()))

if not failures:
    print("\nNo query scored zero.")
    raise SystemExit

for q, issues, r, n_rel in failures:
    print("\n" + "=" * 92)
    print(f"ZERO-SCORING QUERY: {q}")
    print(f"Labels counted as relevant: {issues}")
    print(f"Complaints carrying those labels: {n_rel:,}")
    if n_rel == 0:
        print("\n  -> The label does not exist in this corpus. The test is "
              "checking against a category that was never collected; fix the "
              "label list, not the retriever.")
        continue
    print("\nWhat the retriever actually returned:\n")
    for i, (_, row) in enumerate(r.iterrows(), 1):
        print(f"  [{i}] sim={row['similarity']:.3f}  "
              f"{row['product']} | {row['issue']}")
        print(f"      {str(row[retrieval.TEXT_COL])[:220]}...\n")
    print("  Judge for yourself: are these on-topic? If yes, this is a label "
          "artifact and the retriever is fine. If no, it is a real miss worth "
          "reporting as a limitation.")

# What labels DO the top results carry? Often reveals the right mapping.
print("\n" + "=" * 92)
print("Issue labels appearing in the top 10 for each failing query:")
for q, issues, _, _ in failures:
    r10 = retrieval.retrieve(q, df, emb, model, top_k=10)
    print(f"\n  {q[:70]}")
    for lbl, cnt in r10["issue"].value_counts().items():
        marker = " <- already counted" if lbl in issues else ""
        print(f"      {cnt:>2}  {lbl}{marker}")
