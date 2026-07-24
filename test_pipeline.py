"""
test_pipeline.py — Automated checks for Complaint Intelligence.

Three tiers, in increasing cost:

  TIER 1  Setup and smoke tests. No API key, no network beyond the embedding
          model download. Catches broken imports, missing files, wrong columns,
          and the label-leakage failure mode.

  TIER 2  Retrieval quality regression. Runs the evaluation queries and asserts
          precision stays above a floor. Catches silent breakage — wrong text
          column, unnormalised vectors, a swapped model — that would not raise
          an exception but would quietly wreck results.

  TIER 3  Generation and faithfulness. Requires GEMINI_API_KEY. Checks that
          answers are produced, that every claim carries a resolvable citation,
          that the system refuses questions the corpus cannot answer, and that
          claims are actually supported by the complaints they cite. This last
          check is an LLM-as-judge faithfulness score in the style of RAGAS
          (Es et al., 2024) and Zheng et al. (2023); like any automated judge it
          should be spot-checked by hand before being quoted in the report.

Usage:
    python test_pipeline.py            # tiers 1-2, free, ~2 min
    python test_pipeline.py --full     # adds tier 3, uses API quota, ~3 min
"""

import os
import re
import sys
import time
import argparse

import numpy as np

PASSED, FAILED, SKIPPED = [], [], []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append((name, detail))
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
    return bool(condition)


def skip(name, why):
    SKIPPED.append(name)
    print(f"  SKIP  {name} ({why})")


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 — Setup and smoke
# ══════════════════════════════════════════════════════════════════════════════
def tier1():
    print("\nTIER 1 — SETUP AND SMOKE")
    print("-" * 60)

    try:
        import retrieval
    except ImportError as e:
        check("retrieval.py imports", False,
              f"{e}. retrieval.py must sit next to app.py in the repo root.")
        sys.exit(1)
    check("retrieval.py imports", True)

    for mod in ["pandas", "numpy", "sklearn", "sentence_transformers",
                "rank_bm25", "streamlit"]:
        try:
            __import__(mod)
            check(f"import {mod}", True)
        except ImportError:
            check(f"import {mod}", False, "pip install -r requirements.txt")

    # Deprecated SDK conflicts with google-genai
    try:
        __import__("google.generativeai")
        check("no deprecated google-generativeai", False,
              "Run: pip uninstall google-generativeai")
    except ImportError:
        check("no deprecated google-generativeai", True)

    df = retrieval.load_corpus()
    check("corpus loads", len(df) > 0, f"{len(df)} rows")
    check("corpus is non-trivial", len(df) > 1000, f"only {len(df)} rows")

    for col in [retrieval.TEXT_COL, "product", "issue", "company"]:
        check(f"column present: {col}", col in df.columns)

    # A handful of blank narratives is tolerable; a large share means the
    # cleaning step in 02 is stripping real content.
    blank = df[retrieval.TEXT_COL].astype(str).str.strip().eq("").sum()
    check("blank narratives are negligible", blank / len(df) < 0.01,
          f"{blank} of {len(df)} narratives are blank ({blank/len(df):.2%}) — "
          f"check the cleaning step in 02_copilot_data_prep.ipynb")
    if blank:
        print(f"        note: {blank} blank narrative(s) present; they will "
              f"never be retrieved, but consider dropping them in data prep")

    # The leakage failure mode: text must not contain its own product label.
    sample = df.head(500)
    leaks = sum(1 for _, r in sample.iterrows()
                if str(r["product"]).lower()[:20]
                in str(r[retrieval.TEXT_COL]).lower()[:200])
    check("no label leakage in text column", leaks < 250,
          f"{leaks}/500 narratives start with their own product label — "
          f"TEXT_COL may be pointing at embedding_input")

    print("\n  Building/loading embeddings (first run takes a few minutes)...")
    model = retrieval.load_model()
    emb = retrieval.build_or_load_embeddings(df, model, progress=True)

    check("embedding count matches corpus", len(emb) == len(df),
          f"{len(emb)} vectors vs {len(df)} complaints")
    check("embeddings are unit-normalised",
          np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3),
          "cosine similarity assumes unit vectors; scores will be wrong")
    check("embeddings contain no NaN", not np.isnan(emb).any())

    r = retrieval.retrieve("debt collector keeps calling me", df, emb, model,
                           top_k=5)
    check("retrieve returns results", len(r) > 0)
    check("results are sorted by similarity",
          list(r["similarity"]) == sorted(r["similarity"], reverse=True))
    check("similarity scores are in cosine range",
          r["similarity"].between(-1.01, 1.01).all())

    rf = retrieval.retrieve("payment problem", df, emb, model, top_k=5,
                            product_filter="Mortgage")
    check("product filter works",
          len(rf) == 0 or set(rf["product"]) == {"Mortgage"},
          f"got {set(rf['product']) if len(rf) else 'nothing'}")

    ctx = retrieval.build_context(r)
    check("context is numbered from 1",
          re.findall(r"^\[(\d+)\]", ctx, re.M) == [str(i) for i in
                                                   range(1, len(r) + 1)])

    # Sanity: a debt-collection query should surface debt-collection complaints.
    dc = retrieval.retrieve("a debt collector is harassing me about a debt "
                            "I do not owe", df, emb, model, top_k=10)
    hit_rate = dc["product"].eq("Debt collection").mean()
    check("topical sanity: debt query returns debt complaints", hit_rate >= 0.5,
          f"only {hit_rate:.0%} of top-10 were Debt collection")

    return df, emb, model


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 — Retrieval quality regression
# ══════════════════════════════════════════════════════════════════════════════
EVAL_QUERIES = [
    ("A debt collector keeps calling me about a debt that is not mine",
     ["Attempts to collect debt not owed"]),
    ("The collection agency never sent me written validation of the debt",
     ["Written notification about debt"]),
    ("A collector threatened to sue me or garnish my wages",
     ["Took or threatened to take negative or legal action"]),
    ("My bank charged me overdraft fees I did not authorize",
     ["Problem caused by your funds being low",
      "Problem with a lender or other company charging your account",
      "Managing an account"]),
    ("There are fraudulent charges on my credit card",
     ["Problem with a purchase shown on your statement"]),
    ("My mortgage payment was not applied correctly",
     ["Trouble during payment process"]),
    ("I am behind on my mortgage and cannot get a loan modification",
     ["Struggling to pay mortgage"]),
    ("Someone opened an account in my name and my credit report is wrong",
     ["Incorrect information on your report"]),
    ("A company pulled my credit report without my permission",
     ["Improper use of your report"]),
    ("My car was repossessed without proper notice", ["Repossession"]),
]

# Floor, not the expected value. BGE-small scored P@5 = 0.592 in the full
# 26-query evaluation and BM25 scored 0.431, so anything below 0.40 means
# something is structurally broken rather than merely slightly worse.
P5_FLOOR = 0.40


def tier2(df, emb, model):
    print("\nTIER 2 — RETRIEVAL QUALITY REGRESSION")
    print("-" * 60)
    import retrieval

    p5s = []
    for q, issues in EVAL_QUERIES:
        rel = df["issue"].isin(issues).values
        if rel.sum() == 0:
            print(f"  note: no relevant docs for {q[:45]} — skipping")
            continue
        r = retrieval.retrieve(q, df, emb, model, top_k=5)
        idx = df.index.get_indexer(r.index)
        p5s.append(rel[idx].sum() / 5)

    mean_p5 = float(np.mean(p5s)) if p5s else 0.0
    print(f"\n  Mean P@5 over {len(p5s)} queries: {mean_p5:.3f}")
    check(f"P@5 above floor ({P5_FLOOR})", mean_p5 >= P5_FLOOR,
          f"got {mean_p5:.3f} — check TEXT_COL, the embedding model, and that "
          f"the query prefix is applied")
    zeros = sum(1 for p in p5s if p == 0)
    check("at most one query scores zero", zeros <= 1,
          f"{zeros} queries returned nothing relevant in their top 5 — with "
          f"label-proxied relevance an occasional zero is expected, but "
          f"several suggests a real problem")


# ══════════════════════════════════════════════════════════════════════════════
# TIER 3 — Generation and faithfulness
# ══════════════════════════════════════════════════════════════════════════════
FAITHFULNESS_QUESTIONS = [
    "What problems are consumers reporting with debt collectors?",
    "Why are consumers disputing credit card charges?",
    "What goes wrong during the mortgage payment process?",
    "What issues do consumers report with credit reporting errors?",
    "What problems do consumers have with auto loans?",
]

# Deliberately outside the corpus. A grounded system should say so rather than
# invent an answer; this is the intrinsic-hallucination case from Ji et al. (2023).
OUT_OF_SCOPE = "What do these complaints say about the weather in Antarctica?"


def judge_supported(claim: str, evidence: str) -> bool | None:
    """Ask Gemini whether a single claim is supported by the cited complaint."""
    import gemini_client
    try:
        out = gemini_client.generate(
            prompt=(f"Complaint:\n{evidence[:2000]}\n\nClaim:\n{claim}\n\n"
                    f"Is the claim supported by the complaint? Reply with "
                    f"exactly one word: SUPPORTED or NOT_SUPPORTED."),
            system_instruction="You are a careful fact-checker.",
            max_output_tokens=1024,
            temperature=0.0,
        ).strip().upper()
    except Exception as e:
        print(f"      judge call failed ({e}) — counting as unjudged")
        return None
    if "NOT_SUPPORTED" in out:
        return False
    if "SUPPORTED" in out:
        return True
    return None


def tier3(df, emb, model):
    print("\nTIER 3 — GENERATION AND FAITHFULNESS")
    print("-" * 60)

    # 03_copilot_rag.py starts with a digit, so it must be imported by path.
    import importlib.util
    spec = importlib.util.spec_from_file_location("rag", "03_copilot_rag.py")
    rag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag)

    total_claims, supported, unknown = 0, 0, 0

    for q in FAITHFULNESS_QUESTIONS:
        print(f"\n  Q: {q[:60]}")
        result = rag.answer(q, df, emb, model, top_k=5)
        text = result["answer"]

        check("  answer generated", bool(text.strip()))
        check("  no invalid citations", not result["invalid_citations"],
              f"cited {result['invalid_citations']} but only 5 were retrieved")
        check("  at least one citation used", bool(result["citations_used"]))

        # Faithfulness: each cited sentence checked against the complaint it cites
        sources = result["sources"]
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            import gemini_client as _gc
            cites = _gc.extract_citations(sentence)
            cites = [c for c in cites if 1 <= c <= len(sources)]
            if not cites:
                continue
            evidence = "\n\n".join(
                str(sources[c - 1][retrieval_text_col()]) for c in cites)
            verdict = judge_supported(sentence, evidence)
            total_claims += 1
            if verdict is True:
                supported += 1
            elif verdict is None:
                unknown += 1
            time.sleep(4)          # free-tier Flash is ~10 requests/minute

    if total_claims:
        score = supported / total_claims
        print(f"\n  Faithfulness: {supported}/{total_claims} claims supported "
              f"({score:.0%}), {unknown} unjudged")
        check("faithfulness >= 80%", score >= 0.80,
              f"got {score:.0%} — inspect the failing claims by hand before "
              f"reporting this number")
    else:
        skip("faithfulness score", "no cited claims found to judge")

    print("\n  Out-of-scope refusal test")
    result = rag.answer(OUT_OF_SCOPE, df, emb, model, top_k=5)
    lowered = result["answer"].lower()
    refused = any(p in lowered for p in
                  ["do not", "don't", "not address", "no information",
                   "not mention", "cannot", "unrelated", "not cover",
                   "nothing in", "does not"])
    check("refuses out-of-scope question", refused,
          "the answer did not signal that the complaints lack this "
          "information — possible hallucination")


def retrieval_text_col():
    import retrieval
    return retrieval.TEXT_COL


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="include tier 3 (needs GEMINI_API_KEY, uses quota)")
    args = ap.parse_args()

    print("=" * 60)
    print("COMPLAINT INTELLIGENCE — PIPELINE TESTS")
    print("=" * 60)

    df, emb, model = tier1()
    tier2(df, emb, model)

    if args.full:
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            tier3(df, emb, model)
        else:
            print("\nTIER 3 — SKIPPED: set GEMINI_API_KEY to run generation "
                  "and faithfulness tests")
    else:
        print("\nTIER 3 — not requested (use --full to include)")

    print("\n" + "=" * 60)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed, {len(SKIPPED)} skipped")
    if FAILED:
        print("\nFailures:")
        for name, detail in FAILED:
            print(f"  - {name}")
            if detail:
                print(f"      {detail}")
    print("=" * 60)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
