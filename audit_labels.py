"""
audit_labels.py — Audit the relevance labels used in the retrieval evaluation.

WHY THIS EXISTS
    Relevance in this project is proxied by CFPB issue label: a complaint counts
    as relevant to a query if it carries one of a hand-picked set of labels. If
    that set is wrong, every retriever is penalised for returning correct
    results. This was found in exactly that way — the overdraft-fee query
    omitted "Problem caused by your funds being low", the label the CFPB
    actually uses for overdraft complaints, so all five correct results scored
    zero.

THE TRAP THIS AVOIDS
    The obvious fix — look at what the dense retriever returns and add those
    labels — grades the retriever on its own output and biases the evaluation
    toward whichever system you inspected. Standard practice in information
    retrieval (TREC-style pooling) is to pool the top results from EVERY system
    under comparison, then judge that pool without knowing which system
    contributed what. This script builds that pool.

HOW TO USE IT
    1. Run it. For each query it shows the labels appearing in the pooled top-10
       of BM25, TF-IDF, and dense retrieval, with per-system counts.
    2. For each candidate label, decide from the CFPB taxonomy whether it
       genuinely describes the query — not whether including it would raise
       your score.
    3. Apply the same revised label set to 04_copilot_modeling.py and
       test_pipeline.py, then re-run the full evaluation.

    Labels contributed by all three systems are safe to consider. A label found
    only by one system deserves more scrutiny, since adding it will
    disproportionately favour that system.

Usage:
    python audit_labels.py
    python audit_labels.py --query 3        # audit one query in detail
"""

import re
import argparse
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

import retrieval

POOL_DEPTH = 10

# The evaluation set as it currently stands in 04_copilot_modeling.py.
EVAL_QUERIES = [
    ("A debt collector keeps calling me about a debt that is not mine",
     ["Attempts to collect debt not owed"]),
    ("I am being chased for a debt I already paid off",
     ["Attempts to collect debt not owed", "Written notification about debt"]),
    ("The collection agency never sent me written validation of the debt",
     ["Written notification about debt"]),
    ("A collector threatened to sue me or garnish my wages",
     ["Took or threatened to take negative or legal action"]),
    ("Debt collectors calling my workplace and my relatives",
     ["Communication tactics",
      "Threatened to contact someone or share information improperly"]),
    ("The collector lied about who they were and what I owed",
     ["False statements or representation"]),
    ("My bank charged me overdraft fees I did not authorize",
     ["Problem with a lender or other company charging your account",
      "Managing an account"]),
    ("The bank closed my account without warning or explanation",
     ["Closing an account", "Closing your account"]),
    ("My account was frozen and I cannot access my own money",
     ["Managing an account", "Problem caused by your funds being low"]),
    ("There are fraudulent charges on my credit card",
     ["Problem with a purchase shown on your statement"]),
    ("I was charged fees and interest I never agreed to",
     ["Fees or interest", "Charged fees or interest you didn't expect"]),
    ("My credit card application was denied without a clear reason",
     ["Getting a credit card"]),
    ("A merchant refuses to refund me and the card issuer denied my dispute",
     ["Problem with a purchase shown on your statement",
      "Problem with a company's investigation into an existing problem"]),
    ("My mortgage payment was not applied correctly",
     ["Trouble during payment process"]),
    ("I am behind on my mortgage and cannot get a loan modification",
     ["Struggling to pay mortgage"]),
    ("Problems with the closing process on my home loan",
     ["Closing on a mortgage",
      "Applying for a mortgage or refinancing an existing mortgage"]),
    ("Problems getting my student loan payments counted for forgiveness",
     ["Dealing with your lender or servicer"]),
    ("I cannot afford my monthly loan payments and need help",
     ["Struggling to repay your loan", "Struggling to pay your loan"]),
    ("Someone opened an account in my name and my credit report is wrong",
     ["Incorrect information on your report"]),
    ("The credit bureau ignored my dispute about an error on my report",
     ["Problem with a company's investigation into an existing problem"]),
    ("A company pulled my credit report without my permission",
     ["Improper use of your report"]),
    ("My money transfer never arrived and the company will not refund me",
     ["Fraud or scam", "Money was not available when promised"]),
    ("I cannot access the funds in my digital wallet app",
     ["Trouble accessing funds in your mobile or digital wallet"]),
    ("My car was repossessed without proper notice", ["Repossession"]),
    ("The dealership misled me about my auto loan terms",
     ["Getting a loan or lease", "Managing the loan or lease"]),
    ("There were unauthorized transactions on my prepaid card",
     ["Unauthorized transactions or other transaction problem",
      "Other transaction problem"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=int, default=None,
                    help="1-based index of a single query to inspect in detail")
    args = ap.parse_args()

    print("Loading corpus and embeddings (cached)...")
    df = retrieval.load_corpus()
    model = retrieval.load_model()
    emb = retrieval.build_or_load_embeddings(df, model, progress=False)

    print("Building lexical indexes for pooling...")
    texts = df[retrieval.TEXT_COL].astype(str)
    bm25 = BM25Okapi([re.findall(r"[a-z0-9']+", t.lower()) for t in texts])
    tfv = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=2)
    X = tfv.fit_transform(texts)

    def bm25_top(q):
        return np.argsort(bm25.get_scores(
            re.findall(r"[a-z0-9']+", q.lower())))[::-1][:POOL_DEPTH]

    def tfidf_top(q):
        s = (X @ tfv.transform([q]).T).toarray().ravel()
        return np.argsort(s)[::-1][:POOL_DEPTH]

    def dense_top(q):
        r = retrieval.retrieve(q, df, emb, model, top_k=POOL_DEPTH)
        return df.index.get_indexer(r.index)

    queries = (EVAL_QUERIES if args.query is None
               else [EVAL_QUERIES[args.query - 1]])
    offset = 1 if args.query is None else args.query

    suspect = []
    for qi, (q, labels) in enumerate(queries, offset):
        pools = {"BM25": bm25_top(q), "TF-IDF": tfidf_top(q),
                 "Dense": dense_top(q)}

        # Current score, for context
        rel = df["issue"].isin(labels).values
        p5 = rel[dense_top(q)[:5]].sum() / 5

        per_label = {}
        for sysname, idxs in pools.items():
            for lbl, cnt in Counter(df.iloc[idxs]["issue"]).items():
                per_label.setdefault(lbl, Counter())[sysname] += cnt

        missing = {l: c for l, c in per_label.items() if l not in labels}
        flag = ""
        if p5 <= 0.4 and missing:
            flag = "   <-- REVIEW"
            suspect.append(qi)

        print("\n" + "=" * 88)
        print(f"[{qi}] P@5={p5:.2f}  {q}{flag}")
        print(f"     currently counted relevant: {labels}")

        if not missing:
            print("     no unlisted labels in the pool — mapping looks complete")
            continue

        print("\n     labels in the pool that are NOT currently counted:")
        for lbl, c in sorted(missing.items(),
                             key=lambda kv: -sum(kv[1].values())):
            systems = ", ".join(f"{s}:{n}" for s, n in sorted(c.items()))
            consensus = len(c)
            marker = ("  [all 3 systems]" if consensus == 3
                      else "  [1 system only — scrutinise]" if consensus == 1
                      else "")
            print(f"       {sum(c.values()):>2}  {lbl}  ({systems}){marker}")

    print("\n" + "=" * 88)
    if suspect:
        print(f"Queries worth reviewing: {suspect}")
        print("Inspect one closely with:  python audit_labels.py --query N")
    else:
        print("No query looks mislabelled.")
    print("\nDecide each label from the CFPB taxonomy, not from which choice "
          "raises the score. Then apply the same revised set to BOTH "
          "04_copilot_modeling.py and test_pipeline.py and re-run.")


if __name__ == "__main__":
    main()
