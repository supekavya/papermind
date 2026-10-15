"""
Paper Mind - Retrieval Engine (Step 4)
---------------------------------------
Implements Semantic, Keyword (BM25), and Hybrid Retrieval via Reciprocal Rank Fusion (RRF).
Includes deterministic multi-level tie-breaking and explicit tie-break reporting.
All algorithms are implemented in plain, traceable Python without external orchestration frameworks.
"""

import math
import re
from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np
from embeddings import db, embed_query


class BM25:
    """
    Pure Python BM25 implementation for transparent, explainable lexical scoring.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Counter = Counter()
        self.term_freqs: List[Counter] = []
        self.corpus_size: int = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            return
        self.doc_len = []
        self.term_freqs = []
        self.doc_freqs = Counter()

        total_words = 0
        for doc in corpus:
            tokens = self.tokenize(doc)
            self.doc_len.append(len(tokens))
            total_words += len(tokens)
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for term in tf.keys():
                self.doc_freqs[term] += 1

        self.avg_doc_len = total_words / self.corpus_size if self.corpus_size > 0 else 0.0

    def score(self, query: str) -> List[float]:
        q_tokens = self.tokenize(query)
        scores = [0.0] * self.corpus_size
        if not q_tokens or self.corpus_size == 0:
            return scores

        for term in q_tokens:
            if term not in self.doc_freqs:
                continue
            df = self.doc_freqs[term]
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

            for idx in range(self.corpus_size):
                tf = self.term_freqs[idx].get(term, 0)
                if tf == 0:
                    continue
                d_len = self.doc_len[idx]
                denom_len = (d_len / self.avg_doc_len) if self.avg_doc_len > 0 else 1.0
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * denom_len)
                scores[idx] += idf * (numerator / denominator)

        return scores


def retrieve_semantic(query: str, workspace_id: str, strategy: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Dense semantic vector retrieval via cosine similarity.
    Scoped strictly by workspace_id and chunking_strategy.
    """
    if strategy not in ("fixed", "section_aware"):
        raise ValueError(f"Invalid strategy '{strategy}'. Must be 'fixed' or 'section_aware'.")

    q_emb = embed_query(query)
    return db.search_chunks(q_emb, workspace_id=workspace_id, strategy=strategy, top_k=top_k)


def retrieve_keyword(query: str, workspace_id: str, strategy: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Sparse lexical retrieval via pure Python BM25.
    Scoped strictly by workspace_id and chunking_strategy.
    """
    if strategy not in ("fixed", "section_aware"):
        raise ValueError(f"Invalid strategy '{strategy}'. Must be 'fixed' or 'section_aware'.")

    filtered_chunks = [
        c for c in db.chunks
        if c["workspace_id"] == workspace_id and c["chunking_strategy"] == strategy
    ]

    if not filtered_chunks:
        return []

    bm25 = BM25()
    corpus = [c["chunk_text"] for c in filtered_chunks]
    bm25.fit(corpus)
    scores = bm25.score(query)

    results = []
    for c, score in zip(filtered_chunks, scores):
        results.append({
            "id": c["id"],
            "paper_id": c["paper_id"],
            "workspace_id": c["workspace_id"],
            "chunk_text": c["chunk_text"],
            "chunking_strategy": c["chunking_strategy"],
            "section_label": c["section_label"],
            "page_number": c["page_number"],
            "bm25_score": round(score, 4)
        })

    # Deterministic sort: BM25 score descending, then alphabetical chunk_id
    results.sort(key=lambda x: (x["bm25_score"], -ord(x["id"][0])), reverse=True)
    return results[:top_k]


def retrieve_hybrid(
    query: str,
    workspace_id: str,
    strategy: str,
    top_k: int = 3,
    rrf_k: int = 60,
    tie_break: str = "dense_similarity"
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval fusing full dense and sparse ranked lists via Reciprocal Rank Fusion (RRF).
    
    Deterministic Multi-Level Tie-Break:
      1. Primary: RRF Score (descending)
      2. Secondary: Dense Cosine Similarity (descending) [if tie_break == 'dense_similarity']
      3. Tertiary: BM25 Score (descending)
      4. Quaternary: Chunk ID (alphabetical deterministic fallback)
      
    RRF is calculated over ALL candidates in the scoped workspace and strategy before truncating to top-k.
    """
    if strategy not in ("fixed", "section_aware"):
        raise ValueError(f"Invalid strategy '{strategy}'. Must be 'fixed' or 'section_aware'.")

    filtered_chunks = [
        c for c in db.chunks
        if c["workspace_id"] == workspace_id and c["chunking_strategy"] == strategy
    ]

    if not filtered_chunks:
        return []

    # 1. Full Dense Ranking over all scoped candidates
    q_vec = np.array(embed_query(query), dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 0:
        q_vec = q_vec / q_norm

    dense_candidates = []
    for c in filtered_chunks:
        emb = c["embedding"]
        emb_norm = np.linalg.norm(emb)
        if emb_norm > 0:
            emb = emb / emb_norm
        cos_sim = float(np.dot(q_vec, emb))
        dense_candidates.append({
            "chunk": c,
            "dense_score": cos_sim
        })
    # Sort dense candidates with deterministic ID fallback
    dense_candidates.sort(key=lambda x: (x["dense_score"], x["chunk"]["id"]), reverse=True)
    
    dense_rank_map = {}
    dense_score_map = {}
    for rank, item in enumerate(dense_candidates, start=1):
        c_id = item["chunk"]["id"]
        dense_rank_map[c_id] = rank
        dense_score_map[c_id] = round(item["dense_score"], 4)

    # 2. Full Sparse (BM25) Ranking over all scoped candidates
    bm25 = BM25()
    corpus = [c["chunk_text"] for c in filtered_chunks]
    bm25.fit(corpus)
    bm25_scores = bm25.score(query)

    sparse_candidates = [
        {"chunk": c, "bm25_score": score}
        for c, score in zip(filtered_chunks, bm25_scores)
    ]
    # Sort sparse candidates with deterministic ID fallback
    sparse_candidates.sort(key=lambda x: (x["bm25_score"], x["chunk"]["id"]), reverse=True)

    sparse_rank_map = {}
    sparse_score_map = {}
    for rank, item in enumerate(sparse_candidates, start=1):
        c_id = item["chunk"]["id"]
        sparse_rank_map[c_id] = rank
        sparse_score_map[c_id] = round(item["bm25_score"], 4)

    # 3. Reciprocal Rank Fusion over all candidates
    fused_results = []
    for c in filtered_chunks:
        c_id = c["id"]
        r_dense = dense_rank_map[c_id]
        r_sparse = sparse_rank_map[c_id]
        
        # Exact RRF float score (not pre-rounded to preserve full precision)
        rrf_val = (1.0 / (rrf_k + r_dense)) + (1.0 / (rrf_k + r_sparse))

        fused_results.append({
            "id": c_id,
            "paper_id": c["paper_id"],
            "workspace_id": c["workspace_id"],
            "chunk_text": c["chunk_text"],
            "chunking_strategy": c["chunking_strategy"],
            "section_label": c["section_label"],
            "page_number": c["page_number"],
            "rrf_score_raw": rrf_val,
            "rrf_score": round(rrf_val, 6),
            "dense_rank": r_dense,
            "dense_similarity": dense_score_map[c_id],
            "sparse_rank": r_sparse,
            "bm25_score": sparse_score_map[c_id]
        })

    # 4. Deterministic Multi-Level Sort:
    # Key = (rrf_score, secondary_score, bm25_score, id)
    if tie_break == "dense_similarity":
        fused_results.sort(
            key=lambda x: (x["rrf_score_raw"], x["dense_similarity"], x["bm25_score"], x["id"]),
            reverse=True
        )
    elif tie_break == "bm25_score":
        fused_results.sort(
            key=lambda x: (x["rrf_score_raw"], x["bm25_score"], x["dense_similarity"], x["id"]),
            reverse=True
        )
    else:  # alphabetical id
        fused_results.sort(
            key=lambda x: (x["rrf_score_raw"], x["id"]),
            reverse=True
        )

    # Detect and annotate tied RRF scores for inspection
    for i in range(len(fused_results)):
        is_tied = False
        if i > 0 and math.isclose(fused_results[i]["rrf_score_raw"], fused_results[i-1]["rrf_score_raw"], rel_tol=1e-9):
            is_tied = True
        if i < len(fused_results) - 1 and math.isclose(fused_results[i]["rrf_score_raw"], fused_results[i+1]["rrf_score_raw"], rel_tol=1e-9):
            is_tied = True
        fused_results[i]["is_rrf_tied"] = is_tied

    return fused_results[:top_k]


def retrieve(query: str, workspace_id: str, strategy: str, mode: str = "hybrid", top_k: int = 3, tie_break: str = "dense_similarity") -> List[Dict[str, Any]]:
    """
    Unified retrieval function supporting 'semantic', 'keyword', and 'hybrid' modes.
    """
    if mode == "semantic":
        return retrieve_semantic(query, workspace_id=workspace_id, strategy=strategy, top_k=top_k)
    elif mode == "keyword":
        return retrieve_keyword(query, workspace_id=workspace_id, strategy=strategy, top_k=top_k)
    elif mode == "hybrid":
        return retrieve_hybrid(query, workspace_id=workspace_id, strategy=strategy, top_k=top_k, tie_break=tie_break)
    else:
        raise ValueError(f"Unknown retrieval mode '{mode}'. Supported modes: 'semantic', 'keyword', 'hybrid'.")


if __name__ == "__main__":
    import json
    from pathlib import Path
    from embeddings import embed_and_store_chunks

    out_dir = Path(__file__).parent / "extracted"
    fixed_file = out_dir / "chunks_fixed.json"
    section_file = out_dir / "chunks_section.json"

    with open(fixed_file, "r", encoding="utf-8") as f:
        fixed_chunks = json.load(f)

    with open(section_file, "r", encoding="utf-8") as f:
        section_chunks = json.load(f)

    # Initialize reference workspace
    ref_ws_id = db.create_workspace(name="PaperHelper Reference Workspace")
    ref_paper_id = db.insert_paper(workspace_id=ref_ws_id, filename="research_major.pdf")

    print(f"Ingesting chunks for Workspace ID: {ref_ws_id}...")
    embed_and_store_chunks(fixed_chunks, workspace_id=ref_ws_id, paper_id=ref_paper_id, strategy="fixed")
    embed_and_store_chunks(section_chunks, workspace_id=ref_ws_id, paper_id=ref_paper_id, strategy="section_aware")

    test_queries = [
        "What F1 score did PaperHelper achieve with RAFT and RAG Fusion?",
        "What are the main limitations and failure cases of the PaperHelper system?",
        "How does RAG Fusion improve over basic RAG?"
    ]

    print("\n" + "=" * 90)
    print("STEP 4: SYSTEMATIC BENCHMARK (SEMANTIC vs HYBRID with CONTROLLED TIE-BREAK)")
    print("Tie-Break Rule: Primary=RRF Score, Secondary=Dense Similarity, Tertiary=BM25, Quaternary=Chunk ID")
    print("=" * 90)

    for q in test_queries:
        print(f"\n==========================================================================================")
        print(f"QUERY: {q}")
        print(f"==========================================================================================")

        for strat in ["fixed", "section_aware"]:
            print(f"\n>>> [Strategy: {strat.upper()}]")
            
            # 1. Semantic-only
            sem_res = retrieve(q, workspace_id=ref_ws_id, strategy=strat, mode="semantic", top_k=3)
            print(f"--- Semantic-Only Search (Top 3) ---")
            for i, r in enumerate(sem_res, 1):
                sec = f"[{r['section_label']}] " if r.get('section_label') else ""
                snip = r['chunk_text'][:115].replace('\n', ' ')
                print(f"  {i}. Sim: {r['similarity']:.4f} | {sec}Page {r['page_number']} | {snip}...")

            # 2. Hybrid (Dense + BM25 via full-list RRF with controlled tie-break)
            hyb_res = retrieve(q, workspace_id=ref_ws_id, strategy=strat, mode="hybrid", top_k=3, tie_break="dense_similarity")
            print(f"\n--- Hybrid Search (Dense + BM25 with Full-List RRF) (Top 3) ---")
            for i, r in enumerate(hyb_res, 1):
                sec = f"[{r['section_label']}] " if r.get('section_label') else ""
                snip = r['chunk_text'][:115].replace('\n', ' ')
                tie_tag = " [TIED - Secondary Sort: Dense Sim]" if r.get('is_rrf_tied') else " [Distinct Score]"
                print(f"  {i}. RRF: {r['rrf_score']:.6f} (DenseRank={r['dense_rank']} Sim={r['dense_similarity']:.4f}, BM25Rank={r['sparse_rank']} BM25={r['bm25_score']:.2f}){tie_tag} | {sec}Page {r['page_number']} | {snip}...")

    # Cross-workspace isolation test for hybrid retrieval
    print("\n" + "=" * 90)
    print("TEST: Cross-Workspace Isolation Verification (Hybrid Mode)")
    print("=" * 90)
    other_ws_id = db.create_workspace(name="Other Workspace")
    iso_res = retrieve("What F1 score was achieved?", workspace_id=other_ws_id, strategy="section_aware", mode="hybrid", top_k=3)
    print(f"Querying separate Workspace ID '{other_ws_id}' with mode='hybrid'...")
    print(f"Results returned: {len(iso_res)} chunks (Expected: 0)")
    assert len(iso_res) == 0, "Isolation failure in hybrid mode!"
    print("Cross-workspace isolation verified successfully in hybrid retrieval mode!")
