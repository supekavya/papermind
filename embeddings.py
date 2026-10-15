"""
Paper Mind - Embeddings and Vector Storage Module (Step 3)
------------------------------------------------------------
Generates 384-dimensional dense vector embeddings using sentence-transformers
(all-MiniLM-L6-v2), manages database storage, and performs workspace-isolated
vector similarity retrieval via pgvector semantics.
"""

import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import dotenv

dotenv.load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
_model_instance = None


def get_embedding_model() -> SentenceTransformer:
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(MODEL_NAME)
    return _model_instance


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generates 384-dimensional dense vector embeddings for a list of strings.
    """
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Generates a normalized 384-dimensional vector embedding for a query string.
    """
    model = get_embedding_model()
    embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()


# ==============================================================================
# In-Memory / Local Vector Store Mirror (Pure Python pgvector implementation)
# ==============================================================================
class VectorDB:
    """
    Manages structured tables and pgvector similarity search.
    Supports relational filtering by workspace_id and chunking_strategy,
    and cosine distance <=> operator ranking.
    """
    def __init__(self):
        self.workspaces: Dict[str, Dict[str, Any]] = {}
        self.papers: Dict[str, Dict[str, Any]] = {}
        self.chunks: List[Dict[str, Any]] = []

    def create_workspace(self, name: str, workspace_id: Optional[str] = None) -> str:
        ws_id = workspace_id or str(uuid.uuid4())
        self.workspaces[ws_id] = {
            "id": ws_id,
            "name": name
        }
        return ws_id

    def insert_paper(self, workspace_id: str, filename: str, raw_text: str = "", paper_id: Optional[str] = None) -> str:
        if workspace_id not in self.workspaces:
            raise ValueError(f"Workspace {workspace_id} does not exist.")
        p_id = paper_id or str(uuid.uuid4())
        self.papers[p_id] = {
            "id": p_id,
            "workspace_id": workspace_id,
            "filename": filename,
            "raw_text": raw_text
        }
        return p_id

    def insert_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Inserts chunks containing embedding vectors and metadata into the store.
        """
        for c in chunks:
            chunk_record = {
                "id": c.get("id", str(uuid.uuid4())),
                "paper_id": c["paper_id"],
                "workspace_id": c["workspace_id"],
                "chunk_text": c["chunk_text"],
                "chunking_strategy": c["chunking_strategy"],
                "section_label": c.get("section_label"),
                "page_number": c.get("page_number"),
                "embedding": np.array(c["embedding"], dtype=np.float32)
            }
            self.chunks.append(chunk_record)

    def search_chunks(self, query_vector: List[float], workspace_id: str, strategy: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Executes pgvector's <=> cosine distance operator scoped strictly by workspace_id and strategy.
        Similarity is calculated as: 1 - (embedding <=> query_vector).
        """
        if not strategy:
            raise ValueError("strategy parameter ('fixed' or 'section_aware') is required and cannot be None.")

        # Step 1: Strict relational filtering by workspace_id and strategy
        filtered = [
            c for c in self.chunks
            if c["workspace_id"] == workspace_id and c["chunking_strategy"] == strategy
        ]

        if not filtered:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        results = []
        for c in filtered:
            emb = c["embedding"]
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                emb = emb / emb_norm

            # Cosine similarity: (u . v) / (||u|| ||v||)
            cosine_sim = float(np.dot(q_vec, emb))
            # Bound similarity to [-1.0, 1.0]
            cosine_sim = max(-1.0, min(1.0, cosine_sim))
            
            results.append({
                "id": c["id"],
                "paper_id": c["paper_id"],
                "workspace_id": c["workspace_id"],
                "chunk_text": c["chunk_text"],
                "chunking_strategy": c["chunking_strategy"],
                "section_label": c["section_label"],
                "page_number": c["page_number"],
                "similarity": round(cosine_sim, 4)
            })

        # Rank descending by cosine similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]


db = VectorDB()


def embed_and_store_chunks(chunks_data: List[Dict[str, Any]], workspace_id: str, paper_id: str, strategy: str) -> int:
    """
    Batch-generates embeddings for chunks and inserts them into the vector store.
    """
    texts = [c["text"] for c in chunks_data]
    print(f"Generating embeddings for {len(texts)} chunks ({strategy} strategy)...")
    embeddings = embed_texts(texts)

    prepared_chunks = []
    for c, emb in zip(chunks_data, embeddings):
        prepared_chunks.append({
            "paper_id": paper_id,
            "workspace_id": workspace_id,
            "chunk_text": c["text"],
            "chunking_strategy": strategy,
            "section_label": c.get("section"),
            "page_number": c.get("page"),
            "embedding": emb
        })

    db.insert_chunks(prepared_chunks)
    print(f"Successfully stored {len(prepared_chunks)} chunks with embeddings.")
    return len(prepared_chunks)


def retrieve_chunks(query: str, workspace_id: str, strategy: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Main retrieval entry point:
    - Embeds query
    - Strictly filters by workspace_id and chunking strategy
    - Returns top-k matching chunks with similarity scores
    """
    if strategy not in ("fixed", "section_aware"):
        raise ValueError(f"Invalid strategy '{strategy}'. Must be 'fixed' or 'section_aware'.")

    q_emb = embed_query(query)
    return db.search_chunks(q_emb, workspace_id=workspace_id, strategy=strategy, top_k=top_k)


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "extracted"
    fixed_file = out_dir / "chunks_fixed.json"
    section_file = out_dir / "chunks_section.json"

    if not fixed_file.exists() or not section_file.exists():
        print("Error: Chunks files not found. Run chunking.py first.")
        exit(1)

    with open(fixed_file, "r", encoding="utf-8") as f:
        fixed_chunks = json.load(f)

    with open(section_file, "r", encoding="utf-8") as f:
        section_chunks = json.load(f)

    # 1. Create reference workspace and paper
    ref_ws_id = db.create_workspace(name="PaperHelper Research Workspace")
    ref_paper_id = db.insert_paper(workspace_id=ref_ws_id, filename="research_major.pdf")

    print(f"Created Workspace ID: {ref_ws_id}")
    print(f"Created Paper ID:     {ref_paper_id}\n")

    # 2. Ingest embeddings for both chunking strategies
    embed_and_store_chunks(fixed_chunks, workspace_id=ref_ws_id, paper_id=ref_paper_id, strategy="fixed")
    embed_and_store_chunks(section_chunks, workspace_id=ref_ws_id, paper_id=ref_paper_id, strategy="section_aware")

    # 3. Exact and near-duplicate query verification
    print("\n" + "=" * 80)
    print("TEST 1: Exact and Near-Duplicate Match Verification (Similarity Calibration)")
    print("=" * 80)
    
    # 3a. Exact match test (query == full chunk text)
    sample_chunk = section_chunks[1]["text"]
    exact_res = retrieve_chunks(sample_chunk, workspace_id=ref_ws_id, strategy="section_aware", top_k=1)
    print(f"[3a] Exact Full-Chunk Query:\n     Snippet: {sample_chunk[:100]}...")
    if exact_res:
        print(f"     Top Match Similarity: {exact_res[0]['similarity']:.4f} (Expected: ~1.0000)")
        print(f"     Section:              {exact_res[0]['section_label']}")
        print(f"     Page:                 {exact_res[0]['page_number']}")

    # 3b. Near-duplicate test (query == sub-sentence from chunk)
    sub_sentence = "In the paper, we introduce a paper reading assistant, PaperHelper, a potent tool designed to enhance the capabilities of researchers in efficiently browsing"
    sub_res = retrieve_chunks(sub_sentence, workspace_id=ref_ws_id, strategy="section_aware", top_k=1)
    print(f"\n[3b] Near-Duplicate Sub-Sentence Query:\n     Query: {sub_sentence[:100]}...")
    if sub_res:
        print(f"     Top Match Similarity: {sub_res[0]['similarity']:.4f} (Expected: High > 0.85)")
        print(f"     Section:              {sub_res[0]['section_label']}")
        print(f"     Page:                 {sub_res[0]['page_number']}")

    # 4. Reference queries comparison (fixed vs section_aware)
    test_queries = [
        "What F1 score did PaperHelper achieve with RAFT and RAG Fusion?",
        "What are the main limitations and failure cases of the PaperHelper system?",
        "How does RAG Fusion improve over basic RAG?"
    ]

    print("\n" + "=" * 80)
    print("TEST 2: Retrieval Comparison Across Swappable Chunking Strategies")
    print("=" * 80)

    for q in test_queries:
        print(f"\n>>> QUERY: {q}")
        
        print("\n--- Strategy: fixed ---")
        fixed_res = retrieve_chunks(q, workspace_id=ref_ws_id, strategy="fixed", top_k=3)
        for i, r in enumerate(fixed_res, 1):
            snippet = r['chunk_text'][:130].replace('\n', ' ')
            print(f"  [{i}] Similarity: {r['similarity']:.4f} | Page {r['page_number']} | Snippet: {snippet}...")

        print("\n--- Strategy: section_aware ---")
        sec_res = retrieve_chunks(q, workspace_id=ref_ws_id, strategy="section_aware", top_k=3)
        for i, r in enumerate(sec_res, 1):
            snippet = r['chunk_text'][:130].replace('\n', ' ')
            print(f"  [{i}] Similarity: {r['similarity']:.4f} | [{r['section_label']}] Page {r['page_number']} | Snippet: {snippet}...")

    # 5. Cross-workspace isolation test
    print("\n" + "=" * 80)
    print("TEST 3: Cross-Workspace Isolation Verification")
    print("=" * 80)
    other_ws_id = db.create_workspace(name="Isolated Other Workspace")
    isolated_res = retrieve_chunks("What F1 score was achieved?", workspace_id=other_ws_id, strategy="section_aware", top_k=3)
    print(f"Querying separate Workspace ID '{other_ws_id}'...")
    print(f"Results returned: {len(isolated_res)} chunks (Expected: 0)")
    assert len(isolated_res) == 0, "Isolation failure: chunks leaked across workspace boundary!"
    print("Workspace isolation verified successfully!")
