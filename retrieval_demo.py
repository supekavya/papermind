"""
Paper Mind - Retrieval Demo
-----------------------------
Working end-to-end retrieval: vectorize chunks -> vectorize query ->
cosine similarity -> return top matches, with source (page/section) shown.

Uses TF-IDF as a stand-in for sentence-transformer embeddings (this sandbox
has no internet access to Hugging Face) - same retrieval MECHANISM
(vectorize -> compare -> rank), swappable for real embeddings later.
"""

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_chunks(path):
    with open(path) as f:
        return json.load(f)


def retrieve(query, chunks, top_k=3):
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_vectors = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(query_vector, chunk_vectors)[0]
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def run_query(query, fixed_chunks, section_chunks):
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")

    print(f"\n--- Fixed-size chunking results ---")
    for chunk, score in retrieve(query, fixed_chunks):
        preview = chunk["text"][:150].replace("\n", " ")
        print(f"  score={score:.3f} | page {chunk['page']} | {preview}...")

    print(f"\n--- Section-aware chunking results ---")
    for chunk, score in retrieve(query, section_chunks):
        preview = chunk["text"][:150].replace("\n", " ")
        print(f"  score={score:.3f} | [{chunk['section']}] page {chunk['page']} | {preview}...")


if __name__ == "__main__":
    fixed_chunks = load_chunks("/home/claude/extracted/chunks_fixed.json")
    section_chunks = load_chunks("/home/claude/extracted/chunks_section.json")

    test_queries = [
        "What F1 score did the system achieve?",
        "What are the limitations of the system?",
        "How does RAG Fusion work?",
    ]

    for q in test_queries:
        run_query(q, fixed_chunks, section_chunks)
