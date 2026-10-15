"""
Unit test for verify_claim_grounding in agent.py
Tests:
1. Legitimate citations with real paper wording (should have ZERO false flags)
2. Hallucinated mechanism claim with 'distractor' (must flag ONLY 'distractor')
"""

import sys
import re
from pathlib import Path
import json

# Add project root to sys.path
sys.path.insert(0, r"c:\Users\supev\OneDrive\Desktop\papermind")

from agent import verify_claim_grounding

# Load actual section chunks from extracted/chunks_section.json
with open(r"c:\Users\supev\OneDrive\Desktop\papermind\extracted\chunks_section.json", "r", encoding="utf-8") as f:
    chunks_raw = json.load(f)

accumulated_evidence = []
for c in chunks_raw:
    if c["section"] in ["4 Experimental Results", "3.3 Implementation of RAFT", "5.2 Datasets"]:
        accumulated_evidence.append({
            "id": f"chunk_{c['section']}",
            "section_label": c["section"],
            "page_number": c["page"],
            "chunk_text": c["text"]
        })

print(f"Loaded {len(accumulated_evidence)} chunks for testing.")

# Test Case 1: Legitimate sentences from real LLM generation
legitimate_answer = (
    "**(1) Benchmark Dataset, Sample Size, and Top-k Value:**\n"
    "The testing was conducted using a test set of 100 papers from the MLArxivPapers dataset (arXiv.org submitters, 2024). "
    "The top-k value used was 10, and 500 tokens were generated [Section: 4 Experimental Results, Page: 4].\n\n"
    "**(2) RAFT Training Details:**\n"
    "RAFT was trained by finetuning the OpenAI API using 5,000 domain-specific papers from the field of machine learning, "
    "augmenting the knowledge of PaperHelper [Section: 3.3 Implementation of RAFT, Page: 3]. "
    "These papers were selected from the larger MLArxivPapers dataset (over 104K papers) [Section: 5.2 Datasets, Page: 5].\n\n"
    "**(3) Milvus Hardware Acceleration Advantage:**\n"
    "Milvus has a slight performance advantage due to its support for GPU acceleration, resulting in faster latency compared to Faiss and Qdrant [Section: 4 Experimental Results, Page: 4]."
)

print("\n--- Running Test Case 1: Legitimate Sentences ---")
ver_ans1, audit1 = verify_claim_grounding(legitimate_answer, accumulated_evidence)
print(f"Has unsupported claims: {audit1['has_unsupported_claims']}")
if audit1['flagged_claims']:
    print("Flagged claims:")
    for fc in audit1['flagged_claims']:
        print(f"  - Terms: {fc['unsupported_terms']}")
        print(f"    Claim: {fc['claim']}")
else:
    print("PASS: Zero false flags on legitimate sentences!")

# Test Case 2: Hallucinated claim containing 'distractor'
hallucinated_answer = (
    "RAFT was trained by distinguishing reference facts from distractor documents in advance using 5,000 papers [Section: 3.3 Implementation of RAFT, Page: 3]."
)

print("\n--- Running Test Case 2: Hallucinated 'distractor' Claim ---")
ver_ans2, audit2 = verify_claim_grounding(hallucinated_answer, accumulated_evidence)
print(f"Has unsupported claims: {audit2['has_unsupported_claims']}")
if audit2['flagged_claims']:
    print("Flagged claims:")
    for fc in audit2['flagged_claims']:
        print(f"  - Terms: {fc['unsupported_terms']}")
        print(f"    Claim: {fc['claim']}")
    assert "distractor" in audit2['flagged_claims'][0]['unsupported_terms'], "Expected 'distractor' to be flagged!"
    print("PASS: Successfully and exclusively flagged 'distractor'!")
else:
    print("FAIL: Did not flag hallucinated claim!")
