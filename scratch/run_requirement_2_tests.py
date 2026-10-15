"""
Comprehensive Grounding Tests for Requirement (1) & (2):
1. A grounded sentence containing a number
2. A paraphrase of a grounded claim
3. A wrong number cited to the correct chunk
4. A swapped number taken from the same chunk (500 papers, top-k 100)
5. An entity-relation error: RAG Fusion claimed as a vector DB
6. An uncited factual sentence (citation required)
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, r"c:\Users\supev\OneDrive\Desktop\papermind")

from agent import verify_claim_grounding

# Load actual section chunks
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

print("=" * 80)
print("GROUNDING & ENTITY/RELATION TEST SUITE (Requirement 1 & 2)")
print("=" * 80)

# Test 1: A grounded sentence containing a number
test_1_input = "We evaluate our system performance on 100 papers from MLArxivPapers with top-k set to 10 [Section: 4 Experimental Results, Page: 4]."
ans1, audit1 = verify_claim_grounding(test_1_input, accumulated_evidence)

print("\n--- TEST 1: Grounded sentence containing a number ---")
print(f"INPUT CLAIM:\n  {test_1_input}")
print(f"VERIFIED ANSWER:\n  {ans1}")
print(f"AUDIT:\n{json.dumps(audit1, indent=2)}")

# Test 2: A paraphrase of a grounded claim
test_2_input = "Milvus achieved lower latency compared to other vector databases owing to hardware acceleration on GPUs [Section: 4 Experimental Results, Page: 4]."
ans2, audit2 = verify_claim_grounding(test_2_input, accumulated_evidence)

print("\n--- TEST 2: Paraphrase of a grounded claim ---")
print(f"INPUT CLAIM:\n  {test_2_input}")
print(f"VERIFIED ANSWER:\n  {ans2}")
print(f"AUDIT:\n{json.dumps(audit2, indent=2)}")

# Test 3: A wrong number cited to the correct chunk
test_3_input = "We evaluate our system performance on 950 papers from MLArxivPapers with top-k set to 10 [Section: 4 Experimental Results, Page: 4]."
ans3, audit3 = verify_claim_grounding(test_3_input, accumulated_evidence)

print("\n--- TEST 3: Wrong number cited to the correct chunk ---")
print(f"INPUT CLAIM:\n  {test_3_input}")
print(f"VERIFIED ANSWER:\n{ans3}")
print(f"AUDIT:\n{json.dumps(audit3, indent=2)}")

# Test 4: Swapped numbers taken from the same chunk (500 tokens -> 500 papers, 100 papers -> top-k 100)
test_4_input = "We evaluate our system performance on 500 papers from MLArxivPapers with top-k set to 100 [Section: 4 Experimental Results, Page: 4]."
ans4, audit4 = verify_claim_grounding(test_4_input, accumulated_evidence)

print("\n--- TEST 4: Swapped numbers taken from the same chunk ---")
print(f"INPUT CLAIM:\n  {test_4_input}")
print(f"VERIFIED ANSWER:\n{ans4}")
print(f"AUDIT:\n{json.dumps(audit4, indent=2)}")

# Test 5: Entity/relation error - RAG Fusion claimed as vector database
test_5_input = "Milvus has support for GPU acceleration, resulting in faster latency compared to Faiss, Qdrant, and RAG Fusion [Section: 4 Experimental Results, Page: 4]."
ans5, audit5 = verify_claim_grounding(test_5_input, accumulated_evidence)

print("\n--- TEST 5: Entity/relation error (RAG Fusion claimed as vector database) ---")
print(f"INPUT CLAIM:\n  {test_5_input}")
print(f"VERIFIED ANSWER:\n{ans5}")
print(f"AUDIT:\n{json.dumps(audit5, indent=2)}")

# Test 6: Uncited factual sentence
test_6_input = "The testing was conducted using 100 papers from the MLArxivPapers dataset."
ans6, audit6 = verify_claim_grounding(test_6_input, accumulated_evidence)

print("\n--- TEST 6: Uncited factual sentence (citation required) ---")
print(f"INPUT CLAIM:\n  {test_6_input}")
print(f"VERIFIED ANSWER:\n{ans6}")
print(f"AUDIT:\n{json.dumps(audit6, indent=2)}")
print("\n" + "=" * 80)
