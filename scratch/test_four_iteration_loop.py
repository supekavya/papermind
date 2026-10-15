"""
Test 4-Iteration Agentic Loop (Requirement 4)
Demonstrates genuine 4-iteration execution showing 3 or more distinct reformulated queries in the trace.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, r"c:\Users\supev\OneDrive\Desktop\papermind")

# Ensure UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from embeddings import db, embed_and_store_chunks
from agent import run_agent_loop

# 1. Initialize data in workspace
out_dir = Path(r"c:\Users\supev\OneDrive\Desktop\papermind\extracted")
section_file = out_dir / "chunks_section.json"

with open(section_file, "r", encoding="utf-8") as f:
    section_chunks = json.load(f)

ws_id = db.create_workspace(name="4-Iteration Evaluation Workspace")
paper_id = db.insert_paper(workspace_id=ws_id, filename="research_major.pdf")

embed_and_store_chunks(section_chunks, workspace_id=ws_id, paper_id=paper_id, strategy="section_aware")

# 4 distinct questions requiring 4 separate sections across the entire paper:
# 4 distinct questions requiring 4 separate sections across the paper:
# Facet 1: Abstract (Page 1) - F1 score 60.04 and latency 5.8s
# Facet 2: Section 3.1 Overview (Page 2) - The three tools (search, gather evidence, answer questions)
# Facet 3: Section 3.4 (Page 4) - Mermaid format markdown code for reference relationships
# Facet 4: Section 5.4 Limitations (Page 6) - The three image limitations (dense/abstract, multimodal struggles, lack of expert feedback)
four_facet_question = (
    "Answer all four specific questions from the paper: "
    "(1) According to the Abstract, what exact F1 score and latency did PaperHelper achieve? "
    "(2) According to Section 3.1 Overview, what exact three tools does the assistant utilize? "
    "(3) According to Section 3.4, what syntax format is generated to display connections between cited papers? "
    "(4) According to Section 5.4 Limitations, what exact three primary limitations prevent converting paper images into vectors?"
)

print("\n" + "=" * 90)
print("RUNNING 4-ITERATION AGENT LOOP (max_iterations=4, top_k_per_step=1)")
print("=" * 90)

result = run_agent_loop(
    question=four_facet_question,
    workspace_id=ws_id,
    strategy="section_aware",
    mode="hybrid",
    max_iterations=4,
    top_k_per_step=1
)

print("\n" + "=" * 90)
print("RAW PER-ITERATION TRACE (Requirement 4)")
print("=" * 90)
print(f"Total Retrieval Steps: {result['reasoning_trace']['total_retrieval_calls']}")
print(f"Total Chunks Accumulated: {result['reasoning_trace']['accumulated_chunks_count']}")
print(f"Sufficiency Status: {result['evidence_sufficiency'].upper()}")

reformulated_queries = []
for step in result['reasoning_trace']['steps']:
    it = step['iteration']
    q_used = step['query_used']
    suff = step['sufficiency_evaluation']
    print(f"\n[Iteration {it}]")
    print(f"  Query Executed: '{q_used}'")
    print(f"  Retrieved Chunks: {step['retrieved_count']}")
    for cs in step['chunks_summary']:
        print(f"    - Section: {cs.get('section')} (Page {cs.get('page')}) | RRF: {cs.get('rrf_score')}")
    print(f"  Self-Check is_sufficient: {suff.get('is_sufficient')}")
    print(f"  Evidence Support Score: {suff.get('evidence_support_score')}")
    print(f"  Rationale: {suff.get('rationale')}")
    if not suff.get('is_sufficient'):
        print(f"  Missing Aspects: {suff.get('missing_aspects')}")
    if it > 1:
        reformulated_queries.append(q_used)

print("\n" + "=" * 90)
print(f"DISTINCT REFORMULATED QUERIES ({len(reformulated_queries)} generated):")
for idx, rq in enumerate(reformulated_queries, 1):
    print(f"  Reformulated Query {idx}: '{rq}'")
print("=" * 90)

print("\nFINAL GROUNDED ANSWER:")
print(result['answer'])
print("\nCITATIONS:")
for cit in result['citations']:
    print(f"  - [{cit['section']}, Page {cit['page']}]")
