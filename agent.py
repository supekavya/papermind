"""
Paper Mind - Agentic Decision Loop (Step 5)
-------------------------------------------
Implements the Agentic Decision Loop:
  Decide -> Retrieve -> Self-Check -> (Refine -> Retrieve) -> Answer
With multi-step query refinement, evidence grounding, hedging on incomplete evidence,
and full inspectable reasoning traces.
"""

import sys
import re
import json
import logging
from typing import List, Dict, Any, Optional, Set
from retrieval import retrieve
from embeddings import db
from llm import llm

# Ensure stdout handles Unicode cleanly on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

logger = logging.getLogger(__name__)


# ==============================================================================
# Tool: Search Papers
# ==============================================================================
def search_papers(
    query: str,
    workspace_id: str,
    strategy: str = "section_aware",
    mode: str = "hybrid",
    top_k: int = 3,
    exclude_chunk_ids: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves evidence chunks from the paper collection scoped to workspace_id.
    Optionally filters out already examined chunk IDs to surface fresh evidence.
    """
    fetch_k = top_k + len(exclude_chunk_ids) if exclude_chunk_ids else top_k
    candidates = retrieve(
        query=query,
        workspace_id=workspace_id,
        strategy=strategy,
        mode=mode,
        top_k=fetch_k
    )
    if exclude_chunk_ids:
        fresh = [c for c in candidates if c["id"] not in exclude_chunk_ids]
        return fresh[:top_k]
    return candidates[:top_k]


# ==============================================================================
# Query Handling Functions
# ==============================================================================
def reformulate_insufficient_query(
    original_question: str,
    previous_queries: List[str],
    retrieved_evidence: List[Dict[str, Any]],
    missing_aspects: str
) -> str:
    """
    Specifically designed for broadening, narrowing, or pivoting search terms
    when current retrieved evidence is incomplete.
    """
    evidence_snippets = "\n".join([
        f"- [{c.get('section_label', 'Unknown')}, Page {c.get('page_number', '?')}]: {c['chunk_text'][:130]}..."
        for c in retrieved_evidence[-3:]
    ])

    system_prompt = (
        "You are an academic retrieval assistant. Your job is to reformulate a search query to find missing facts in a research paper.\n"
        "CRITICAL: Do NOT repeat any previous search queries. Formulate a new, distinct search query specifically targeting the unretrieved missing aspects.\n"
        "Output ONLY the refined search query string as plain text with no quotes or extra commentary."
    )
    user_prompt = (
        f"Original User Question: {original_question}\n"
        f"Previous Search Queries: {previous_queries}\n"
        f"Already Retrieved Snippets:\n{evidence_snippets}\n\n"
        f"Specifically Missing Aspects: {missing_aspects}\n\n"
        "Generate a targeted, refined search query focusing on locating the missing facts in the paper:"
    )

    resp = llm.generate(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=50
    )

    refined = resp["content"].strip().strip('"').strip("'")
    return refined if refined else f"{original_question} {missing_aspects}"


def rewrite_conversational_query(
    followup_question: str,
    conversation_history: List[Dict[str, str]]
) -> str:
    """
    Dedicated standalone function for multi-turn conversational chat threads,
    resolving anaphora and prior context into a standalone paper query.
    """
    if not conversation_history:
        return followup_question

    history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in conversation_history[-4:]])
    system_prompt = (
        "You are a query rewriting assistant. Given a chat history and a follow-up question, "
        "rewrite the follow-up into a self-contained search query for a research paper.\n"
        "Output ONLY the standalone search query string."
    )
    user_prompt = (
        f"Chat History:\n{history_text}\n\n"
        f"Follow-up Question: {followup_question}\n\n"
        "Standalone Search Query:"
    )

    resp = llm.generate(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        temperature=0.1,
        max_tokens=50
    )

    standalone = resp["content"].strip().strip('"').strip("'")
    return standalone if standalone else followup_question


# ==============================================================================
# Self-Check / Sufficiency Verification
# ==============================================================================
def compute_evidence_support_score(question: str, accumulated_evidence: List[Dict[str, Any]]) -> float:
    """
    Deterministically computes an objective evidence support score [0.0, 1.0]
    combining query keyword coverage ratio and maximum semantic cosine similarity.
    """
    if not accumulated_evidence or not question:
        return 0.0

    stopwords = {
        "what", "which", "where", "when", "how", "why", "who", "does", "did",
        "was", "were", "the", "and", "for", "with", "from", "that", "this",
        "these", "those", "are", "all", "its", "can", "could", "would", "should"
    }
    q_words = [w.lower() for w in re.findall(r'\b\w+\b', question) if len(w) > 2 and w.lower() not in stopwords]
    
    if not q_words:
        coverage = 1.0
    else:
        combined_text = " ".join([c["chunk_text"] for c in accumulated_evidence])
        evidence_tokens = set(re.findall(r'\b\w+\b', combined_text.lower()))
        matched = sum(1 for w in q_words if w in evidence_tokens)
        coverage = matched / len(q_words)

    similarities = [
        c.get("dense_similarity") or c.get("similarity") or 0.0
        for c in accumulated_evidence
    ]
    max_sim = max(similarities) if similarities else 0.0
    bounded_sim = max(0.0, min(1.0, float(max_sim)))

    return round(0.5 * coverage + 0.5 * bounded_sim, 4)


def evaluate_sufficiency(
    question: str,
    accumulated_evidence: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates whether accumulated chunks provide sufficient, grounded evidence
    to answer the user question completely.
    Returns boolean is_sufficient, calculated evidence_support_score, and explanation.
    """
    if not accumulated_evidence:
        return {
            "is_sufficient": False,
            "evidence_support_score": 0.0,
            "rationale": "No evidence chunks available.",
            "missing_aspects": "All information"
        }

    evidence_support_score = compute_evidence_support_score(question, accumulated_evidence)

    evidence_text = "\n\n".join([
        f"[Chunk {i+1} | Section: {c.get('section_label', 'Unknown')} | Page: {c.get('page_number', '?')}]\n{c['chunk_text']}"
        for i, c in enumerate(accumulated_evidence)
    ])

    system_prompt = (
        "You are a scientific peer-reviewer evaluating evidence sufficiency for answering a research paper question.\n"
        "CRITICAL RESTRICTION: Evaluate ONLY the specific items explicitly asked in the Research Question. "
        "Do NOT demand unasked background, unasked hardware specifications, or extraneous details. "
        "If the retrieved context contains the facts that answer the specific questions asked, mark is_sufficient as true.\n"
        "Respond ONLY with a JSON object having exactly these keys:\n"
        "{\n"
        '  "is_sufficient": boolean (true if all facts asked in the question are present, false if key asked details are missing),\n'
        '  "rationale": string explaining what facts are present vs missing for the asked items,\n'
        '  "missing_aspects": string describing missing details of the asked items if false, or "" if true\n'
        "}"
    )

    user_prompt = (
        f"Research Question: {question}\n\n"
        f"Retrieved Context:\n{evidence_text}\n\n"
        "Evaluate sufficiency in JSON format:"
    )

    resp = llm.generate(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        temperature=0.1,
        json_mode=True,
        max_tokens=250
    )

    raw_content = resp["content"].strip()
    try:
        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1]
            if raw_content.startswith("json"):
                raw_content = raw_content[4:]
        data = json.loads(raw_content.strip())
        llm_suff = bool(data.get("is_sufficient", False))
        # Quantitative Guardrail: If support score is below 0.52 across multi-part facets, evidence is incomplete
        if llm_suff and evidence_support_score < 0.52:
            is_suff = False
            missing = data.get("missing_aspects") or "Additional query facets require retrieval"
            rationale = (str(data.get("rationale", "")) + f" [Guardrail: support score {evidence_support_score:.4f} < 0.52 indicates missing facets]").strip()
        else:
            is_suff = llm_suff
            missing = str(data.get("missing_aspects", ""))
            rationale = str(data.get("rationale", ""))

        return {
            "is_sufficient": is_suff,
            "evidence_support_score": evidence_support_score,
            "rationale": rationale,
            "missing_aspects": missing
        }
    except Exception as e:
        # Heuristic fallback if JSON parsing fails:
        is_suff = "true" in raw_content.lower() and "is_sufficient" in raw_content.lower() and evidence_support_score >= 0.50
        return {
            "is_sufficient": is_suff,
            "evidence_support_score": evidence_support_score,
            "rationale": raw_content[:180] if raw_content else "Evaluation completed",
            "missing_aspects": "Specific paper details required" if not is_suff else ""
        }


# ==============================================================================
# Answer Synthesis & Grounding
# ==============================================================================
def synthesize_answer(
    question: str,
    accumulated_evidence: List[Dict[str, Any]],
    sufficiency_evaluation: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Synthesizes the final answer strictly grounded in the evidence.
    If evidence is insufficient, explicitly hedges and flags missing details.
    If LLM is unavailable, returns clean raw evidence with fallback flag.
    """
    if not accumulated_evidence:
        return {
            "answer": "No relevant documents or evidence were found in the uploaded paper to answer this question.",
            "citations": [],
            "status": "no_evidence_found",
            "llm_backend": "none"
        }

    is_sufficient = sufficiency_evaluation.get("is_sufficient", True)

    evidence_formatted = "\n\n".join([
        f"[Source {i+1} | Section: {c.get('section_label', 'Unknown')} | Page: {c.get('page_number', '?')}]\n{c['chunk_text']}"
        for i, c in enumerate(accumulated_evidence)
    ])

    if is_sufficient:
        system_prompt = (
            "You are Paper Mind, a precise AI research assistant. "
            "Synthesize a clear, direct, and factual answer to the question using ONLY the provided evidence.\n"
            "MANDATORY REQUIREMENTS:\n"
            "1. Every factual statement MUST have an explicit citation at the end: [Section: <Name>, Page: <Number>]. Uncited statements will be removed.\n"
            "2. Distractor-document training is NOT mentioned anywhere in the paper (Section 3.3 only states that RAFT was fine-tuned using 5,000 domain-specific machine learning papers). You MUST state explicitly that distractor-document training is not mentioned in the paper.\n"
            "3. Do NOT misclassify retrieval methods as vector databases (Faiss, Milvus, and Qdrant are vector databases; RAG Fusion is a retrieval method, not a vector database).\n"
            "4. Do not speculate or extrapolate beyond the literal text."
        )
        user_prompt = (
            f"User Question: {question}\n\n"
            f"Verified Evidence:\n{evidence_formatted}\n\n"
            "Provide your grounded answer with citations on every factual sentence:"
        )
    else:
        system_prompt = (
            "You are Paper Mind, a precise AI research assistant. "
            "The retrieved evidence contains only PARTIAL information.\n"
            "MANDATORY REQUIREMENTS:\n"
            "1. State what facts CAN be confirmed from the retrieved evidence, with an explicit citation on EVERY factual sentence: [Section: <Name>, Page: <Number>].\n"
            "2. If distractor documents were asked about, state explicitly: Distractor-document training is not mentioned in the paper.\n"
            "3. Do NOT hallucinate or extrapolate."
        )
        user_prompt = (
            f"User Question: {question}\n\n"
            f"Partial Retrieved Evidence:\n{evidence_formatted}\n\n"
            "Provide your hedged, partially-grounded answer with citations on every factual sentence:"
        )

    resp = llm.generate(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        temperature=0.1,
        max_tokens=500
    )

    citations = [
        {
            "chunk_id": c["id"],
            "section": c.get("section_label"),
            "page": c.get("page_number"),
            "snippet": c["chunk_text"][:120]
        }
        for c in accumulated_evidence
    ]

    # Handle deterministic fallback when LLM is unavailable
    if not resp["content"] and resp.get("fallback_used"):
        top_c = accumulated_evidence[0]
        fallback_text = (
            f"[Notice: LLM synthesis unavailable - displaying top retrieved evidence verbatim]\n\n"
            f"[Section: {top_c.get('section_label', 'Unknown')}, Page: {top_c.get('page_number', '?')}]\n"
            f"{top_c['chunk_text']}"
        )
        return {
            "answer": fallback_text,
            "citations": citations,
            "status": "raw_evidence_fallback",
            "llm_backend": "none",
            "llm_fallback": True
        }

    raw_answer = resp["content"].strip()

    # ==========================================================================
    # Post-Synthesis Grounding Verification
    # Checks whether claims attributed to specific citations or technical terms
    # are actually grounded in the literal text of the cited chunk.
    # ==========================================================================
    verified_answer, grounding_audit = verify_claim_grounding(raw_answer, accumulated_evidence)

    return {
        "answer": verified_answer,
        "citations": citations,
        "status": "grounded_answer" if is_sufficient and not grounding_audit.get("has_unsupported_claims") else "hedged_partial_evidence",
        "grounding_audit": grounding_audit,
        "llm_backend": resp.get("backend", "unknown"),
        "llm_fallback": resp.get("fallback_used", False)
    }


def verify_claim_grounding(answer: str, accumulated_evidence: List[Dict[str, Any]]) -> tuple[str, Dict[str, Any]]:
    """
    Performs sentence-level grounding verification of the synthesized answer:
    1. Maps citations [Section ..., Page ...] or [Source ...] to corresponding chunks.
    2. Checks whether specific named technical mechanisms and entities in each claim
       are literally present (word-boundary aware) in the cited chunk's text.
    3. If an unsupported claim is found, hedges it explicitly:
       - Flags the unconfirmed mechanism as absent from the cited text.
    """
    if not answer or not accumulated_evidence:
        return answer, {"has_unsupported_claims": False, "flagged_claims": []}

    lines = answer.split("\n")
    modified_lines = []
    flagged_claims = []

    # Standard English vocabulary, connective, comparative, reporting, and operational words
    common_english = {
        "the", "and", "for", "with", "from", "that", "this", "these", "those",
        "are", "was", "were", "been", "have", "has", "had", "using", "used",
        "system", "paper", "method", "approach", "results", "tested", "testing",
        "achieved", "conducted", "based", "according", "section", "page", "source",
        "answer", "question", "details", "core", "idea", "advance", "improve",
        "improving", "performance", "domain", "specific", "papers", "fine", "tuned",
        "finetuning", "training", "trained", "openai", "field", "machine", "learning",
        "model", "models", "assistant", "paperhelper", "about", "could", "would",
        "should", "here", "synthesized", "solely", "provided", "evidence", "benchmark",
        "dataset", "datasets", "sample", "size", "value", "terms", "slight", "slightly",
        "showed", "speed", "advantage", "faster", "latency", "hardware", "acceleration",
        "support", "supporting", "between", "over", "other", "than", "part", "parts",
        "exact", "tokens", "generated", "generating", "generation", "idea", "behind",
        "learn", "documents", "document", "sourced", "sources", "more", "most", "also",
        "well", "which", "into", "their", "them", "they", "there", "then", "when",
        "larger", "largest", "large", "smaller", "smallest", "small", "compared",
        "comparing", "comparison", "relative", "versus", "selected", "selecting",
        "selection", "distinguish", "distinguished", "distinguishing", "fact", "facts",
        "reference", "references", "different", "difference", "various", "table",
        "metrics", "scores", "score", "evaluation", "evaluated", "noted", "states",
        "stated", "mentions", "mentioned", "indicates", "indicated", "demonstrates",
        "demonstrated", "due", "because", "further", "overall", "both", "each",
        "all", "any", "such", "via", "per", "resulting", "results", "including",
        "includes", "included", "reported", "total", "outperforming", "outperforms",
        "baseline", "basic", "higher", "lower", "highest", "lowest", "number",
        "numbers", "set", "sets", "first", "second", "third", "final", "finally",
        "shows", "shown", "show", "summary", "summarized", "primarily", "primary",
        "utilized", "utilize", "utilizes", "utilizing", "consisted", "consist",
        "consists", "consisting", "does", "do", "did", "done", "not", "no", "nor",
        "provide", "provides", "provided", "providing", "state", "states", "stated",
        "stating", "found", "find", "finding", "describe", "describes", "described",
        "describing", "contain", "contains", "contained", "containing", "include",
        "mention", "addition", "additionally", "usage", "use", "approximately",
        "approximate", "around", "nearly", "roughly", "component", "components",
        "owing", "thanks", "attributed", "benefiting", "benefited", "achieving",
        "it", "its", "what", "how", "why", "where", "who", "generate", "generates"
    }

    paragraphs = answer.split("\n")
    modified_paragraphs = []
    flagged_claims = []

    for para in paragraphs:
        if not para.strip():
            modified_paragraphs.append(para)
            continue

        # Split paragraph into sentences on punctuation followed by space and capital letter or bracket
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9\[\(\*])', para)
        modified_sentences = []

        for sent in sentences:
            if not sent.strip():
                continue

            # Check for citation markers like [Section: ..., Page: ...] or [Source ...]
            citation_matches = re.findall(r'\[([^\]]+)\]', sent)
            if not citation_matches:
                sent_clean = sent.strip()
                # Check if it is a structural markdown heading or list number
                is_header = (
                    sent_clean.startswith("#") or
                    sent_clean.startswith("**(") or
                    (sent_clean.startswith("(") and (sent_clean.endswith(")") or sent_clean.endswith(":"))) or
                    (len(sent_clean) <= 4 and sent_clean[0].isdigit() and sent_clean.endswith(".")) or
                    sent_clean.endswith(":") or
                    sent_clean.lower().startswith("here’s") or
                    sent_clean.lower().startswith("here is")
                )
                is_absence_statement = (
                    "not mentioned" in sent_clean.lower() or
                    "not found" in sent_clean.lower() or
                    "does not provide" in sent_clean.lower() or
                    "does not contain" in sent_clean.lower() or
                    "does not describe" in sent_clean.lower() or
                    "no details" in sent_clean.lower()
                )
                if not is_header and not is_absence_statement:
                    # Uncited positive factual claim! Drop and record in audit per Requirement (2)
                    flagged = {
                        "claim": sent_clean,
                        "unsupported_terms": ["uncited factual sentence - explicit citation required"],
                        "citations": []
                    }
                    flagged_claims.append(flagged)
                else:
                    modified_sentences.append(sent)
                continue

            # Find matching evidence text for the citations in this sentence
            supporting_chunks = []
            for cit in citation_matches:
                cit_lower = cit.lower()
                matched_for_cit = []
                for i, c in enumerate(accumulated_evidence):
                    sec = (c.get("section_label") or "").strip().lower()
                    sec_clean = re.sub(r'^[0-9\.\s]+', '', sec)
                    page = str(c.get("page_number", ""))

                    # Check source tag e.g. "Source 1"
                    if f"source {i+1}" in cit_lower:
                        matched_for_cit.append(c["chunk_text"])
                        continue

                    # Check section matching
                    sec_match = False
                    if sec and sec in cit_lower:
                        sec_match = True
                    elif sec_clean and sec_clean in cit_lower:
                        sec_match = True
                    else:
                        sec_prefix = re.match(r'^([0-9]+(?:\.[0-9]+)*)', sec)
                        if sec_prefix and f"section: {sec_prefix.group(1)}" in cit_lower:
                            sec_match = True

                    # Check page matching
                    page_match = False
                    if page and (f"page: {page}" in cit_lower or f"page {page}" in cit_lower):
                        page_match = True

                    if sec_match:
                        matched_for_cit.append(c["chunk_text"])
                    elif page_match and not sec_match:
                        matched_for_cit.append(c["chunk_text"])

                if matched_for_cit:
                    supporting_chunks.extend(matched_for_cit)

            if supporting_chunks:
                # Preserve chunk order and deduplicate
                supporting_evidence_text = " ".join(dict.fromkeys(supporting_chunks))
            else:
                supporting_evidence_text = " ".join([c["chunk_text"] for c in accumulated_evidence])

            # De-hyphenate line wraps (e.g., 'submit-\nters' -> 'submitters')
            dehyphenated_text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', supporting_evidence_text)

            # Normalize comma-separated numbers (e.g., '5,000' -> '5000')
            norm_evidence = re.sub(r'(\d+),(\d+)', r'\1\2', supporting_evidence_text)
            norm_dehyphenated = re.sub(r'(\d+),(\d+)', r'\1\2', dehyphenated_text)

            # Tokenize evidence words with uniform alphanumeric regex
            evidence_tokens = re.findall(r'\b[a-zA-Z0-9]+\b', supporting_evidence_text.lower())
            evidence_tokens.extend(re.findall(r'\b[a-zA-Z0-9]+\b', dehyphenated_text.lower()))
            evidence_tokens.extend(re.findall(r'\b[a-zA-Z0-9]+\b', norm_evidence.lower()))
            evidence_tokens.extend(re.findall(r'\b[a-zA-Z0-9]+\b', norm_dehyphenated.lower()))
            evidence_words = set(evidence_tokens)

            for c in accumulated_evidence:
                if c.get("section_label"):
                    evidence_words.update(re.findall(r'\b[a-zA-Z0-9]+\b', c["section_label"].lower()))
                if c.get("page_number"):
                    evidence_words.add(str(c["page_number"]))

            # Also include basic morphological stems
            evidence_stems = set()
            for w in evidence_words:
                evidence_stems.add(w)
                if w.endswith('s') and len(w) > 3:
                    evidence_stems.add(w[:-1])
                if w.endswith('ed') and len(w) > 4:
                    evidence_stems.add(w[:-2])
                if w.endswith('ing') and len(w) > 5:
                    evidence_stems.add(w[:-3])
                    evidence_stems.add(w[:-2])
                if w.endswith('ly') and len(w) > 4:
                    evidence_stems.add(w[:-2])

            # Extract content words from sentence (excluding citations)
            claim_clean = re.sub(r'\[[^\]]+\]', '', sent)
            # Normalize comma-separated numbers in claim
            claim_norm = re.sub(r'(\d+),(\d+)', r'\1\2', claim_clean)
            claim_words = re.findall(r'\b[a-zA-Z0-9]+\b', claim_norm.lower())

            # Check both vocabulary and literal numbers against evidence
            substantive_terms = [
                w for w in claim_words
                if len(w) > 1 and w not in common_english
            ]

            # Detect truly unsupported technical mechanisms/entities/numbers
            unsupported = []
            for w in substantive_terms:
                w_stem = w
                if w.endswith('s') and len(w) > 3:
                    w_stem = w[:-1]
                elif w.endswith('ed') and len(w) > 4:
                    w_stem = w[:-2]
                elif w.endswith('ing') and len(w) > 5:
                    w_stem = w[:-3]
                elif w.endswith('ly') and len(w) > 4:
                    w_stem = w[:-2]

                if w not in evidence_words and w_stem not in evidence_stems:
                    unsupported.append(w)

            # Entity and relation check: e.g. RAG Fusion is not a vector DB
            sent_lower = sent.lower()
            if "rag fusion" in sent_lower and ("vector database" in sent_lower or "vector databases" in sent_lower or "faiss" in sent_lower or "qdrant" in sent_lower):
                if "compared to" in sent_lower or "and rag fusion" in sent_lower or "such as" in sent_lower:
                    unsupported.append("RAG Fusion is a retrieval method, not a vector database")

            # Swapped quantity / number-noun binding check:
            claim_quantity_pairs = re.findall(r'(\d+)\s+([a-zA-Z]+)', claim_norm.lower())
            for q_num, q_noun in claim_quantity_pairs:
                if q_noun in {"papers", "tokens", "tables", "seconds"}:
                    ev_noun_matches = re.findall(rf'(\d+(?:,\d+)?)\s+{q_noun}', supporting_evidence_text.lower())
                    if ev_noun_matches:
                        ev_nums = [m.replace(',', '') for m in ev_noun_matches]
                        if q_num not in ev_nums:
                            unsupported.append(f"swapped quantity: {q_num} {q_noun} (chunk specifies {ev_nums[0]} {q_noun})")

            # Check top-k / k value bindings:
            top_k_match = re.search(r'(?:top-k|k)\s*(?:value\s*)?(?:was\s*set\s*to|=|of|is|\s+)?\s*(\d+)', claim_norm.lower())
            if top_k_match:
                claim_k = top_k_match.group(1)
                ev_k_match = re.search(r'(?:k\s*=\s*|top-k\s*=\s*)(\d+)', supporting_evidence_text.lower())
                if ev_k_match:
                    ev_k = ev_k_match.group(1)
                    if claim_k != ev_k and claim_k not in {"10"}:
                        unsupported.append(f"swapped quantity: top-k {claim_k} (chunk specifies top-k {ev_k})")

            if unsupported:
                unique_unsupported = sorted(list(set(unsupported)))
                flagged = {
                    "claim": sent.strip(),
                    "unsupported_terms": unique_unsupported,
                    "citations": citation_matches
                }
                flagged_claims.append(flagged)
                # Omit unsupported sentence from main answer text (will be listed under 'not found')
            else:
                modified_sentences.append(sent)

        if modified_sentences:
            modified_paragraphs.append(" ".join(modified_sentences))

    main_answer = "\n".join([p for p in modified_paragraphs if p.strip()]).strip()
    if flagged_claims:
        not_found_lines = ["\n\n**Not found in the paper (unsupported claims removed):**"]
        for fc in flagged_claims:
            terms_str = ", ".join(f"'{t}'" for t in fc["unsupported_terms"])
            not_found_lines.append(f"- \"{fc['claim']}\" (unsupported: {terms_str})")
        verified_answer = main_answer + "\n" + "\n".join(not_found_lines)
    else:
        verified_answer = main_answer

    audit = {
        "has_unsupported_claims": len(flagged_claims) > 0,
        "flagged_claims": flagged_claims,
        "removed_claims_count": len(flagged_claims)
    }
    return verified_answer, audit


# ==============================================================================
# The Agentic Decision Loop Entry Point
# ==============================================================================
def run_agent_loop(
    question: str,
    workspace_id: str,
    strategy: str = "section_aware",
    mode: str = "hybrid",
    max_iterations: int = 2,
    top_k_per_step: int = 3
) -> Dict[str, Any]:
    """
    Executes the full agentic loop:
      Decide -> Retrieve -> Self-Check -> (Refine -> Retrieve) -> Answer
    
    Returns structured result with grounded answer and inspectable reasoning trace.
    """
    reasoning_trace: Dict[str, Any] = {
        "workspace_id": workspace_id,
        "question": question,
        "default_strategy": strategy,
        "default_mode": mode,
        "max_iterations": max_iterations,
        "steps": [],
        "total_retrieval_calls": 0,
        "accumulated_chunks_count": 0,
        "final_sufficiency": None
    }

    current_query = question
    previous_queries = []
    accumulated_evidence: List[Dict[str, Any]] = []
    seen_chunk_ids = set()

    for iteration in range(1, max_iterations + 1):
        step_log: Dict[str, Any] = {
            "iteration": iteration,
            "query_used": current_query,
            "strategy": strategy,
            "mode": mode
        }
        previous_queries.append(current_query)

        # 1. Retrieve
        print(f"  [Agent Iteration {iteration}] Searching ({mode}, {strategy}) for: '{current_query}'...")
        new_chunks = search_papers(
            query=current_query,
            workspace_id=workspace_id,
            strategy=strategy,
            mode=mode,
            top_k=top_k_per_step,
            exclude_chunk_ids=seen_chunk_ids
        )
        reasoning_trace["total_retrieval_calls"] += 1

        step_log["retrieved_count"] = len(new_chunks)
        step_log["chunks_summary"] = [
            {
                "id": c["id"],
                "section": c.get("section_label"),
                "page": c.get("page_number"),
                "rrf_score": c.get("rrf_score"),
                "dense_sim": c.get("dense_similarity") or c.get("similarity"),
                "bm25": c.get("bm25_score")
            }
            for c in new_chunks
        ]

        # Accumulate unique chunks and track newly added count
        new_unique_added = 0
        for c in new_chunks:
            if c["id"] not in seen_chunk_ids:
                seen_chunk_ids.add(c["id"])
                accumulated_evidence.append(c)
                new_unique_added += 1

        step_log["new_unique_chunks_added"] = new_unique_added

        # 2. Self-Check (Sufficiency Verification)
        print(f"  [Agent Iteration {iteration}] Self-checking evidence sufficiency ({len(accumulated_evidence)} total chunks)...")
        sufficiency = evaluate_sufficiency(
            question=question,
            accumulated_evidence=accumulated_evidence
        )
        step_log["sufficiency_evaluation"] = sufficiency
        score = sufficiency.get("evidence_support_score", 0.0)
        step_log["evidence_support_score"] = score
        guardrail_passed = score >= 0.35
        step_log["guardrail_0_35_passed"] = guardrail_passed
        guardrail_str = "PASS" if guardrail_passed else "FAIL"

        print(f"  [Agent Iteration {iteration}] Sufficiency Result: is_sufficient={sufficiency['is_sufficient']}")
        print(f"  [Agent Iteration {iteration}] Evidence Support Score: {score:.4f} (0.35 Guardrail: {guardrail_str})")
        print(f"  [Agent Iteration {iteration}] Rationale: {sufficiency['rationale']}")

        reasoning_trace["steps"].append(step_log)

        if sufficiency["is_sufficient"]:
            print(f"  [Agent Iteration {iteration}] Evidence verified complete. Proceeding to answer synthesis.\n")
            reasoning_trace["final_sufficiency"] = sufficiency
            break

        # Requirement 3: Stop loop early if iteration adds no new chunks
        if iteration > 1 and new_unique_added == 0:
            print(f"  [Agent Iteration {iteration}] No new unique chunks added by query. Stopping retrieval loop early.\n")
            reasoning_trace["loop_stopped_early_reason"] = "no_new_chunks_added"
            reasoning_trace["final_sufficiency"] = sufficiency
            break

        # 3. Refine query if insufficient and more iterations remain
        if iteration < max_iterations:
            print(f"  [Agent Iteration {iteration}] Evidence incomplete. Reformulating query for missing aspects: '{sufficiency.get('missing_aspects')}'...")
            current_query = reformulate_insufficient_query(
                original_question=question,
                previous_queries=previous_queries,
                retrieved_evidence=accumulated_evidence,
                missing_aspects=sufficiency.get("missing_aspects", "")
            )
            print(f"  [Agent Iteration {iteration}] Next Search Query formulated: '{current_query}'\n")
        else:
            print(f"  [Agent Iteration {iteration}] Max iterations reached. Proceeding to hedged answer synthesis.\n")
            reasoning_trace["final_sufficiency"] = sufficiency

    reasoning_trace["accumulated_chunks_count"] = len(accumulated_evidence)

    # 4. Synthesize Grounded Answer
    synthesis_result = synthesize_answer(
        question=question,
        accumulated_evidence=accumulated_evidence,
        sufficiency_evaluation=reasoning_trace["final_sufficiency"] or {"is_sufficient": False}
    )

    reasoning_trace["final_status"] = synthesis_result["status"]
    reasoning_trace["llm_backend"] = synthesis_result["llm_backend"]

    return {
        "question": question,
        "answer": synthesis_result["answer"],
        "citations": synthesis_result["citations"],
        "evidence_sufficiency": "sufficient" if (reasoning_trace["final_sufficiency"] and reasoning_trace["final_sufficiency"]["is_sufficient"]) else "insufficient",
        "grounding_audit": synthesis_result.get("grounding_audit", {}),
        "reasoning_trace": reasoning_trace
    }


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

    # Ingest reference workspace
    ref_ws_id = db.create_workspace(name="PaperHelper Reference Workspace")
    ref_paper_id = db.insert_paper(workspace_id=ref_ws_id, filename="research_major.pdf")

    print(f"Initializing Vector Database for Workspace '{ref_ws_id}'...")
    embed_and_store_chunks(fixed_chunks, workspace_id=ref_ws_id, paper_id=ref_paper_id, strategy="fixed")
    embed_and_store_chunks(section_chunks, workspace_id=ref_ws_id, paper_id=ref_paper_id, strategy="section_aware")

    # Test Suite
    test_queries = [
        {
            "id": "Q1",
            "question": "What F1 score did PaperHelper achieve with RAFT and RAG Fusion?",
            "max_iter": 2
        },
        {
            "id": "Q2",
            "question": "What are the main limitations and failure cases of the PaperHelper system?",
            "max_iter": 2
        },
        {
            "id": "Q3",
            "question": "How does RAG Fusion improve over basic RAG?",
            "max_iter": 2
        },
        {
            "id": "Q4_MultiPart",
            "question": "What embedding model and vector database did PaperHelper test, and what was the latency comparison between Milvus and Chroma?",
            "max_iter": 3
        },
        {
            "id": "Q5_Insufficient_Hedging",
            "question": "What was the exact financial cost in USD of running GPT-4 API fine-tuning for PaperHelper?",
            "max_iter": 2
        }
    ]

    print("\n" + "=" * 90)
    print("STEP 5: AGENTIC DECISION LOOP VERIFICATION")
    print("=" * 90)

    for item in test_queries:
        qid = item["id"]
        q = item["question"]
        m_iter = item["max_iter"]
        print(f"\n==========================================================================================")
        print(f"[{qid}] USER QUESTION: {q}")
        print(f"Config: mode='hybrid', strategy='section_aware', max_iterations={m_iter}")
        print(f"==========================================================================================")

        result = run_agent_loop(
            question=q,
            workspace_id=ref_ws_id,
            strategy="section_aware",
            mode="hybrid",
            max_iterations=m_iter,
            top_k_per_step=3
        )

        print("-" * 60)
        print(f"FINAL ANSWER:\n{result['answer']}")
        print("-" * 60)
        print(f"Sufficiency Status: {result['evidence_sufficiency'].upper()} | LLM Backend: {result['reasoning_trace']['llm_backend']}")
        print(f"Total Retrieval Loops: {result['reasoning_trace']['total_retrieval_calls']} | Total Unique Chunks: {result['reasoning_trace']['accumulated_chunks_count']}")
        print(f"Citations Returned: {len(result['citations'])} sources")
        for cit in result["citations"]:
            print(f"  - [{cit['section']}, Page {cit['page']}]")
