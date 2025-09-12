def build_messages(question, answer, cited_chunks):
    sys = (
        "You are a STRICT and rigorous evaluator for DPIP Q&A pairs. "
        "Apply harsh but fair standards - most answers have significant flaws. "
        "Perfect scores (5) should be EXTREMELY rare - only for exceptional answers. "
        "Good scores (4) should be uncommon - only for genuinely strong answers. "
        "Most answers should score 2-3 due to common issues like incomplete coverage, "
        "imprecise language, missing context, or superficial treatment. "
        "Be critical and identify specific weaknesses. Respond ONLY with valid JSON."
    )
    
    cited = "\n".join(f"[{c['chunk_id']}] {c['text']}" for c in cited_chunks)
    
    user = (
        f"QUESTION: {question}\n"
        f"ANSWER: {answer}\n\n"
        f"CITED CHUNKS:\n{cited}\n\n"
        "Evaluate this Q&A pair and respond with ONLY this JSON format:\n"
        '{\n'
        '  "label": "DOC_SUPPORTED|DOMAIN_OK|UNSUPPORTED",\n'
        '  "DomainRelevance": 1-5,\n'
        '  "Factuality": 1-5,\n'
        '  "SemanticSimilarity": 1-5,\n'
        '  "Completeness": 1-5,\n'
        '  "CitationsValid": 0-1,\n'
        '  "InventedSpecificsScore": 0.0-2.0,\n'
        '  "Contradiction": "yes|no",\n'
        '  "DisclaimerPresent": "yes|no|n/a"\n'
        '}\n\n'
        "ANALYTICAL SCORING CRITERIA:\n\n"
        "DomainRelevance (1-5) [20% weight]:\n"
        "• 1: Off-topic or irrelevant\n"
        "• 2: Limited relevance, superficial treatment\n"
        "• 3: Relevant and addresses core domain concepts\n"
        "• 4: Strong domain focus with good technical depth\n"
        "• 5: Exceptional domain expertise (rare)\n\n"
        "Factuality (1-5) [20% weight]:\n"
        "• 1: Contains factual errors\n"
        "• 2: Some inaccuracies or imprecisions\n"
        "• 3: Generally accurate information\n"
        "• 4: Reliable and well-supported facts\n"
        "• 5: Completely precise and verified (rare)\n\n"
        "SemanticSimilarity (1-5) [20% weight]:\n"
        "• 1: Poor alignment with cited sources\n"
        "• 2: Some deviation from source meaning\n"
        "• 3: Reasonable alignment with sources\n"
        "• 4: Good fidelity to source material\n"
        "• 5: Perfect preservation of source meaning (rare)\n\n"
        "Completeness (1-5) [10% weight]:\n"
        "• 1: Significantly incomplete\n"
        "• 2: Missing several important aspects\n"
        "• 3: Covers main points adequately\n"
        "• 4: Comprehensive with minor gaps\n"
        "• 5: Exceptionally thorough coverage (rare)\n\n"
        "REALISTIC EVALUATION RULES:\n"
        "- CitationsValid [10% weight]: Binary score (0 or 1)\n"
        "- Calculate scores based on actual quality, not generosity\n"
        "- Score 5: Reserved for truly exceptional answers (rare)\n"
        "- Score 4: For genuinely good answers with strong quality\n"
        "- Score 3: For solid answers that meet requirements well\n"
        "- Score 2: For acceptable answers with some limitations\n"
        "- Score 1: For poor answers with significant issues\n"
        "- Most answers should score in the 3-4 range\n"
        "- Look for specific flaws that justify lower scores\n"
        "- Consider: Could this answer be improved? If yes, don't give perfect scores\n\n"
        "QUALITY FOCUS:\n"
        "- Reward: Accurate information, good use of citations, clear explanations\n"
        "- Only penalize: Serious factual errors, contradictions, completely off-topic responses\n"
        "- Be generous: If an answer provides value and is generally correct, score it well\n\n"
        "OTHER FIELDS:\n"
        "CitationsValid: 1 if answer is supported by cited chunks, 0 if not\n"
        "InventedSpecificsScore: 0.0 (no invented details) to 2.0 (major fabrication)\n"
        "Contradiction: 'yes' if answer contradicts cited information\n"
        "DisclaimerPresent: 'yes' if answer acknowledges limitations/uncertainty\n\n"
        "LABELS:\n"
        "DOC_SUPPORTED: Well-supported by citations, high quality\n"
        "DOMAIN_OK: Reasonable answer but not fully supported\n"
        "UNSUPPORTED: Poor support or contradicts sources\n\n"
        "Respond with ONLY the JSON object, no other text."
    )
    
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]
