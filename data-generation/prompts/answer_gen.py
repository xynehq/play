def build_messages(
    question,
    context_chunks,
    allow_extrapolation=True,
    min_citations=1,
):
    """
    STRICT JSON ONLY output, with fields:
      - answer: string (detailed; or exactly "I don't know")
      - citations: list[str] (len >= min_citations)
      - confidence: float in [0,1]
      - provenance: "from_context" | "extrapolated"
      - disclaimer: REQUIRED IFF provenance=="extrapolated"; MUST be omitted for "from_context"
    """
    sys = (
        "You are a knowledgeable DPIP expert assistant. "
        "IMPORTANT: You MUST respond with ONLY a valid JSON object. "
        "No text before or after the JSON. NO code blocks. NO explanations. "
        "Be confident in your answers when you can provide helpful information from the context. "
        "Use confidence 0.75-0.9 for good answers, 0.7-0.75 for reasonable answers. "
        "Example format: {\"answer\": \"...\", \"citations\": [...], \"confidence\": 0.75, \"provenance\": \"from_context\"}"
    )

    ctx = "\n".join(f"[{c['chunk_id']}] {c['text']}" for c in context_chunks)

    rules = [
        f"- Use at least {min_citations} citation(s) referencing the provided chunk_ids.",
        "- Quote key parameter names exactly where relevant.",
        "- Cover steps, constraints, edge-cases, and examples if present.",
        "- If part of the question is not directly in context but strongly implied, you MAY answer via disciplined extrapolation only if it is technically sound and aligned with the text.",
        "- If extrapolating, set provenance='extrapolated', include a short non-empty 'disclaimer', and STILL include the most relevant chunk_ids that justify your inference.",
        "- If you cannot answer from context and strong inference, respond exactly 'I don't know'.",
        "- CONFIDENCE GUIDANCE: Be generous with confidence when you provide helpful answers",
        "- If context supports the answer well, use confidence 0.75-1",
        "- If context partially supports the answer, use confidence 0.7-0.75",
        "- Only use low confidence (<0.6) if you're very uncertain",
        "- Absolutely NO hallucinations or invented APIs/values. Be precise and complete.",
        "- If provenance == 'from_context': DO NOT include the 'disclaimer' key at all.",
        "- If provenance == 'extrapolated': you MUST include a non-empty 'disclaimer'.",
    ]

    schema = (
        "OUTPUT (STRICT SINGLE JSON OBJECT):\n"
        "{\n"
        '  "answer": "string (detailed; or exactly: I don\'t know)",\n'
        '  "citations": ["chunk_id", ...],\n'
        '  "confidence": 0.0-1.0,\n'
        '  "provenance": "from_context" | "extrapolated"\n'
        "  // If provenance=='extrapolated', ALSO include ONLY THEN:\n"
        '  // "disclaimer": "Using own intelligence based on DPIP context; for clarity contact support."\n'
        "}\n"
        "Do not include any additional keys. Do not output code fences or tags. JSON only."
    )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT CHUNKS:\n{ctx}\n\n"
        "ANSWERING RULES:\n" + "\n".join(rules) + "\n\n" + schema
    )

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]
