def build_messages(
    context_chunks,
    max_questions_per_chunk=6,
    require_variety=True,
    allow_multi_chunk=True,
):
    """
    Produces JSONL: one question per line.
    Fields: question, target_chunk_ids, difficulty, intent, category
    """
    sys = (
        "You are a DPIP domain expert and curriculum designer. "
        "Your job is to create many high-quality questions that a NEW integrator, "
        "developer, SRE, or product manager would naturally ask when reading these docs. "
        "Think privately, then OUTPUT ONLY JSONL—one object per line."
    )

    ctx = "\n".join(f"[{c['chunk_id']}] {c['text']}" for c in context_chunks)
    user = (
        "CONTEXT CHUNKS:\n"
        f"{ctx}\n\n"
        "INSTRUCTIONS:\n"
        "1) Carefully study the context.\n"
        "2) Generate diverse, realistic questions:\n"
        "   - Intents: 'what_is', 'how_to', 'configuration', 'troubleshooting', 'constraints', 'examples'.\n"
        "   - Categories: 'getting_started', 'auth', 'api', 'deploy', 'ops', 'limits', 'billing', 'security'.\n"
        "   - Difficulties: easy, medium, hard. Balance them.\n"
        "3) Avoid duplicates/near-duplicates.\n"
        "4) Prefer questions grounded in these chunks. If a practical question needs TWO+ chunks, set target_chunk_ids accordingly.\n"
        "5) No answers here. ONLY questions as JSON lines.\n\n"
        "OUTPUT FORMAT (STRICT JSONL; one object per line):\n"
        '{"question":"...",\n'
        ' "target_chunk_ids":["chunk_id", ...],\n'
        ' "difficulty":"easy|medium|hard",\n'
        ' "intent":"what_is|how_to|configuration|troubleshooting|constraints|examples",\n'
        ' "category":"getting_started|auth|api|deploy|ops|limits|billing|security"}\n\n'
        f"Generate up to {max_questions_per_chunk} questions. Be diverse, useful, and realistic."
    )

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]
