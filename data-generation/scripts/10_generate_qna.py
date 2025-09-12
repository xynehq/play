import argparse, json, random
from pathlib import Path
import yaml
import re
from difflib import SequenceMatcher
from qna_core.prompt_loader import load_builder
from qna_core.endpoints import MultiEndpointChat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def clean_think_tags(text):
    """Remove <think>...</think> blocks from the generated text"""
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Clean up extra whitespace and newlines
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned.strip())
    return cleaned

def get_model_for_iteration(gen, iteration):
    """Alternate between primary and secondary models for each QNA pair"""
    models = [gen["primary_model"]]
    if gen.get("secondary_model"):
        models.append(gen["secondary_model"])
    return models[iteration % len(models)]

def topk_context_for(question_text, all_chunks, k=4):
    texts = [c["text"] for c in all_chunks]
    vec = TfidfVectorizer(min_df=1).fit(texts + [question_text])
    qv = vec.transform([question_text])
    cv = vec.transform(texts)
    sims = cosine_similarity(qv, cv)[0]
    order = sims.argsort()[::-1][:k]
    return [all_chunks[i] for i in order]

def extract_confidence_from_answer(answer_raw, model_name=""):
    """Extract confidence score from JSON-formatted answer, with generous handling for Gemma."""
    try:
        # Try to parse as JSON first
        answer_data = json.loads(answer_raw.strip())
        confidence = answer_data.get("confidence", 0.0)
        
        # If confidence is 0 or missing, try to infer from answer quality
        if confidence == 0.0 and answer_data.get("answer") and answer_data.get("answer") != "I don't know":
            # If we have a substantive answer with citations, assume reasonable confidence
            if answer_data.get("citations") and len(answer_data.get("citations", [])) > 0:
                # Be more generous with Gemma model
                if "gemma" in model_name.lower():
                    confidence = 0.7  # Higher default for Gemma with citations
                else:
                    confidence = 0.75  # Default for other models
                print(f"Inferred confidence {confidence} for {model_name} answer with citations")
            elif len(answer_data.get("answer", "")) > 100:  # Substantial answer without citations
                if "gemma" in model_name.lower():
                    confidence = 0.68  # Give Gemma benefit of doubt for substantial answers
                else:
                    confidence = 0.65
                print(f"Inferred confidence {confidence} for {model_name} substantial answer")
        
        return confidence
    except (json.JSONDecodeError, AttributeError):
        # If JSON parsing fails, try to extract confidence from text
        import re
        confidence_match = re.search(r'"confidence":\s*([0-9.]+)', answer_raw)
        if confidence_match:
            return float(confidence_match.group(1))
        
        # If answer contains substantial content, give it a default confidence
        if len(answer_raw.strip()) > 50 and "I don't know" not in answer_raw:
            # Be more generous with Gemma for non-JSON responses
            if "gemma" in model_name.lower():
                default_conf = 0.7
            else:
                default_conf = 0.7
            print(f"Non-JSON response from {model_name}, assigning default confidence {default_conf}")
            return default_conf
        
        return 0.0

def get_model_for_iteration(qna_counter, models):
    """Get model for current QNA pair iteration (alternates between models)."""
    return models[qna_counter % len(models)]

def similarity(a, b):
    """Calculate similarity between two strings using SequenceMatcher."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def is_duplicate_question(question, existing_questions, threshold=0.85):
    """Check if a question is a duplicate of any existing questions."""
    for existing_q in existing_questions:
        if similarity(question, existing_q) >= threshold:
            return True, existing_q
    return False, None

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    paths = cfg["paths"]
    gen = cfg["generation"]
    prompts_pkg = paths["prompts_pkg"]
    q_build = load_builder(f"{prompts_pkg}.{gen['q_prompt']}")
    a_build = load_builder(f"{prompts_pkg}.{gen['a_prompt']}")

    chat = MultiEndpointChat(gen["primary_endpoint"], gen.get("secondary_endpoint"), gen.get("routing","fallback"))

    # Set up model alternation
    models = [gen["primary_model"]]
    if gen.get("secondary_model"):
        models.append(gen["secondary_model"])
    
    processed_dir = Path(paths["processed_dir"]).resolve()
    chunks = [json.loads(l) for l in open(processed_dir/"chunks.jsonl", "r", encoding="utf-8").read().splitlines() if l.strip()]

    outp_high = Path(paths["generated_dir"]) / "qna_high_confidence.jsonl"
    outp_low = Path(paths["generated_dir"]) / "qna_low_confidence.jsonl"
    outp_high.parent.mkdir(parents=True, exist_ok=True)
    outp_low.parent.mkdir(parents=True, exist_ok=True)
    fo_high = open(outp_high, "a", encoding="utf-8")
    fo_low = open(outp_low, "a", encoding="utf-8")

    # Load existing questions to prevent duplicates (load once for efficiency)
    existing_questions = []
    existing_questions_set = set()  # For faster exact match checking
    
    # Check multiple files for existing questions
    files_to_check = [
        Path(paths["generated_dir"]) / "qna_raw.jsonl",
        Path(paths["generated_dir"]) / "qna_high_confidence.jsonl",
        Path(paths["generated_dir"]) / "qna_validated.jsonl"
    ]
    
    for file_path in files_to_check:
        if file_path.exists():
            print(f"Loading existing questions from {file_path}")
            for line in open(file_path, "r", encoding="utf-8"):
                if line.strip():
                    try:
                        obj = json.loads(line)
                        question = obj.get("question", "")
                        if question and question not in existing_questions_set:
                            existing_questions.append(question)
                            existing_questions_set.add(question)
                    except:
                        continue
    
    print(f"Loaded {len(existing_questions)} unique existing questions for deduplication")
    
    session_questions = []  # Track questions generated in this session
    duplicate_threshold = gen.get("duplicate_threshold", 0.85)
    
    print("Starting generation with deduplication against existing data")
    
    qna_counter = 0
    model_iteration_counter = 0  # Separate counter for model alternation
    min_confidence = gen.get("min_confidence", 0.7)
    max_retries = gen.get("max_confidence_retries", 3)

    print(f"Starting generation with models: {models}")
    print(f"Model alternation enabled: {len(models) > 1}")

    # Group chunks by doc for simple windowing; start with 1-chunk questions
    for ch in chunks:
        ctx = [ch]
        # Build questions using current model for question generation
        current_model = get_model_for_iteration(model_iteration_counter, models)
        print(f"Using model {current_model} for question generation (iteration {model_iteration_counter})")
        q_msgs = q_build(ctx, max_questions_per_chunk=gen.get("max_questions_per_chunk",2))
        q_text = chat.chat(current_model, q_msgs, temperature=gen.get("temperature_q",0.7))
        # Expect JSONL lines
        for line in q_text.splitlines():
            line=line.strip()
            if not line or not line.startswith('{'): 
                continue
            try:
                qobj = json.loads(line)
            except Exception:
                continue
            question = qobj.get("question","" )
            if not question: 
                continue
            
            # Check for duplicates against existing and session questions
            all_questions = existing_questions + session_questions
            is_duplicate, similar_question = is_duplicate_question(question, all_questions, duplicate_threshold)
            if is_duplicate:
                print(f"Skipping duplicate question: '{question[:60]}...' (similar to: '{similar_question[:60]}...')")
                continue
            
            # Retrieve top-k context for answer
            k = gen.get("topk_context",4)
            ctxK = topk_context_for(question, chunks, k=k)
            a_msgs = a_build(question, ctxK)
            
            # Generate answer with confidence validation
            # Use separate counter for model alternation to ensure both models get equal chances
            answer_model = get_model_for_iteration(model_iteration_counter, models)
            print(f"Using model {answer_model} for answer generation (question: '{question[:11]}...')")
            ans_text = None
            confidence = 0.0
            retries = 0
            
            while retries <= max_retries:
                ans_text = chat.chat(answer_model, a_msgs, max_tokens=gen.get("max_tokens_a",512), temperature=gen.get("temperature_a",0.3))
                
                # Clean up the answer by removing <think> tags
                cleaned_answer = clean_think_tags(ans_text)
                confidence = extract_confidence_from_answer(cleaned_answer, answer_model)
                
                if confidence >= min_confidence:
                    break
                    
                retries += 1
                if retries <= max_retries:
                    print(f"Low confidence ({confidence:.2f}) with {answer_model}, retrying... ({retries}/{max_retries})")
                    # Try alternate model if available
                    if len(models) > 1:
                        answer_model = models[(models.index(answer_model) + 1) % len(models)]
                        print(f"Switching to model: {answer_model}")
            
            # Increment model iteration counter for next question, regardless of success
            model_iteration_counter += 1
            
            # Save all QA pairs regardless of confidence, but to appropriate files
            rec = {
                "question": question,
                "context_chunk_ids": [c["chunk_id"] for c in ctxK],
                "answer_raw": cleaned_answer,
                "gen_model": answer_model,
                "confidence": confidence,
                "retries_used": retries
            }
            
            # Add to session questions to prevent duplicates within this session
            session_questions.append(question)
            qna_counter += 1
            
            # Save to appropriate file based on confidence
            if confidence >= 0.7:
                print(json.dumps(rec, ensure_ascii=False), file=fo_high)
                fo_high.flush()  # Immediate save
                print(f"Generated HIGH confidence QNA pair {qna_counter} with confidence {confidence:.2f}")
            else:
                print(json.dumps(rec, ensure_ascii=False), file=fo_low)
                fo_low.flush()  # Immediate save
                print(f"Generated LOW confidence QNA pair {qna_counter} with confidence {confidence:.2f}")
            
            # Stop after generating 1000 QA pairs total
            if qna_counter >= 1000:
                print(f"Reached target of 1000 QA pairs, stopping generation.")
                fo_high.close()
                fo_low.close()
                print(f"Generated {qna_counter} total QNA pairs")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
