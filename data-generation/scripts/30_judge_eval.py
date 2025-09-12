import argparse, json, yaml
from pathlib import Path
from qna_core.prompt_loader import load_builder
from qna_core.endpoints import MultiEndpointChat
from qna_core.validate import parse_first_json

def load_chunks_map(processed_dir):
    mp = {}
    for line in open(processed_dir/"chunks.jsonl", "r", encoding="utf-8"):
        if line.strip():
            obj = json.loads(line)
            mp[obj["chunk_id"]] = obj
    return mp

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    paths = cfg["paths"]
    judge_cfg = cfg["judge"]
    gen_dir = Path(paths["generated_dir"])
    proc_dir = Path(paths["processed_dir"])
    chunks_map = load_chunks_map(proc_dir)

    build = load_builder(f"{paths['prompts_pkg']}.{judge_cfg['module']}")
    chat = MultiEndpointChat(judge_cfg["endpoint"], None, "fallback")  # judge uses single endpoint

    inp = gen_dir/"qna_high_confidence_validated.jsonl"
    outp = gen_dir/"qna_validated.jsonl"
    rej = gen_dir/"rejects.jsonl"

    fo = open(outp, "w", encoding="utf-8")
    fr = open(rej, "w", encoding="utf-8")

    thr = judge_cfg.get("thresholds", {})
    # Backward/forward-compatible thresholds with sensible defaults
    thr_domain = int(thr.get("domain_relevance", thr.get("DomainRelevance", 4)))
    thr_fact = int(thr.get("factuality", thr.get("Factuality", 4)))
    thr_sem = int(thr.get("semantic_similarity", thr.get("SemanticSimilarity", 4)))
    thr_comp = int(thr.get("completeness", thr.get("Completeness", 4)))
    thr_overall = float(thr.get("overall", 4))
    thr_citations = int(thr.get("citations_valid", thr.get("CitationsValid", 1)))
    thr_iss_max = float(thr.get("invented_specifics_max", thr.get("InventedSpecificsScoreMax", 1.0)))
    require_disclaimer_for_domain_ok = bool(thr.get("require_disclaimer_for_domain_ok", True))
    for line in open(inp, "r", encoding="utf-8"):
        if not line.strip(): continue
        obj = json.loads(line)
        # Do not reject based on step-1 JSON compliance here; proceed to judge everything
        # Prepare judge inputs
        # Parse original generator JSON to recover fields even if step-1 failed JSON rules
        ans_obj = parse_first_json(obj.get("answer_raw", "")) or {}
        answer_text = obj.get("answer", ans_obj.get("answer", ""))
        # Prefer citations from step-1; if absent, fall back to parsed answer JSON; if still empty, use retrieval context
        cids = obj.get("citations") or ans_obj.get("citations") or []
        if not cids:
            cids = obj.get("context_chunk_ids", [])
        cited = [chunks_map[c] for c in cids if c in chunks_map]
        msgs = build(obj["question"], answer_text, cited)
        resp = chat.chat(judge_cfg["model"], msgs, max_tokens=1024, temperature=0.0)
        
        # Try multiple JSON extraction strategies
        j = None
        
        # Strategy 1: Try to parse the entire response as JSON
        try:
            j = json.loads(resp.strip())
        except:
            pass
        
        # Strategy 2: Extract JSON between first { and last }
        if j is None:
            try:
                start = resp.find("{")
                end = resp.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = resp[start:end]
                    j = json.loads(json_str)
            except:
                pass
        
        # Strategy 3: Look for JSON after common prefixes
        if j is None:
            for prefix in ["```json", "```", "JSON:", "json:", "{", "Response:"]:
                try:
                    if prefix in resp:
                        start_idx = resp.find(prefix) + len(prefix)
                        remaining = resp[start_idx:].strip()
                        if remaining.startswith("{"):
                            end_idx = remaining.rfind("}") + 1
                            if end_idx > 0:
                                json_str = remaining[:end_idx]
                                j = json.loads(json_str)
                                break
                except:
                    continue
        
        # Strategy 4: Try to extract JSON using regex
        if j is None:
            import re
            try:
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(json_pattern, resp)
                for match in matches:
                    try:
                        j = json.loads(match)
                        break
                    except:
                        continue
            except:
                pass
        
        # If all strategies fail, reject with parse error
        if j is None:
            print(json.dumps({"reason":"judge_parse_error","raw":resp[:500], **obj}, ensure_ascii=False), file=fr)
            continue
        obj["judge"] = j

        # Backward compatibility: map old fields if present
        if all(k in j for k in ("faithfulness","completeness","specificity","style")):
            overall = (j.get("faithfulness",0)+j.get("completeness",0)+j.get("specificity",0)+j.get("style",0))/4.0
            ok = (
                int(j.get("citations_valid", j.get("CitationsValid", 0))) >= thr_citations and
                j.get("faithfulness",0) >= thr_fact and
                j.get("completeness",0) >= thr_comp and
                j.get("specificity",0) >= thr_domain and
                overall >= thr_overall
            )
        else:
            # New framework with weighted scoring
            label = (j.get("label") or "").upper()
            domain = int(j.get("DomainRelevance", 0))
            factual = int(j.get("Factuality", 0))
            semsim = int(j.get("SemanticSimilarity", 0))
            comp = int(j.get("Completeness", 0))
            iss = float(j.get("InventedSpecificsScore", 0.0) or 0.0)
            citations_ok = int(j.get("CitationsValid", j.get("citations_valid", 0)))
            contradiction = (j.get("Contradiction","no").lower() == "yes")
            disclaimer = (j.get("DisclaimerPresent","n/a").lower())
            
            # Calculate weighted composite score (out of 100)
            # Domain relevance: 20%, Factuality: 20%, Semantic similarity: 20%, Overall: 20%
            # Completeness: 10%, Citations valid: 10%
            # Invented specifics penalty (max -1 point)
            
            # Normalize scores to 0-100 scale (assuming original scores are 1-5)
            domain_score = max(0, (domain - 1) / 4 * 100) if domain > 0 else 0
            factual_score = max(0, (factual - 1) / 4 * 100) if factual > 0 else 0
            semsim_score = max(0, (semsim - 1) / 4 * 100) if semsim > 0 else 0
            comp_score = max(0, (comp - 1) / 4 * 100) if comp > 0 else 0
            
            # Overall score (average of domain, factual, semsim, comp)
            overall_raw = (domain + factual + semsim + comp) / 4.0
            overall_score = max(0, (overall_raw - 1) / 4 * 100) if overall_raw > 0 else 0
            
            # Citations score (binary: 100 if valid, 0 if not)
            citations_score = 100 if citations_ok >= thr_citations else 0
            
            # Calculate weighted composite score
            composite_score = (
                domain_score * 0.20 +      # 20%
                factual_score * 0.20 +     # 20%
                semsim_score * 0.20 +      # 20%
                overall_score * 0.20 +     # 20%
                comp_score * 0.10 +        # 10%
                citations_score * 0.10     # 10%
            )
            
            # Apply invented specifics penalty (max -1 point)
            iss_penalty = min(1.0, max(0, iss)) if iss > thr_iss_max else 0
            composite_score = max(0, composite_score - iss_penalty)
            
            # Store composite score in judge object
            j["composite_score"] = round(composite_score, 2)
            
            # Apply 70% threshold for acceptance
            score_threshold = 70.0
            ok = (
                (label in {"DOC_SUPPORTED","DOMAIN_OK"}) and
                not contradiction and
                composite_score >= score_threshold and
                (not require_disclaimer_for_domain_ok or label != "DOMAIN_OK" or disclaimer == "yes")
            )
            
            # Keep original overall for backward compatibility
            overall = overall_raw
        if ok:
            print(json.dumps(obj, ensure_ascii=False), file=fo)
        else:
            reason = {
                "reason": "judge_reject",
                "overall": overall,
                "label": j.get("label"),
                "citations_ok": j.get("CitationsValid", j.get("citations_valid", 0)),
                "composite_score": j.get("composite_score", 0),
            }
            print(json.dumps({**reason, **obj}, ensure_ascii=False), file=fr)

    fo.close(); fr.close()
    print("Wrote:", outp, "&", rej)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
