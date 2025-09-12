import argparse, json, yaml, re
from pathlib import Path
from qna_core.validate import parse_first_json, lexical_overlap, semantic_sim

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
    val = cfg["validation"]
    gen_dir = Path(paths["generated_dir"])
    proc_dir = Path(paths["processed_dir"])
    chunks_map = load_chunks_map(proc_dir)

    inp = gen_dir/"qna_high_confidence.jsonl"
    outp = gen_dir/"qna_high_confidence_validated.jsonl"
    fo = open(outp, "w", encoding="utf-8")

    for line in open(inp, "r", encoding="utf-8"):
        if not line.strip(): 
            continue
        obj = json.loads(line)

        raw = obj.get("answer_raw","")
        # Strict JSON-only: allow surrounding whitespace, nothing else
        start = raw.find("{"); end = raw.rfind("}")
        if start < 0 or end < 0:
            obj["pass_step1"] = False
            obj["reject_reason"] = "no_json_found"
            print(json.dumps(obj, ensure_ascii=False), file=fo); 
            continue
        prefix = raw[:start].strip()
        suffix = raw[end+1:].strip()
        if prefix or suffix:
            obj["pass_step1"] = False
            obj["reject_reason"] = "non_json_output"
            print(json.dumps(obj, ensure_ascii=False), file=fo); 
            continue

        # Parse JSON
        try:
            ans = json.loads(raw[start:end+1])
        except Exception:
            obj["pass_step1"] = False
            obj["reject_reason"] = "json_parse_error"
            print(json.dumps(obj, ensure_ascii=False), file=fo); 
            continue

        # --- schema & provenance checks ---
        ok = True
        reason = None
        prov = ans.get("provenance", None)
        cits = ans.get("citations", [])
        conf = ans.get("confidence", None)

        if prov not in ("from_context", "extrapolated"):
            ok, reason = False, "bad_provenance"
        elif prov == "from_context" and ("disclaimer" in ans):
            ok, reason = False, "unexpected_disclaimer_for_from_context"
        elif prov == "extrapolated" and (not isinstance(ans.get("disclaimer",""), str) or not ans.get("disclaimer","").strip()):
            ok, reason = False, "missing_disclaimer_for_extrapolation"

        if not isinstance(cits, list) or len(cits) < 1:
            ok, reason = False, reason or "no_citations"

        if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
            ok, reason = False, reason or "bad_confidence"

        # Require at least one citation overlapping the candidate retrieval context if present
        ctx_ids = set(obj.get("context_chunk_ids", []))
        if ctx_ids and not (ctx_ids.intersection(set(cits))):
            ok, reason = False, reason or "no_context_citation_overlap"

        if not ok:
            obj["pass_step1"] = False
            obj["reject_reason"] = reason
            print(json.dumps(obj, ensure_ascii=False), file=fo)
            continue

        # Lexical & semantic checks
        cited_text = "\n".join(chunks_map[c]["text"] for c in cits if c in chunks_map)
        answer_text = ans.get("answer","")
        scores = {
            "lex_overlap": lexical_overlap(answer_text, cited_text) if cited_text else 0.0,
            "sem_sim": semantic_sim(answer_text, cited_text) if cited_text else 0.0,
        }

        obj["answer"] = answer_text
        obj["citations"] = cits
        obj["provenance"] = prov
        if "disclaimer" in ans:
            obj["disclaimer"] = ans["disclaimer"]
        obj["scores_step1"] = scores

        obj["pass_step1"] = (
            scores["lex_overlap"] >= val.get("lexical_min_overlap_ratio", 0.25) and
            scores["sem_sim"] >= val.get("semantic_min_cos_sim", 0.60)
        )

        print(json.dumps(obj, ensure_ascii=False), file=fo)

    fo.close()
    print("Wrote:", outp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
