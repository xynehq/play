import argparse, json, yaml
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def dedup_by_similarity(items, key=lambda x: x, threshold=0.92):
    if not items: return []
    texts = [key(x) for x in items]
    vec = TfidfVectorizer(min_df=1).fit(texts)
    X = vec.transform(texts)
    keep = []
    seen = set()
    for i,t in enumerate(texts):
        if i in seen: continue
        sims = cosine_similarity(X[i], X)[0]
        dup_idx = [j for j,s in enumerate(sims) if s>=threshold]
        for j in dup_idx: seen.add(j)
        keep.append(items[i])
    return keep

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    paths = cfg["paths"]
    exp = cfg["export"]
    gen_dir = Path(paths["generated_dir"])
    sft_dir = Path(paths["processed_dir"]).parent/"sft"
    sft_dir.mkdir(parents=True, exist_ok=True)

    data = [json.loads(l) for l in open(gen_dir/"qna_validated.jsonl", "r", encoding="utf-8") if l.strip()]
    # Dedup by question
    data = dedup_by_similarity(data, key=lambda x: x["question"], threshold=exp.get("dedup_near_threshold",0.92))
    # Cap per doc if desired
    bydoc = {}
    for d in data:
        doc_id = (d.get("citations") or ["unknown"]) [0].split("_")[0]
        bydoc.setdefault(doc_id, []).append(d)
    max_per = exp.get("max_per_doc", 300)
    final = []
    for doc_id, arr in bydoc.items():
        final.extend(arr[:max_per])

    out = sft_dir/"train.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for it in final:
            ctx_lines = []
            # include top-k context chunks
            # (optional; minimal approach keeps only cited chunk snippets)
            ctx_lines.append("CITED CONTEXT:")
            for cid in it.get("citations", []):
                ctx_lines.append(f"[{cid}] (excerpt) ...")
            user_content = it["question"] + "\n\n" + "\n".join(ctx_lines)
            assistant = it.get("answer","" ) + "\n\nCitations: " + ", ".join(it.get("citations",[]))
            rec = {
                "messages": [
                    {"role":"system","content":"You are a DPIP assistant. Use provided context only."},
                    {"role":"user","content": user_content},
                    {"role":"assistant","content": assistant},
                ]
            }
            if cfg["export"].get("keep_scores", True):
                rec["meta"] = {
                    "model": it.get("gen_model",""),
                    "scores_step1": it.get("scores_step1",{}),
                    "judge": it.get("judge",{})
                }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Wrote:", out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
