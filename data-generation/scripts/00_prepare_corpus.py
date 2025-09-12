import argparse, json, os
from pathlib import Path
from qna_core.loaders import read_doc
from qna_core.chunking import split_into_chunks, naive_token_count
import yaml, uuid

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    paths = cfg["paths"]
    chunking = cfg["chunking"]
    docs_dir = Path(paths["docs_dir"])
    processed_dir = Path(paths["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    docs_out = open(processed_dir/"documents.jsonl", "w", encoding="utf-8")
    chunks_out = open(processed_dir/"chunks.jsonl", "w", encoding="utf-8")

    for p in sorted(docs_dir.glob("*")):
        if p.suffix.lower() not in {".pdf",".docx",".txt",".md"}:
            continue
        doc_id = p.stem
        text = read_doc(p)
        print(f"Loaded {p.name} ({len(text)} chars)" )
        print(json.dumps({"doc_id":doc_id, "title":p.name, "text":text[:1200]+("..." if len(text)>1200 else "")}, ensure_ascii=False), file=docs_out)
        chunks = split_into_chunks(text, chunking.get("target_tokens",380), chunking.get("overlap_tokens",60))
        for idx, ch in enumerate(chunks):
            rec = {"chunk_id": f"{doc_id}_{idx:04d}", "doc_id": doc_id, "idx": idx, "text": ch, "tokens": naive_token_count(ch)}
            print(json.dumps(rec, ensure_ascii=False), file=chunks_out)

    docs_out.close(); chunks_out.close()
    print("Wrote:", processed_dir/"documents.jsonl", processed_dir/"chunks.jsonl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
