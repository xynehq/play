# scripts/ingest_docx.py
import os, glob, re, json
from pathlib import Path
from typing import List
from docx import Document

RAW_DIR = "data/raw"
OUT_JSONL = "data/processed/dpip_cpt.jsonl"
os.makedirs("data/processed", exist_ok=True)

def clean_text(t: str) -> str:
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def docx_to_txt(docx_path: str) -> str:
    doc = Document(docx_path)
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return clean_text("\n\n".join(parts))

def naive_paragraph_chunk(text: str, max_chars=1800, min_chars=600) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf, size = [], [], 0
    for p in paras:
        if size + len(p) > max_chars and size >= min_chars:
            chunks.append("\n\n".join(buf)); buf, size = [], 0
        buf.append(p); size += len(p) + 2
    if buf: chunks.append("\n\n".join(buf))
    return chunks

def main():
    # 1) docx -> txt
    docx_files = glob.glob(os.path.join(RAW_DIR, "*.docx"))
    print(f"Processing {len(docx_files)} DOCX files...")
    
    for p in docx_files:
        try:
            txt = docx_to_txt(p)
            out_txt = Path(p).with_suffix(".txt")
            out_txt.write_text(txt, encoding="utf-8")
            print(f"Wrote {out_txt}")
        except Exception as e:
            print(f"Error processing {p}: {e}")

    # 2) txt -> cpt jsonl
    count = 0
    total_chars = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for p in glob.glob(os.path.join(RAW_DIR, "*.txt")):
            title = Path(p).stem
            text = clean_text(Path(p).read_text(encoding="utf-8", errors="ignore"))
            for ch in naive_paragraph_chunk(text):
                if len(ch) < 200: 
                    continue
                tagged = f"<dpip_doc>\nTITLE: {title}\nBODY:\n{ch}"
                f.write(json.dumps({"text": tagged}, ensure_ascii=False) + "\n")
                count += 1
                total_chars += len(tagged)
    
    avg_len = total_chars // count if count > 0 else 0
    print(f"Wrote {count} CPT chunks -> {OUT_JSONL}")
    print(f"Average chunk length: {avg_len} characters")

if __name__ == "__main__":
    main()
