# scripts/utils/model_store.py
import os, re
from pathlib import Path
from typing import Optional, Dict, Any

def _safe_dir_name(repo_id: str) -> str:
    return re.sub(r"[/:@#]+", "-", repo_id).strip("-")

def _has_required_files(local_dir: Path) -> bool:
    # minimal check; different repos vary, so be permissive
    must_have = ["config.json", "tokenizer_config.json"]
    return all((local_dir / f).exists() for f in must_have)

def prepare_local_model_dir(cfg_model: Dict[str, Any], hf_token: Optional[str] = None) -> str:
    """
    Returns a path to a local directory that contains the model.
    If cfg_model['name'] is a path, we trust it.
    If it's a repo_id, we snapshot_download into cfg_model['local_dir'] if missing.
    """
    name = cfg_model["name"]
    trust_remote_code = bool(cfg_model.get("trust_remote_code", False))
    revision = cfg_model.get("revision", None)

    # If user passed a local folder, just use it
    if os.path.isdir(name):
        return name

    # Else treat as repo_id, download once
    from huggingface_hub import snapshot_download

    target = cfg_model.get("local_dir") or f"models/{_safe_dir_name(name)}"
    target_path = Path(target)
    target_path.mkdir(parents=True, exist_ok=True)

    if not _has_required_files(target_path):
        from huggingface_hub import snapshot_download

        kwargs = dict(
            repo_id=name,
            revision=revision,
            local_dir=target_path,
            local_dir_use_symlinks=False,
            token=hf_token,
        )

        # Prefer modern param name (glob patterns). Fall back gracefully.
        try:
            # skip markdown & large safetensors if you want to be lean; adjust as needed
            kwargs["ignore_patterns"] = ["*.md", "README*", "LICENSE*"]
            snapshot_download(**kwargs)
        except TypeError:
            # Old hub versions: no ignore_patterns; try without
            kwargs.pop("ignore_patterns", None)
            snapshot_download(**kwargs)
    return str(target_path)
