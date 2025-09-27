# scripts/utils/model_store.py
import os, re
from pathlib import Path
from typing import Optional, Dict, Any

def _safe_dir_name(repo_id: str) -> str:
    return re.sub(r"[/:@#]+", "-", repo_id).strip("-")

def _has_required_files(local_dir: Path) -> bool:
    # Check for both config files AND model weight files
    config_files = ["config.json", "tokenizer_config.json"]
    model_files = [
        "pytorch_model.bin",           # Full PyTorch model
        "model.safetensors",          # Single safetensors file
        "model.safetensors.index.json", # Sharded safetensors index
        "tf_model.h5",                # TensorFlow model
        "model.ckpt.index",           # TensorFlow checkpoint
        "flax_model.msgpack"          # Flax model
    ]
    
    # Must have all config files
    has_configs = all((local_dir / f).exists() for f in config_files)
    
    # Must have at least one model weight file
    has_model = any((local_dir / f).exists() for f in model_files)
    
    return has_configs and has_model

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
        print(f"[model_store] Model files missing in {target_path}, downloading from {name}...")
        
        # Clean up any existing cache directory that might interfere
        cache_dir = target_path / ".cache"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
        
        kwargs = dict(
            repo_id=name,
            revision=revision,
            local_dir=target_path,
            local_dir_use_symlinks=False,
            token=hf_token,
            # Don't use cache to ensure files go directly to target_dir
            cache_dir=None,
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
        
        # Debug: list what was actually downloaded
        print(f"[model_store] Files in {target_path} after download:")
        for item in target_path.iterdir():
            if item.is_file():
                print(f"  - {item.name} ({item.stat().st_size / (1024*1024):.1f}MB)")
            else:
                print(f"  - {item.name}/ (directory)")
    else:
        print(f"[model_store] Model files found in {target_path}")
    
    return str(target_path)
