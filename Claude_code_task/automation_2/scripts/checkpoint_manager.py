"""
Checkpoint/Resume Manager for PR Automation System

This module provides checkpoint detection and resume functionality to allow
interrupted runs to continue from the last completed attempt without re-execution.

Key Features:
- Detect last completed attempt from disk artifacts
- Track attempt status (in_progress, completed, failed)
- Enable idempotent resume operations
- Prevent re-computation of completed attempts
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class AttemptStatus:
    """Enum-like class for attempt states"""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


def detect_last_completed_attempt(run_dir: Path) -> int:
    """
    Scan attempt directories to find the last completed attempt.
    
    A completed attempt must have:
    - evaluation.json with valid verdict
    - status.json with state="completed" or "failed"
    
    Args:
        run_dir: Model-specific run directory (e.g., Data/runs/PR-9729/glm-latest)
    
    Returns:
        - 0 if no attempts completed
        - N if attempt N is the last completed attempt
    """
    last_completed = 0
    attempt_num = 1
    
    while True:
        attempt_dir = run_dir / f"p{attempt_num}"
        if not attempt_dir.exists():
            break
        
        # Check for both old and new naming conventions
        eval_file = attempt_dir / "eval.json"
        status_file = attempt_dir / "status.json"
        
        # An attempt is considered complete if:
        # 1. It has an evaluation file with a valid verdict
        # 2. It has a status file marking it as completed/failed
        if eval_file.exists():
            try:
                eval_data = json.loads(eval_file.read_text(encoding="utf-8"))
                verdict = eval_data.get("verdict")
                
                # If status.json exists, use it for definitive completion state
                if status_file.exists():
                    status_data = json.loads(status_file.read_text(encoding="utf-8"))
                    state = status_data.get("state")
                    
                    if state in [AttemptStatus.COMPLETED, AttemptStatus.FAILED]:
                        last_completed = attempt_num
                else:
                    # Backward compatibility: if eval.json exists with valid verdict,
                    # consider it completed (for runs before status.json was added)
                    if verdict in ["PASS", "FAIL", "EXECUTION_FAILED"]:
                        last_completed = attempt_num
            except (json.JSONDecodeError, KeyError):
                # Invalid or incomplete attempt - don't count it
                pass
        
        attempt_num += 1
    
    return last_completed


def load_attempt_evaluation(attempt_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load evaluation data from an attempt directory.
    
    Args:
        attempt_dir: Path to attempt directory
    
    Returns:
        Evaluation dict with 'verdict' and 'reason', or None if not found
    """
    eval_file = attempt_dir / "eval.json"
    
    if not eval_file.exists():
        return None
    
    try:
        return json.loads(eval_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_attempt_status(attempt_dir: Path, state: str):
    """
    Save attempt status atomically.
    
    Args:
        attempt_dir: Path to attempt directory
        state: One of AttemptStatus.IN_PROGRESS, COMPLETED, FAILED
    """
    status_file = attempt_dir / "status.json"
    
    status_data = {
        "state": state,
        "timestamp": datetime.now().isoformat()
    }
    
    # Atomic write: write to temp file then rename
    temp_file = status_file.with_suffix(".tmp")
    temp_file.write_text(json.dumps(status_data, indent=2), encoding="utf-8")
    temp_file.replace(status_file)


def mark_attempt_in_progress(attempt_dir: Path):
    """Mark an attempt as in-progress."""
    save_attempt_status(attempt_dir, AttemptStatus.IN_PROGRESS)


def mark_attempt_completed(attempt_dir: Path):
    """Mark an attempt as completed successfully."""
    save_attempt_status(attempt_dir, AttemptStatus.COMPLETED)


def mark_attempt_failed(attempt_dir: Path):
    """Mark an attempt as failed."""
    save_attempt_status(attempt_dir, AttemptStatus.FAILED)


def is_attempt_complete(attempt_dir: Path) -> bool:
    """
    Check if an attempt is complete (has evaluation and proper status).
    
    Args:
        attempt_dir: Path to attempt directory
    
    Returns:
        True if attempt is complete, False otherwise
    """
    status_file = attempt_dir / "status.json"
    eval_file = attempt_dir / "eval.json"
    
    if not eval_file.exists():
        return False
    
    if status_file.exists():
        try:
            status_data = json.loads(status_file.read_text(encoding="utf-8"))
            state = status_data.get("state")
            return state in [AttemptStatus.COMPLETED, AttemptStatus.FAILED]
        except (json.JSONDecodeError, OSError):
            return False
    
    # Backward compatibility: if eval.json exists, consider it complete
    return True


def get_resume_info(run_dir: Path) -> Dict[str, Any]:
    """
    Get comprehensive resume information for a run.
    
    Args:
        run_dir: Model-specific run directory
    
    Returns:
        Dict with resume information including:
        - last_completed_attempt: int (0 if none)
        - should_resume: bool
        - resume_from_attempt: int (next attempt to run)
        - last_verdict: str or None
    """
    last_completed = detect_last_completed_attempt(run_dir)
    
    resume_info = {
        "last_completed_attempt": last_completed,
        "should_resume": last_completed > 0,
        "resume_from_attempt": last_completed + 1,
        "last_verdict": None
    }
    
    if last_completed > 0:
        last_attempt_dir = run_dir / f"p{last_completed}"
        eval_data = load_attempt_evaluation(last_attempt_dir)
        
        if eval_data:
            resume_info["last_verdict"] = eval_data.get("verdict")
    
    return resume_info


def validate_attempt_artifacts(attempt_dir: Path, attempt_num: int) -> bool:
    """
    Validate that an attempt has all required artifacts.
    
    Args:
        attempt_dir: Path to attempt directory
        attempt_num: Attempt number (for logging)
    
    Returns:
        True if artifacts are valid, False otherwise
    """
    required_files = [
        f"p{attempt_num}_prompt.txt",
        "eval.json"
    ]
    
    for filename in required_files:
        filepath = attempt_dir / filename
        if not filepath.exists():
            return False
    
    # Validate eval.json has proper structure
    try:
        eval_data = json.loads((attempt_dir / "eval.json").read_text(encoding="utf-8"))
        if "verdict" not in eval_data:
            return False
    except (json.JSONDecodeError, OSError):
        return False
    
    return True
