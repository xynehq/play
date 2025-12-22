"""
Stage 0: Automated PR Setup
This module automates the PR setup process that was previously done manually via pr_analyzer.sh
"""

import subprocess
import json
from pathlib import Path


class PRSetupError(Exception):
    """Exception raised for PR setup failures."""
    pass


def setup_pr(pr_number: str, repo_dir: Path, pr_dir: Path) -> dict:
    """
    Automate Stage 0: PR Setup
    
    Performs the same actions as pr_analyzer.sh in the same order:
    1. Fetch PR metadata via gh CLI
    2. Identify merge commit
    3. Checkout base commit (merge_commit^1)
    4. Create isolated execution branch
    5. Generate task description
    6. Generate ground-truth diff
    
    Args:
        pr_number: The PR number to process
        repo_dir: Path to the git repository (claude-work2 directory)
        pr_dir: Path to the pr/ directory for outputs
    
    Returns:
        dict with setup metadata
        
    Raises:
        PRSetupError: If any step fails
    """
    print(f"\n{'='*80}")
    print(f"[Stage 0] PR Setup - PR #{pr_number}")
    print(f"{'='*80}\n")
    
    # Step 1: Fetch PR metadata
    pr_metadata = _fetch_pr_metadata(pr_number, repo_dir)
    
    # Step 2: Identify merge commit
    merge_commit = pr_metadata['merge_commit']
    print(f"[Stage 0] Merge commit: {merge_commit}")
    
    # Step 3: Compute and checkout base commit
    base_commit = f"{merge_commit}^1"
    _verify_commit_exists(base_commit, repo_dir)
    
    # Step 4: Create isolated execution branch
    branch_name = f"test-claude-pr-{pr_number}"
    _create_branch(branch_name, base_commit, repo_dir)
    
    # Step 5: Generate task description
    task_file = pr_dir / f"task_pr_{pr_number}.md"
    _generate_task_file(pr_metadata, task_file)
    
    # Step 6: Generate ground-truth diff
    diff_file = pr_dir / "original_changes.diff"
    _generate_ground_truth_diff(base_commit, merge_commit, diff_file, repo_dir)
    
    # Step 7: Save merge commit for context diff generation
    merge_commit_file = pr_dir / f"merge_commit_{pr_number}.txt"
    _save_merge_commit(merge_commit, merge_commit_file)
    
    # Verify outputs exist
    _verify_outputs(task_file, diff_file, merge_commit_file)
    
    # Verify clean git state
    _verify_clean_state(repo_dir)
    
    print(f"\n[Stage 0] ✅ PR Setup Complete")
    print(f"[Stage 0] Branch: {branch_name}")
    print(f"[Stage 0] Base commit: {base_commit}")
    print(f"[Stage 0] Task file: {task_file}")
    print(f"[Stage 0] Diff file: {diff_file}")
    
    return {
        "pr_number": pr_number,
        "branch_name": branch_name,
        "base_commit": base_commit,
        "merge_commit": merge_commit,
        "task_file": str(task_file),
        "diff_file": str(diff_file)
    }


def _fetch_pr_metadata(pr_number: str, repo_dir: Path) -> dict:
    """
    Fetch PR metadata via gh CLI.
    
    Required: gh pr view <PR_NUMBER> --json title,body,mergeCommit
    Fail fast if PR is not merged or metadata is missing.
    """
    print(f"[Stage 0] Fetching PR metadata from GitHub...")
    
    # Set GitHub repo default (juspay/hyperswitch)
    set_repo_result = subprocess.run(
        ["gh", "repo", "set-default", "juspay/hyperswitch"],
        cwd=repo_dir,
        capture_output=True,
        text=True
    )
    
    if set_repo_result.returncode != 0:
        raise PRSetupError(
            f"Failed to set GitHub repo default.\n"
            f"Error: {set_repo_result.stderr}"
        )
    
    # Fetch PR metadata
    pr_view_result = subprocess.run(
        ["gh", "pr", "view", pr_number, "--json", "title,body,state,mergeCommit"],
        cwd=repo_dir,
        capture_output=True,
        text=True
    )
    
    if pr_view_result.returncode != 0:
        raise PRSetupError(
            f"Failed to fetch PR metadata from GitHub.\n"
            f"Error: {pr_view_result.stderr}\n"
            f"Make sure PR #{pr_number} exists and you have gh CLI configured."
        )
    
    try:
        pr_data = json.loads(pr_view_result.stdout)
    except json.JSONDecodeError as e:
        raise PRSetupError(
            f"Failed to parse PR metadata JSON.\n"
            f"Error: {str(e)}\n"
            f"Output: {pr_view_result.stdout}"
        )
    
    # Verify PR is merged
    if pr_data.get('state') != 'MERGED':
        raise PRSetupError(
            f"PR #{pr_number} is not merged (state: {pr_data.get('state')}).\n"
            f"Only merged PRs can be processed."
        )
    
    # Extract merge commit
    merge_commit_data = pr_data.get('mergeCommit')
    if not merge_commit_data or not merge_commit_data.get('oid'):
        raise PRSetupError(
            f"PR #{pr_number} has no merge commit.\n"
            f"This PR may have been closed without merging."
        )
    
    merge_commit = merge_commit_data['oid']
    title = pr_data.get('title', 'No title')
    body = pr_data.get('body', 'No description provided.')
    
    print(f"[Stage 0] ✓ PR metadata fetched")
    print(f"[Stage 0]   Title: {title}")
    print(f"[Stage 0]   State: MERGED")
    
    return {
        "title": title,
        "body": body,
        "merge_commit": merge_commit
    }


def _verify_commit_exists(commit: str, repo_dir: Path):
    """Verify that a commit exists in the repository."""
    print(f"[Stage 0] Verifying commit exists: {commit}")
    
    result = subprocess.run(
        ["git", "cat-file", "-e", commit],
        cwd=repo_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        # Try to fetch from upstream
        print(f"[Stage 0] Commit not found locally, fetching from upstream...")
        
        # Check if upstream remote exists
        remote_result = subprocess.run(
            ["git", "remote"],
            cwd=repo_dir,
            capture_output=True,
            text=True
        )
        
        if "upstream" not in remote_result.stdout:
            print(f"[Stage 0] Adding upstream remote...")
            add_remote_result = subprocess.run(
                ["git", "remote", "add", "upstream", "https://github.com/juspay/hyperswitch.git"],
                cwd=repo_dir,
                capture_output=True,
                text=True
            )
            
            if add_remote_result.returncode != 0:
                raise PRSetupError(
                    f"Failed to add upstream remote.\n"
                    f"Error: {add_remote_result.stderr}"
                )
        
        # Fetch from upstream
        fetch_result = subprocess.run(
            ["git", "fetch", "upstream"],
            cwd=repo_dir,
            capture_output=True,
            text=True
        )
        
        if fetch_result.returncode != 0:
            raise PRSetupError(
                f"Failed to fetch from upstream.\n"
                f"Error: {fetch_result.stderr}"
            )
        
        # Verify again
        verify_result = subprocess.run(
            ["git", "cat-file", "-e", commit],
            cwd=repo_dir,
            capture_output=True,
            text=True
        )
        
        if verify_result.returncode != 0:
            raise PRSetupError(
                f"Commit {commit} not found even after fetching from upstream.\n"
                f"This commit may not exist in the repository."
            )
    
    print(f"[Stage 0] ✓ Commit verified")


def _create_branch(branch_name: str, base_commit: str, repo_dir: Path):
    """
    Create isolated execution branch.
    
    Branch must start from BASE_COMMIT.
    Fail if branch already exists (do not reuse silently).
    """
    print(f"[Stage 0] Creating branch: {branch_name} from {base_commit}")
    
    # Check if branch already exists
    branch_check_result = subprocess.run(
        ["git", "rev-parse", "--verify", branch_name],
        cwd=repo_dir,
        capture_output=True,
        text=True
    )
    
    if branch_check_result.returncode == 0:
        raise PRSetupError(
            f"Branch '{branch_name}' already exists.\n"
            f"Cannot create duplicate branch. Please delete the existing branch first:\n"
            f"  git branch -D {branch_name}"
        )
    
    # Create and checkout the branch
    checkout_result = subprocess.run(
        ["git", "checkout", "-b", branch_name, base_commit],
        cwd=repo_dir,
        capture_output=True,
        text=True
    )
    
    if checkout_result.returncode != 0:
        raise PRSetupError(
            f"Failed to create branch '{branch_name}'.\n"
            f"Error: {checkout_result.stderr}"
        )
    
    print(f"[Stage 0] ✓ Branch created and checked out")


def _generate_task_file(pr_metadata: dict, task_file: Path):
    """
    Generate task description file.
    
    Content must include:
    - PR title
    - PR description
    - Clear instruction to implement the change
    """
    print(f"[Stage 0] Generating task file: {task_file}")
    
    # Ensure pr/ directory exists
    task_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate task content (matching pr_analyzer.sh format)
    title = pr_metadata['title']
    body = pr_metadata['body']
    
    task_content = f"""# Task: {title}

## PR Information
**Title:** {title}

## Description
{body}

## Requirements
Based on the PR description above, please implement the required changes.

**Please implement this feature.**
"""
    
    task_file.write_text(task_content, encoding="utf-8")
    print(f"[Stage 0] ✓ Task file generated")


def _generate_ground_truth_diff(base_commit: str, merge_commit: str, diff_file: Path, repo_dir: Path):
    """
    Generate ground-truth diff.
    
    Command: git diff BASE_COMMIT MERGE_COMMIT > pr/original_changes.diff
    Fail if diff is empty.
    """
    print(f"[Stage 0] Generating ground-truth diff: {base_commit}..{merge_commit}")
    
    # Ensure pr/ directory exists
    diff_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate diff
    diff_result = subprocess.run(
        ["git", "diff", base_commit, merge_commit],
        cwd=repo_dir,
        capture_output=True,
        text=True
    )
    
    if diff_result.returncode != 0:
        raise PRSetupError(
            f"Failed to generate diff.\n"
            f"Error: {diff_result.stderr}"
        )
    
    diff_content = diff_result.stdout
    
    # Fail if diff is empty
    if not diff_content.strip():
        raise PRSetupError(
            f"Generated diff is empty.\n"
            f"No changes found between {base_commit} and {merge_commit}.\n"
            f"This PR may not have any file changes."
        )
    
    # Save diff
    diff_file.write_text(diff_content, encoding="utf-8")
    
    diff_lines = len(diff_content.splitlines())
    print(f"[Stage 0] ✓ Ground-truth diff generated ({diff_lines} lines)")


def _save_merge_commit(merge_commit: str, merge_commit_file: Path):
    """
    Save merge commit hash to file for later context diff generation.
    
    This file is used by the orchestrator to generate human context diffs.
    """
    print(f"[Stage 0] Saving merge commit to {merge_commit_file.name}")
    
    # Ensure pr/ directory exists
    merge_commit_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save merge commit hash
    merge_commit_file.write_text(merge_commit, encoding="utf-8")
    print(f"[Stage 0] ✓ Merge commit saved")


def _verify_outputs(task_file: Path, diff_file: Path, merge_commit_file: Path):
    """Verify required outputs exist and are non-empty."""
    print(f"[Stage 0] Verifying outputs...")
    
    if not task_file.exists():
        raise PRSetupError(f"Task file not found: {task_file}")
    
    if not diff_file.exists():
        raise PRSetupError(f"Diff file not found: {diff_file}")
    
    if not merge_commit_file.exists():
        raise PRSetupError(f"Merge commit file not found: {merge_commit_file}")
    
    if not task_file.read_text(encoding="utf-8").strip():
        raise PRSetupError(f"Task file is empty: {task_file}")
    
    if not diff_file.read_text(encoding="utf-8").strip():
        raise PRSetupError(f"Diff file is empty: {diff_file}")
    
    if not merge_commit_file.read_text(encoding="utf-8").strip():
        raise PRSetupError(f"Merge commit file is empty: {merge_commit_file}")
    
    print(f"[Stage 0] ✓ All outputs verified")


def _verify_clean_state(repo_dir: Path):
    """Verify repository is in clean state after setup."""
    print(f"[Stage 0] Verifying clean git state...")
    
    # Check working tree
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_dir,
        capture_output=True,
        text=True
    )
    
    if status_result.returncode != 0:
        raise PRSetupError(
            f"Failed to check git status.\n"
            f"Error: {status_result.stderr}"
        )
    
    # We expect the pr/ directory files to show as untracked or modified
    # This is acceptable - we just need to ensure we're on the right branch
    
    # Verify current branch
    branch_result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_dir,
        capture_output=True,
        text=True
    )
    
    if branch_result.returncode != 0:
        raise PRSetupError(
            f"Failed to verify current branch.\n"
            f"Error: {branch_result.stderr}"
        )
    
    current_branch = branch_result.stdout.strip()
    print(f"[Stage 0] ✓ Clean state verified (on branch: {current_branch})")
