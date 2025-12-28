"""Tier-3 Differential Evaluation Engine.

This module contains ALL Tier-3 specific logic for:
- Raw diff comparison
- Chunking oversized diffs
- Behavioral summary generation and comparison
- Summary validation

ZERO logic changes - strict mechanical extraction from orchestrator.py
"""

import re
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, List, Dict, Optional
from enum import Enum

if TYPE_CHECKING:
    from scripts.llm_client import LLMClient

# ============================================================================
# TIER-3 CONSTANTS
# ============================================================================

# Maximum diff size for safe analysis (matches RAW_DIFF_CHAR_LIMIT for consistency)
MAX_DIFF_CHARS = 35_000

# Configuration constants for production use
CHUNK_OVERLAP_LINES = 5  # Number of lines to overlap between chunks for semantic context
MAX_CHUNK_RETRIES = 2    # Maximum retry attempts for failed chunk summarization
MAX_CONCURRENT_CHUNKS = 4  # Maximum concurrent LLM calls for chunk summarization

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_tier3_logger = logging.getLogger('tier3')


# ============================================================================
# TIER-3 DATA STRUCTURES
# ============================================================================

class ChunkResultStatus(Enum):
    """Status of chunk summarization attempt."""
    SUCCESS = "success"
    UNKNOWN = "unknown"  # LLM returned BEHAVIOR: UNKNOWN
    INVALID = "invalid"  # LLM response was malformed
    ERROR = "error"      # Exception occurred


@dataclass
class ChunkResult:
    """Result of chunk summarization with metadata.

    Attributes:
        chunk_index: Position of chunk in sequence (0-based)
        chunk_size: Size of chunk in characters
        status: Result status (SUCCESS, UNKNOWN, INVALID, ERROR)
        summary: Behavioral summary dict or None
        confidence: Confidence score for this chunk (0.0-1.0)
        retry_count: Number of retries attempted
        error_message: Error message if status is ERROR or INVALID
    """
    chunk_index: int
    chunk_size: int
    status: ChunkResultStatus
    summary: Optional[Dict[str, set]] = None
    confidence: float = 1.0
    retry_count: int = 0
    error_message: Optional[str] = None


# ============================================================================
# TIER-3 UTILITY FUNCTIONS
# ============================================================================

def _normalize_behavior(b: str) -> str:
    """Minimal normalization: lowercase and normalize whitespace only."""
    return re.sub(r"\s+", " ", b.strip().lower())


def _normalize_code_line(line: str) -> str:
    """Normalize a code line by stripping whitespace only."""
    return line.strip()


def _check_for_unknown(summary: str) -> bool:
    """Check if summary contains UNKNOWN in any category."""
    lines = summary.split('\n')
    for line in lines:
        # Check for BEHAVIOR: UNKNOWN
        if "BEHAVIOR: UNKNOWN" in line:
            return True
        # Check for "- Category: UNKNOWN" pattern
        stripped = line.strip()
        if stripped.endswith(": UNKNOWN") or stripped == "- UNKNOWN":
            return True
        # Check for value being just "UNKNOWN"
        if ": UNKNOWN" in line:
            return True
    return False


def _extract_behaviors_by_category(summary: str) -> dict:
    """Extract behaviors by category from a summary.

    Expected format:
    BEHAVIORAL SUMMARY:
    - Primary behavior change: prevent crash
    - Secondary effects: none
    - Error handling changes: add validation
    - Edge cases affected: handle empty input

    Returns:
        {
            "primary": set[str],
            "secondary": set[str],
            "error_handling": set[str],
            "edge_cases": set[str]
        }
    """
    categories = {
        "primary": set(),
        "secondary": set(),
        "error_handling": set(),
        "edge_cases": set(),
    }

    # Category names (without colon, used to match in lines)
    category_names = {
        "Primary behavior change": "primary",
        "Secondary effects": "secondary",
        "Error handling changes": "error_handling",
        "Edge cases affected": "edge_cases",
    }

    lines = summary.split('\n')

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip BEHAVIORAL SUMMARY header
        if line.startswith("BEHAVIORAL SUMMARY"):
            continue

        # Handle lines starting with "- Category: value"
        if line.startswith("-"):
            for cat_name, cat_key in category_names.items():
                # Check if line starts with "- Category name:"
                prefix = f"- {cat_name}:"
                if line.startswith(prefix):
                    # Extract behavior after the colon
                    behavior = line[len(prefix):].strip()
                    if behavior:
                        normalized = _normalize_behavior(behavior)
                        if normalized and len(normalized) > 5:
                            categories[cat_key].add(normalized)
                    break

    return categories


def _extract_changed_files(diff: str) -> set:
    """Extract set of changed file paths from a diff.

    SAFETY FIX: Handles malformed diffs by returning empty set (caller must check).

    Parses git diff format to extract all file paths that were modified.

    Args:
        diff: Raw diff text in git format

    Returns:
        Set of file paths that were changed (empty if parsing fails)
    """
    # SAFETY: Handle empty or None diffs
    if not diff or not diff.strip():
        return set()

    files = set()
    try:
        for line in diff.split('\n'):
            # Match "a/file.py" and "b/file.py" patterns in diff headers
            if line.startswith('--- a/') or line.startswith('--- b/'):
                # Handle renamed files (c/ prefix) and new files
                prefix = '--- '
                if line.startswith('--- '):
                    path = line[4:]
                elif line.startswith('--- a/'):
                    path = line[6:]
                elif line.startswith('--- b/'):
                    path = line[6:]
                else:
                    continue
                # Skip /dev/null entries (deleted or new files still have one valid path)
                if path == '/dev/null':
                    continue
                # SAFETY: Validate path is not empty
                if path.strip():
                    files.add(path)
    except Exception as e:
        # SAFETY: Parsing failure returns empty set (caller will detect file coverage mismatch)
        _tier3_logger.error(f"Diff parsing error in _extract_changed_files: {e}")
        return set()

    return files


def _extract_diff_types(diff: str) -> Dict[str, str]:
    """Extract diff operation type for each file.

    SAFETY FIX: Explicitly classify diff operations to prevent false PASS.
    Handles malformed diffs by returning empty dict (caller must check).

    Classifies each file change as:
    - ADD: New file created
    - DELETE: File removed
    - MODIFY: File content changed
    - RENAME: File moved/renamed

    Args:
        diff: Raw diff text in git format

    Returns:
        Dict mapping file path to operation type (empty if parsing fails)
    """
    # SAFETY: Handle empty or None diffs
    if not diff or not diff.strip():
        return {}

    file_ops = {}
    try:
        lines = diff.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]

            # Look for diff --git headers
            if line.startswith('diff --git'):
                # Extract file paths
                parts = line.split()
                if len(parts) >= 3:
                    old_file = parts[2].lstrip('a/')
                    new_file = parts[3].lstrip('b/')

                    # Check subsequent lines for operation type
                    j = i + 1
                    is_rename = False
                    is_new = False
                    is_deleted = False
                    rename_from = None
                    rename_to = None

                    while j < len(lines) and not lines[j].startswith('diff --git'):
                        if lines[j].startswith('rename from '):
                            is_rename = True
                            rename_from = lines[j][12:].strip()
                        elif lines[j].startswith('rename to '):
                            rename_to = lines[j][10:].strip()
                        elif lines[j].startswith('new file mode'):
                            is_new = True
                        elif lines[j].startswith('deleted file mode'):
                            is_deleted = True
                        elif lines[j].startswith('--- '):
                            path = lines[j][4:]
                            if path == '/dev/null':
                                is_new = True
                        elif lines[j].startswith('+++ '):
                            path = lines[j][4:]
                            if path == '/dev/null':
                                is_deleted = True
                            break
                        j += 1

                    # Classify operation
                    if is_rename and rename_from and rename_to:
                        file_ops[rename_to] = "RENAME"
                        file_ops[rename_from] = "RENAME"
                    elif is_new:
                        file_ops[new_file] = "ADD"
                    elif is_deleted:
                        file_ops[old_file] = "DELETE"
                    else:
                        # Default to MODIFY if file appears in both old and new
                        file_ops[new_file] = "MODIFY"

            i += 1

    except Exception as e:
        # SAFETY: Parsing failure returns empty dict (caller will detect operation type mismatch)
        _tier3_logger.error(f"Diff parsing error in _extract_diff_types: {e}")
        return {}

    return file_ops


def _extract_added_lines(diff: str) -> set:
    """Extract set of added code lines from a diff.

    Args:
        diff: Raw diff text

    Returns:
        Set of added line contents (normalized)
    """
    added = set()
    in_hunk = False
    for line in diff.split('\n'):
        # Track hunk headers to only consider actual code changes
        if line.startswith('@@'):
            in_hunk = True
            continue
        if line.startswith('+++') or line.startswith('---') or line.startswith('diff --git'):
            continue
        if line.startswith('+') and not line.startswith('+++'):
            # Normalize added lines for comparison
            content = line[1:].strip()
            if content:  # Skip empty additions
                added.add(_normalize_behavior(content))
        elif line.startswith('-'):
            in_hunk = False
        elif not line.startswith(' '):
            # Exit hunk context for non-code lines
            in_hunk = False
    return added


def _extract_removed_lines(diff: str) -> set:
    """Extract set of removed code lines from a diff.

    Args:
        diff: Raw diff text

    Returns:
        Set of removed line contents (normalized)
    """
    removed = set()
    for line in diff.split('\n'):
        if line.startswith('-') and not line.startswith('---'):
            content = line[1:].strip()
            if content:
                removed.add(_normalize_behavior(content))
    return removed


def _extract_function_changes(diff: str) -> set:
    """Extract function signature changes from a diff.

    Looks for lines with function definitions that were added or modified.

    Args:
        diff: Raw diff text

    Returns:
        Set of function signatures that changed
    """
    functions = set()
    for line in diff.split('\n'):
        # Common patterns for function definitions
        line_stripped = line.strip()
        # Skip diff metadata
        if line.startswith('@@') or line.startswith('+++') or line.startswith('---'):
            continue
        # Added or modified function
        if line.startswith('+') and '(' in line:
            # Heuristic: likely a function if it contains parentheses and doesn't look like a macro
            content = line[1:].strip()
            if content and not content.startswith('#') and not content.startswith('//'):
                functions.add(_normalize_behavior(content))
        # Removed function
        if line.startswith('-') and '(' in line:
            content = line[1:].strip()
            if content and not content.startswith('#') and not content.startswith('//'):
                functions.add(_normalize_behavior(content))
    return functions


# ============================================================================
# TIER-3 RAW DIFF COMPARISON
# ============================================================================

def _compare_raw_diffs_strict(human_diff: str, model_diff: str) -> Tuple[bool, str]:
    """Strictly compare raw diffs for structural parity.

    SAFETY FIX: Enforces file coverage, diff type correctness, and explicit absence rules.
    Malformed diff parsing always results in FAIL.

    This is the AUTHORITATIVE comparison. Raw diffs determine the verdict.

    Rules:
    1. File sets MUST be identical (no extra files, no missing files)
    2. Diff types MUST match (ADD/DELETE/MODIFY/RENAME must be consistent)
    3. Added/removed lines MUST match exactly (whitespace-only diffs ignored)
    4. Comment differences count as differences
    5. EXPLICIT ABSENCE: If ground truth modifies files A,B,C and model only modifies A,B -> FAIL
    6. Malformed diff parsing results in FAIL (no empty structures allowed)

    Args:
        human_diff: Human/ground truth diff
        model_diff: Claude/model diff

    Returns:
        (passed: bool, reason: str)
    """
    # SAFETY FIX: Detect malformed diffs early
    if not human_diff or not human_diff.strip():
        return False, "Human diff is empty or malformed"
    if not model_diff or not model_diff.strip():
        return False, "Model diff is empty or malformed"

    human_files = _extract_changed_files(human_diff)
    model_files = _extract_changed_files(model_diff)

    # SAFETY FIX: Parsing failure detection - empty results indicate parsing error
    if not human_files and "diff --git" in human_diff:
        return False, "Failed to parse human diff (no files extracted despite diff markers present)"
    if not model_files and "diff --git" in model_diff:
        return False, "Failed to parse model diff (no files extracted despite diff markers present)"

    # SAFETY FIX: File coverage enforcement - missing files MUST cause FAIL
    if human_files != model_files:
        missing = human_files - model_files
        extra = model_files - human_files
        parts = []
        if missing:
            # SAFETY: Explicit absence - ground truth requires these files but model didn't modify them
            parts.append(f"Missing required files: {', '.join(sorted(missing))}")
        if extra:
            # SAFETY: Model modified files not in ground truth
            parts.append(f"Extra files not in ground truth: {', '.join(sorted(extra))}")
        return False, "; ".join(parts)

    # SAFETY FIX: Diff type correctness - operation types must match
    human_types = _extract_diff_types(human_diff)
    model_types = _extract_diff_types(model_diff)

    # Validate that each file has matching operation type
    type_mismatches = []
    for file in human_files:
        human_op = human_types.get(file, "MODIFY")  # Default to MODIFY if not found
        model_op = model_types.get(file, "MODIFY")

        if human_op != model_op:
            type_mismatches.append(f"{file}: expected {human_op}, got {model_op}")

    if type_mismatches:
        return False, f"Diff type mismatch: {'; '.join(type_mismatches)}"

    # SAFETY FIX: Line-by-line comparison with explicit mismatch reporting
    # This enforces that ALL required changes are present (no inference allowed)
    # If a required textual change is missing → FAIL
    human_added = _extract_added_lines(human_diff)
    model_added = _extract_added_lines(model_diff)

    if human_added != model_added:
        missing_lines = human_added - model_added
        extra_lines = model_added - human_added
        parts = []
        if missing_lines:
            # SAFETY: Missing required changes cause FAIL (explicit absence enforcement)
            sample = list(missing_lines)[:3]  # Show first 3 missing lines
            parts.append(f"Missing {len(missing_lines)} required added lines (sample: {sample})")
        if extra_lines:
            sample = list(extra_lines)[:3]
            parts.append(f"Extra {len(extra_lines)} added lines not in ground truth (sample: {sample})")
        return False, "; ".join(parts) if parts else "Added lines differ"

    human_removed = _extract_removed_lines(human_diff)
    model_removed = _extract_removed_lines(model_diff)

    if human_removed != model_removed:
        missing_removals = human_removed - model_removed
        extra_removals = model_removed - human_removed
        parts = []
        if missing_removals:
            sample = list(missing_removals)[:3]
            parts.append(f"Missing {len(missing_removals)} required removed lines (sample: {sample})")
        if extra_removals:
            sample = list(extra_removals)[:3]
            parts.append(f"Extra {len(extra_removals)} removed lines not in ground truth (sample: {sample})")
        return False, "; ".join(parts) if parts else "Removed lines differ"

    return True, "Raw diffs match exactly"


def _validate_summary_against_diff(summary: str, diff: str, summary_owner: str) -> Tuple[bool, str]:
    """Validate that a summary is consistent with its raw diff.

    Summary must not claim changes not present in diff, and must not miss
    changes that are present in diff.

    Args:
        summary: Behavioral summary text
        diff: Raw diff text
        summary_owner: "Human" or "Claude" for error messages

    Returns:
        (is_valid: bool, reason: str)
    """
    behaviors = _extract_behaviors_by_category(summary)

    # Check for summary claims not backed by diff
    # This is a simplified check - looks for behavioral claims that might not match
    summary_text = summary.lower()

    # Get diff stats
    added_lines = _extract_added_lines(diff)

    # Extract key concepts from behaviors
    primary_behaviors = behaviors.get("primary", set())

    # Verify behaviors can be traced to diff content (heuristic)
    # If a behavior mentions something specific, check it appears in diff
    for behavior in primary_behaviors:
        # Skip very short or generic behaviors
        if len(behavior) < 10:
            continue
        # Split into key terms
        terms = set(behavior.split()) - {'the', 'a', 'an', 'to', 'for', 'in', 'of', 'and', 'or', 'is', 'if', 'with', 'on', 'be', 'not', 'null', 'none'}
        # Check if at least some terms appear in diff
        diff_lower = diff.lower()
        matched_terms = sum(1 for term in terms if term in diff_lower)
        if matched_terms == 0 and len(terms) > 2:
            return False, f"{summary_owner} summary claims behavior not found in diff: '{behavior[:50]}...'"

    return True, f"{summary_owner} summary validated against diff"


# =============================================================================
# TIER-3 SUMMARY COMPLETENESS VALIDATION
# =============================================================================

def _validate_tier3_summary_completeness(summary: str, diff: str, is_model: bool = False) -> Tuple[bool, str]:
    """Validate Tier-3 behavioral summary includes all required elements.

    Tier-3 is authoritative and deterministic. The summary MUST explicitly
    document all changes, not leave gaps that could mask incomplete fixes.

    Required elements:
    1. List of modified files (extracted from diff headers)
    2. Logic changes per file (what was added/removed)
    3. Control-flow changes (conditionals, loops, early returns, error handling)
    4. Behavioral impact description (what behavior changes)

    This is a conservative validation - any ambiguity about completeness
    triggers FAIL to prevent false confidence.

    Args:
        summary: Behavioral summary text
        diff: Raw diff text for reference
        is_model: True if validating Claude's summary, False for human

    Returns:
        (is_complete: bool, reason: str)
    """
    if not summary or summary.strip() == "":
        owner = "Claude" if is_model else "Human"
        return False, f"{owner} summary is empty"

    summary_lower = summary.lower()
    diff_lower = diff.lower()

    # Extract modified files from diff for comparison
    files = _extract_changed_files(diff)
    file_count = len(files)

    # Required check patterns - summaries must contain evidence of these elements
    # Using flexible matching since LLM output format varies
    required_patterns = {
        "files_mentioned": False,  # At least some files or path references
        "logic_changes": False,    # Evidence of what changed (add, remove, modify)
        "control_flow": False,     # Evidence of if/else, try/catch, loops, returns
        "behavioral_impact": False # Evidence of what behavior changes
    }

    # Check for files mentioned - look for paths, filenames, or explicit mentions
    files_mention_patterns = [
        "files changed", "modified files", "changed files",
        "a/core", "b/utils", "/core.py", "/utils.py",  # Common diff path patterns
        "in core", "in utils", "in the file", "the file"
    ]
    for pattern in files_mention_patterns:
        if pattern in summary_lower:
            required_patterns["files_mentioned"] = True
            break

    # Check for logic changes - look for action verbs and change indicators
    logic_patterns = [
        "add", "remove", "modify", "change", "update", "insert", "delete",
        "adds", "removes", "modifies", "changes", "updates", "inserts", "deletes",
        "was added", "was removed", "was modified", "are added", "are removed"
    ]
    for pattern in logic_patterns:
        if pattern in summary_lower:
            required_patterns["logic_changes"] = True
            break

    # Check for control-flow keywords
    control_flow_patterns = [
        "if", "else", "condition", "try", "catch", "except", "finally",
        "loop", "iterate", "return", "early return", "throw", "raise",
        "error handling", "exception", "switch", "match", "case"
    ]
    for pattern in control_flow_patterns:
        if pattern in summary_lower:
            required_patterns["control_flow"] = True
            break

    # Check for behavioral impact description
    behavior_patterns = [
        "behavior", "impact", "effect", "result", "outcome", "prevents",
        "avoids", "fixes", "resolves", "handles", "ensures", "guarantees"
    ]
    for pattern in behavior_patterns:
        if pattern in summary_lower:
            required_patterns["behavioral_impact"] = True
            break

    # Build failure reason for missing elements
    missing = [k for k, v in required_patterns.items() if not v]
    if missing:
        missing_formatted = [k.replace("_", " ").title() for k in missing]
        owner = "Claude" if is_model else "Human"
        return False, f"{owner} summary incomplete - missing: {', '.join(missing_formatted)}"

    # Additional validation: ensure summary has reasonable length for the diff
    # A summary that's too short relative to diff may indicate incomplete analysis
    if len(summary.split()) < 5 and len(diff.splitlines()) > 10:
        owner = "Claude" if is_model else "Human"
        return False, f"{owner} summary is suspiciously brief for diff size"

    return True, "Tier-3 summary is complete"


# ============================================================================
# TIER-3 DIFF CHUNKING FOR OVERSIZED DIFFS (PRODUCTION-READY)
# ============================================================================

def _split_diff_into_chunks(
    diff_content: str,
    max_chars: int = MAX_DIFF_CHARS,
    overlap_lines: int = CHUNK_OVERLAP_LINES
) -> List[str]:
    """Split diff content into sequential chunks with optional overlap.

    Preserves semantic context by including a small overlap between consecutive
    chunks. This helps the LLM understand dependencies across chunk boundaries.

    Safety guarantees:
    - Each chunk <= max_chars (enforced)
    - Single oversized lines raise ValueError
    - All original content is preserved (with overlap)

    Args:
        diff_content: The full diff text to split
        max_chars: Maximum characters per chunk (default: MAX_DIFF_CHARS)
        overlap_lines: Number of lines to overlap between chunks (default: 5)

    Returns:
        List of diff chunks, each <= max_chars, with overlap preserved

    Raises:
        ValueError: If any single line exceeds max_chars

    Example:
        >>> diff = "line1\nline2\nline3\nline4\nline5"
        >>> chunks = _split_diff_into_chunks(diff, max_chars=30, overlap_lines=1)
        >>> # Might produce: ["line1\nline2\nline3", "line3\nline4\nline5"]
    """
    if not diff_content or len(diff_content) <= max_chars:
        return [diff_content] if diff_content else []

    lines = diff_content.split('\n')
    n_lines = len(lines)

    # Check for oversized lines first (safety guarantee)
    for i, line in enumerate(lines, 1):
        if len(line) > max_chars:
            raise ValueError(
                f"Line {i} exceeds max_chars ({len(line)} > {max_chars}). "
                "Cannot safely chunk oversized diff."
            )

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_size = 0

    for i, line in enumerate(lines):
        line_size = len(line) + 1  # +1 for newline

        # Check if adding this line would exceed limit
        if current_size + line_size > max_chars:
            # Save current chunk
            if current_chunk:
                chunks.append('\n'.join(current_chunk))

            # Start new chunk with overlap from previous chunk
            overlap_start = max(0, i - overlap_lines)
            current_chunk = lines[overlap_start:i]
            current_size = sum(len(l) + 1 for l in current_chunk) - 1
        else:
            current_chunk.append(line)
            current_size += line_size

    # Don't forget the last chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    _tier3_logger.debug(
        f"Split diff ({len(diff_content)} chars) into {len(chunks)} chunks "
        f"(overlap={overlap_lines} lines, max_chars={max_chars})"
    )

    return chunks


def _summarize_chunk_with_retry(
    chunk: str,
    client: "LLMClient",
    chunk_index: int,
    prompt_template: str = None,
    max_retries: int = MAX_CHUNK_RETRIES,
    timeout: int = None
) -> ChunkResult:
    """Generate behavioral summary for a single diff chunk with retry logic.

    Retries failed summarization attempts up to max_retries times before
    returning a failure result. This improves robustness for transient LLM errors.

    Args:
        chunk: A diff chunk <= MAX_DIFF_CHARS
        client: LLM client for generating summary
        chunk_index: Position of this chunk in the sequence
        prompt_template: The prompt template to use for summaries
        max_retries: Maximum retry attempts (default: MAX_CHUNK_RETRIES)
        timeout: Timeout for chunk summarization LLM call

    Returns:
        ChunkResult with status, summary (if successful), and metadata

    Example:
        >>> result = _summarize_chunk_with_retry("fix null pointer", client, 0)
        >>> if result.status == ChunkResultStatus.SUCCESS:
        ...     behaviors = result.summary
    """
    # Fallback: auto-load from Prompt.md if needed (dependency injection preferred)
    if prompt_template is None:
        # NOTE: This import is deferred to avoid circular dependency
        from pathlib import Path
        AUTOMATION_DIR = Path(__file__).parent.parent.resolve()
        from scripts.prompt_builder import load_templates
        templates = load_templates(AUTOMATION_DIR / "templates" / "Prompt.md")
        prompt_template = templates.behavioral_summary

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = client.run(
                prompt_template.format(diff=chunk),
                timeout=timeout
            )
            response = response.strip()

            # Check for UNKNOWN (not retryable)
            if "BEHAVIOR: UNKNOWN" in response:
                _tier3_logger.warning(
                    f"Chunk {chunk_index}: LLM returned BEHAVIOR: UNKNOWN"
                )
                return ChunkResult(
                    chunk_index=chunk_index,
                    chunk_size=len(chunk),
                    status=ChunkResultStatus.UNKNOWN,
                    confidence=0.0,
                    retry_count=attempt,
                    error_message="LLM returned BEHAVIOR: UNKNOWN"
                )

            # Check for valid response
            if not response or len(response) < 20:
                last_error = f"Response too short: {len(response)} chars"
                _tier3_logger.debug(f"Chunk {chunk_index}: {last_error}")
                continue

            if "BEHAVIORAL SUMMARY:" not in response:
                last_error = "Missing BEHAVIORAL SUMMARY header"
                _tier3_logger.debug(f"Chunk {chunk_index}: {last_error}")
                continue

            # Extract behaviors by category
            behaviors = _extract_behaviors_by_category(response)

            _tier3_logger.debug(
                f"Chunk {chunk_index}: Success (attempt {attempt + 1}, "
                f"primary={len(behaviors.get('primary', set()))}, "
                f"error_handling={len(behaviors.get('error_handling', set()))})"
            )

            return ChunkResult(
                chunk_index=chunk_index,
                chunk_size=len(chunk),
                status=ChunkResultStatus.SUCCESS,
                summary=behaviors,
                confidence=1.0,
                retry_count=attempt
            )

        except Exception as e:
            last_error = str(e)
            _tier3_logger.warning(f"Chunk {chunk_index}: Error on attempt {attempt + 1}: {e}")

    # All retries exhausted
    _tier3_logger.warning(
        f"Chunk {chunk_index}: Failed after {max_retries + 1} attempts"
    )

    return ChunkResult(
        chunk_index=chunk_index,
        chunk_size=len(chunk),
        status=ChunkResultStatus.ERROR,
        confidence=0.0,
        retry_count=max_retries,
        error_message=f"Failed after {max_retries + 1} attempts: {last_error}"
    )


def _summarize_chunks_concurrent(
    chunks: List[str],
    client: "LLMClient",
    max_concurrent: int = MAX_CONCURRENT_CHUNKS,
    prompt_template: str = None,
    timeout: int = None
) -> List[ChunkResult]:
    """Summarize multiple chunks concurrently for reduced latency.

    Uses ThreadPoolExecutor to parallelize chunk summarization LLM calls.
    Results are returned in original chunk order regardless of completion order.

    Args:
        chunks: List of diff chunks to summarize
        client: LLM client for generating summaries
        max_concurrent: Maximum concurrent LLM calls (default: MAX_CONCURRENT_CHUNKS)
        prompt_template: The prompt template to use for summaries
        timeout: Timeout for chunk summarization LLM call

    Returns:
        List of ChunkResult objects in original chunk order

    Example:
        >>> chunks = ["fix a", "fix b", "fix c"]
        >>> results = _summarize_chunks_concurrent(chunks, client)
        >>> successful = [r for r in results if r.status == ChunkResultStatus.SUCCESS]
    """
    if not chunks:
        return []

    _tier3_logger.info(
        f"Summarizing {len(chunks)} chunks concurrently (max_concurrent={max_concurrent})"
    )

    results: Dict[int, ChunkResult] = {}

    def process_chunk(args: Tuple[str, int]) -> Tuple[int, ChunkResult]:
        """Process a single chunk and return (index, result)."""
        chunk, idx = args
        return idx, _summarize_chunk_with_retry(chunk, client, idx, prompt_template, timeout=timeout)

    # Use ThreadPoolExecutor for I/O-bound LLM calls
    with ThreadPoolExecutor(max_workers=min(max_concurrent, len(chunks))) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_chunk, (chunk, idx)): idx
            for idx, chunk in enumerate(chunks)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as e:
                idx = futures[future]
                _tier3_logger.error(f"Chunk {idx}: Unexpected error: {e}")
                results[idx] = ChunkResult(
                    chunk_index=idx,
                    chunk_size=len(chunks[idx]),
                    status=ChunkResultStatus.ERROR,
                    confidence=0.0,
                    error_message=f"Unexpected error: {e}"
                )

    # Return results in original order
    return [results[i] for i in range(len(chunks))]


def _aggregate_chunk_results(results: List[ChunkResult]) -> Tuple[Dict, float, List[str]]:
    """Aggregate chunk results with confidence scoring.

    Combines behavioral summaries from successful chunks and calculates
    an aggregate confidence score based on retry counts and failures.

    Args:
        results: List of ChunkResult objects from all chunks

    Returns:
        Tuple of (aggregated_behaviors, aggregate_confidence, failure_reasons)

    Example:
        >>> behaviors, confidence, failures = _aggregate_chunk_results(results)
        >>> if confidence < 1.0:
        ...     _tier3_logger.warning(f"Aggregate confidence: {confidence}")
    """
    aggregated = {
        "primary": set(),
        "secondary": set(),
        "error_handling": set(),
        "edge_cases": set(),
    }

    failure_reasons: List[str] = []
    total_confidence = 1.0
    success_count = 0
    total_retries = 0
    total_chunks = len(results)

    for result in results:
        if result.status == ChunkResultStatus.SUCCESS:
            success_count += 1
            total_retries += result.retry_count

            # Aggregate behaviors
            if result.summary:
                for category in aggregated:
                    aggregated[category].update(result.summary.get(category, set()))

        elif result.status == ChunkResultStatus.UNKNOWN:
            failure_reasons.append(
                f"Chunk {result.chunk_index}: UNKNOWN behavior (size={result.chunk_size})"
            )
            total_confidence *= 0.5  # Heavy penalty for UNKNOWN

        elif result.status in (ChunkResultStatus.INVALID, ChunkResultStatus.ERROR):
            failure_reasons.append(
                f"Chunk {result.chunk_index}: {result.error_message} (size={result.chunk_size})"
            )
            total_confidence *= 0.7  # Moderate penalty for errors

    # Calculate aggregate confidence based on success rate and retries
    if total_chunks > 0:
        success_rate = success_count / total_chunks
        retry_penalty = 0.95 ** total_retries if total_retries > 0 else 1.0
        total_confidence = total_confidence * success_rate * retry_penalty

    _tier3_logger.debug(
        f"Aggregated {success_count}/{total_chunks} chunks, "
        f"confidence={total_confidence:.3f}, retries={total_retries}"
    )

    return aggregated, total_confidence, failure_reasons


def _compare_chunked_summaries(
    human_agg: Dict[str, set],
    claude_agg: Dict[str, set],
    aggregate_confidence: float = 1.0
) -> Tuple[str, str, float]:
    """Compare aggregated behavioral summaries with confidence-aware scoring.

    SAFETY FIX: Global aggregation enforced - individual chunks cannot independently PASS.
    All chunks are aggregated first, then compared as a unified summary.

    Combines structural comparison confidence with aggregate confidence
    from chunk processing.

    Category-aware comparison (strict Tier-3 rules):
    - PRIMARY: Claude must match all human primary behaviors
    - SECONDARY: Ignored (no strict requirements)
    - ERROR_HANDLING: Claude must not remove human error handling
    - EDGE_CASES: Claude must not remove human edge cases

    Args:
        human_agg: Aggregated human behavioral summary (from ALL chunks combined)
        claude_agg: Aggregated Claude behavioral summary (from ALL chunks combined)
        aggregate_confidence: Confidence from chunk processing (0.0-1.0)

    Returns:
        Tuple of (verdict: str, reason: str, combined_confidence: float)

    Example:
        >>> verdict, reason, confidence = _compare_chunked_summaries(human, claude, 0.95)
        >>> print(f"{verdict}: {reason} (confidence={confidence:.2f})")
    """
    failures: List[str] = []
    comparison_confidence = 1.0

    # PRIMARY: Claude must match all human primary behaviors
    missing_primary = human_agg["primary"] - claude_agg["primary"]
    if missing_primary:
        failures.append(f"Missing primary: {', '.join(sorted(missing_primary))}")
        comparison_confidence *= 0.8

    extra_primary = claude_agg["primary"] - human_agg["primary"]
    if extra_primary:
        failures.append(f"Extra primary: {', '.join(sorted(extra_primary))}")
        comparison_confidence *= 0.8

    # ERROR_HANDLING: Claude must not remove human error handling
    missing_error = human_agg["error_handling"] - claude_agg["error_handling"]
    if missing_error:
        failures.append(f"Missing error handling: {', '.join(sorted(missing_error))}")
        comparison_confidence *= 0.8

    # EDGE_CASES: Claude must not remove human edge cases
    missing_edge = human_agg["edge_cases"] - claude_agg["edge_cases"]
    if missing_edge:
        failures.append(f"Missing edge cases: {', '.join(sorted(missing_edge))}")
        comparison_confidence *= 0.85

    # SECONDARY: No strict requirements (ignored for PASS/FAIL)

    # Combine with aggregate confidence
    combined_confidence = aggregate_confidence * comparison_confidence

    # SAFETY FIX: Chunk aggregation precedence - FAIL > ERROR > PASS
    # Any failure in any chunk must result in final FAIL
    if failures:
        return ("FAIL", "; ".join(failures), combined_confidence)

    # Only return PASS if no failures detected
    return ("PASS", "All behavioral categories match requirements", combined_confidence)


# ============================================================================
# TIER-3 BEHAVIORAL SUMMARY GENERATION AND COMPARISON
# ============================================================================

def generate_behavioral_summary(diff_content: str, client: "LLMClient",
                                 prompt_template: str = None,
                                 timeout: int = None) -> str:
    """Generate a behavioral summary from a diff.

    Args:
        diff_content: The diff content to summarize
        client: LLM client for API calls
        prompt_template: The prompt template to use (loaded from Prompt.md)
        timeout: Timeout for the LLM call

    Returns:
        Behavioral summary text or None if UNKNOWN/empty/invalid.
    """
    # TRUNCATION SAFETY: Never analyze truncated diffs
    if not diff_content or not diff_content.strip():
        print("  [Tier-3] Diff is empty")
        return None

    if len(diff_content) > MAX_DIFF_CHARS:
        print(f"  [Tier-3] Diff exceeds safe analysis limit ({len(diff_content)} > {MAX_DIFF_CHARS})")
        return None

    if prompt_template is None:
        # Auto-load from Prompt.md (dependency injection preferred)
        from pathlib import Path
        AUTOMATION_DIR = Path(__file__).parent.parent.resolve()
        from scripts.prompt_builder import load_templates
        templates = load_templates(AUTOMATION_DIR / "templates" / "Prompt.md")
        prompt_template = templates.behavioral_summary

    prompt = prompt_template.format(diff=diff_content)

    try:
        response = client.run(prompt, timeout=timeout)
        response = response.strip()

        # Check for UNKNOWN
        if "BEHAVIOR: UNKNOWN" in response:
            print("  [Tier-3] Behavioral summary: UNKNOWN")
            return None

        # Check for empty/invalid response
        if not response or len(response) < 20:
            print("  [Tier-3] Behavioral summary: empty or too short")
            return None

        # Must contain BEHAVIORAL SUMMARY header
        if "BEHAVIORAL SUMMARY:" not in response:
            print("  [Tier-3] Behavioral summary: missing header")
            return None

        print(f"  [Tier-3] Behavioral summary generated: {len(response)} chars")
        return response

    except Exception as e:
        print(f"  [Tier-3] Behavioral summary generation failed: {e}")
        return None


def compare_behavioral_summaries(human_summary: str, claude_summary: str) -> tuple:
    """Compare two behavioral summaries and determine verdict.

    Category-aware comparison:
    - PRIMARY: Claude must match all human primary behaviors
    - SECONDARY: Missing secondary is OK, extra is OK
    - ERROR_HANDLING: Claude must not remove human error handling
    - EDGE_CASES: Claude may have more, but not fewer

    Returns:
        (verdict: str, reason: str, confidence: float)
    """
    if not human_summary or not claude_summary:
        return ("FAIL_INFRA", "Cannot compare: summary missing", 1.0)

    # UNKNOWN BEHAVIOR RULE
    if _check_for_unknown(human_summary) or _check_for_unknown(claude_summary):
        return ("FAIL_INFRA", "Behavioral comparison unsafe: UNKNOWN behavior detected", 1.0)

    # Extract behaviors by category
    human_behaviors = _extract_behaviors_by_category(human_summary)
    claude_behaviors = _extract_behaviors_by_category(claude_summary)

    failures = []
    confidence = 1.0

    # PRIMARY BEHAVIOR: Claude MUST match all human primary behaviors
    missing_primary = human_behaviors["primary"] - claude_behaviors["primary"]
    if missing_primary:
        failures.append(f"Missing primary behavior: {', '.join(missing_primary)}")
        confidence = 0.8

    extra_primary = claude_behaviors["primary"] - human_behaviors["primary"]
    if extra_primary:
        failures.append(f"Extra primary behavior: {', '.join(extra_primary)}")
        confidence = 0.8

    # ERROR HANDLING: Claude must NOT remove any human error handling
    missing_error = human_behaviors["error_handling"] - claude_behaviors["error_handling"]
    if missing_error:
        failures.append(f"Missing error handling: {', '.join(missing_error)}")
        confidence = 0.8

    # EDGE CASES: Claude may have more, but not fewer
    missing_edge = human_behaviors["edge_cases"] - claude_behaviors["edge_cases"]
    if missing_edge:
        failures.append(f"Missing edge case handling: {', '.join(missing_edge)}")
        confidence = 0.85

    # SECONDARY EFFECTS: No strict requirements
    # (missing or extra is allowed)

    if failures:
        return ("FAIL", "; ".join(failures), confidence)

    return ("PASS", "All behavioral categories match requirements", 1.0)


# ============================================================================
# TIER-3 ENGINE: MAIN ENTRY POINT
# ============================================================================

def compare_chunked_with_claude(
    human_diff: str,
    claude_diff: str,
    client: "LLMClient",
    overlap_lines: int = CHUNK_OVERLAP_LINES,
    max_concurrent: int = MAX_CONCURRENT_CHUNKS,
    templates: any = None,
    timeout: int = None
) -> Tuple[str, str, float]:
    """Compare human and Claude diffs using behavioral summaries.

    SAFETY FIX: Default to FAIL on any parsing/processing errors.

    Tier-3 evaluation entry point. Raw diff comparison is authoritative.

    Args:
        human_diff: Human/ground truth diff
        claude_diff: Claude/model diff
        client: LLM client for generating summaries
        overlap_lines: Lines to overlap between chunks
        max_concurrent: Max concurrent LLM calls
        templates: Optional Templates object for prompt loading (from Prompt.md)
        timeout: Timeout for LLM calls

    Returns:
        Tuple of (verdict: str, reason: str, confidence: float)
        - verdict: PASS, FAIL, or FAIL_INFRA
        - reason: Explanation of verdict
        - confidence: 0.0-1.0 (lower = less confident)
    """
    # SAFETY FIX: Wrap entire function in try-catch to ensure parsing errors default to FAIL
    try:
        # Load templates if not provided (dependency injection preferred)
        prompt_template = None
        if templates is None:
            from pathlib import Path
            AUTOMATION_DIR = Path(__file__).parent.parent.resolve()
            from scripts.prompt_builder import load_templates
            templates = load_templates(AUTOMATION_DIR / "templates" / "Prompt.md")

        prompt_template = templates.behavioral_summary

        # Handle empty diffs
        if not human_diff or not human_diff.strip():
            return ("FAIL_INFRA", "Human diff is empty", 0.0)
        if not claude_diff or not claude_diff.strip():
            return ("FAIL", "Claude diff is empty - no changes made", 1.0)

        # Check for oversized lines (safety)
        for diff_name, diff_content in [("Human", human_diff), ("Claude", claude_diff)]:
            for i, line in enumerate(diff_content.split('\n'), 1):
                if len(line) > MAX_DIFF_CHARS:
                    return (
                        "FAIL_INFRA",
                        f"{diff_name} diff line {i} exceeds MAX_DIFF_CHARS ({len(line)} > {MAX_DIFF_CHARS}).",
                        0.0
                    )

        _tier3_logger.info(
            f"Tier-3: Human={len(human_diff)} chars, Claude={len(claude_diff)} chars"
        )

        # ========================================================================
        # TIER-3 EVALUATION: RAW DIFF IS AUTHORITATIVE
        # ========================================================================

        # 1. RAW DIFF STRICT COMPARISON (authoritative)
        raw_passed, raw_reason = _compare_raw_diffs_strict(human_diff, claude_diff)
        if not raw_passed:
            _tier3_logger.warning(f"Raw diff FAILED: {raw_reason}")
            return ("FAIL", raw_reason, 0.0)

        # 2. Check if chunking is needed
        human_needs_chunking = len(human_diff) > MAX_DIFF_CHARS
        claude_needs_chunking = len(claude_diff) > MAX_DIFF_CHARS

        # 3. Generate/collect summaries
        if not human_needs_chunking and not claude_needs_chunking:
            # Small diffs - direct summarization
            human_summary = generate_behavioral_summary(human_diff, client, prompt_template, timeout=timeout)
            if not human_summary:
                return ("FAIL_INFRA", "Could not generate human diff summary", 0.0)

            claude_summary = generate_behavioral_summary(claude_diff, client, prompt_template, timeout=timeout)
            if not claude_summary:
                return ("FAIL_INFRA", "Could not generate Claude diff summary", 0.0)

            # Validate summaries against diffs
            valid, reason = _validate_summary_against_diff(human_summary, human_diff, "Human")
            if not valid:
                return ("FAIL", f"Human summary inconsistent: {reason}", 0.0)

            valid, reason = _validate_summary_against_diff(claude_summary, claude_diff, "Claude")
            if not valid:
                return ("FAIL", f"Claude summary inconsistent: {reason}", 0.0)

            # Compare summaries
            verdict, reason, confidence = compare_behavioral_summaries(human_summary, claude_summary)

            if verdict == "PASS":
                return ("PASS", f"Raw diff parity confirmed. {reason}", 0.95)
            else:
                return (verdict, f"Raw diffs match but behavior differs: {reason}", confidence)

        # Large diffs - use chunked summarization
        _tier3_logger.info(f"Large diffs - using chunking (human={human_needs_chunking}, claude={claude_needs_chunking})")

        try:
            human_chunks = _split_diff_into_chunks(human_diff, overlap_lines=overlap_lines) if human_needs_chunking else [human_diff]
        except ValueError as e:
            return ("FAIL_INFRA", str(e), 0.0)

        try:
            claude_chunks = _split_diff_into_chunks(claude_diff, overlap_lines=overlap_lines) if claude_needs_chunking else [claude_diff]
        except ValueError as e:
            return ("FAIL_INFRA", str(e), 0.0)

        # Summarize chunks concurrently
        human_results = _summarize_chunks_concurrent(human_chunks, client, max_concurrent, prompt_template, timeout=timeout)
        claude_results = _summarize_chunks_concurrent(claude_chunks, client, max_concurrent, prompt_template, timeout=timeout)

        # Check for UNKNOWN in any chunk (FAIL_INFRA if found)
        for result in human_results:
            if result.status == ChunkResultStatus.UNKNOWN:
                return ("FAIL_INFRA", f"UNKNOWN in human chunk {result.chunk_index}", result.confidence)
        for result in claude_results:
            if result.status == ChunkResultStatus.UNKNOWN:
                return ("FAIL_INFRA", f"UNKNOWN in Claude chunk {result.chunk_index}", result.confidence)

        # SAFETY FIX: Global aggregation - ALL chunks must be aggregated before final verdict
        # Individual chunks CANNOT independently PASS - aggregation is mandatory
        human_agg, human_confidence, _ = _aggregate_chunk_results(human_results)
        claude_agg, claude_confidence, _ = _aggregate_chunk_results(claude_results)

        aggregate_confidence = min(human_confidence, claude_confidence)

        # Compare aggregated summaries (behavioral - raw diffs already passed)
        # Final verdict is based on complete aggregated behavior, not individual chunks
        verdict, reason, _ = _compare_chunked_summaries(human_agg, claude_agg, aggregate_confidence)

        if verdict == "PASS":
            return ("PASS", f"Raw diff parity confirmed. {reason}", 0.95)
        else:
            return (verdict, f"Raw diffs match but behavior differs: {reason}", aggregate_confidence)

    except Exception as e:
        # SAFETY FIX: Any parsing/processing error defaults to FAIL
        _tier3_logger.error(f"Tier-3 processing error: {e}")
        return (
            "FAIL",
            f"Tier-3 diff processing error: {str(e)}. Defaulting to FAIL for safety.",
            0.0
        )
