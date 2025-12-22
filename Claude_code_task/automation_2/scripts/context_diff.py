#!/usr/bin/env python3
"""
Context-Aware Git Diff Generator

This script generates git diffs that include complete function/method context
instead of just the changed lines. This helps LLMs understand the full context
of code changes.

Usage:
    python3 scripts/context_diff.py                    # Diff of unstaged changes
    python3 scripts/context_diff.py --staged           # Diff of staged changes
    python3 scripts/context_diff.py HEAD~1             # Diff against a commit
    python3 scripts/context_diff.py main               # Diff against a branch
    python3 scripts/context_diff.py commit1 commit2    # Diff between two commits
    python3 scripts/context_diff.py -o output.diff     # Save to file
"""

import subprocess
import sys
import re
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict


@dataclass
class DiffHunk:
    """Represents a single hunk in a diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]
    header: str


@dataclass
class FileDiff:
    """Represents the diff for a single file."""
    old_path: str
    new_path: str
    hunks: list[DiffHunk]
    is_new: bool = False
    is_deleted: bool = False
    is_binary: bool = False


@dataclass
class FunctionInfo:
    """Information about a function/method in the code."""
    name: str
    start_line: int
    end_line: int
    signature: str


# Language-specific patterns for detecting function/method definitions
# Kept only languages used in this codebase: Rust, JavaScript, TypeScript, Python, Shell, TOML
FUNCTION_PATTERNS = {
    # Rust - prioritize fn definitions over impl blocks
    '.rs': [
        # Function definitions (these are the primary target)
        r'^\s*(pub(\s*\([^)]*\))?\s+)?(async\s+)?fn\s+\w+',
        # Test attributes (to capture test functions)
        r'^\s*#\[(tokio::)?test',
    ],
    # Python
    '.py': [
        r'^\s*def\s+\w+',
        r'^\s*async\s+def\s+\w+',
        r'^\s*class\s+\w+',
        r'^\s*@\w+',  # decorators
    ],
    # JavaScript
    '.js': [
        r'^\s*(export\s+)?(async\s+)?function\s+\w+',
        r'^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?\(?.*\)?\s*=>',
        r'^\s*(export\s+)?class\s+\w+',
        r'^\s*\w+\s*\([^)]*\)\s*{',  # method in class
        r'^\s*(async\s+)?\w+\s*:\s*(async\s+)?\(?.*\)?\s*=>',
    ],
    # TypeScript
    '.ts': [
        r'^\s*(export\s+)?(async\s+)?function\s+\w+',
        r'^\s*(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?\(?.*\)?\s*=>',
        r'^\s*(export\s+)?class\s+\w+',
        r'^\s*(public|private|protected)?\s*(async\s+)?\w+\s*\([^)]*\)',
        r'^\s*interface\s+\w+',
        r'^\s*type\s+\w+',
    ],
    # Shell
    '.sh': [
        r'^\s*\w+\s*\(\)\s*{',
        r'^\s*function\s+\w+',
    ],
    '.bash': [
        r'^\s*\w+\s*\(\)\s*{',
        r'^\s*function\s+\w+',
    ],
    # TOML (for config files - treat sections as "functions")
    '.toml': [
        r'^\s*\[+[^\]]+\]+',
    ],
}

# Bracket/block matching for different languages
BLOCK_DELIMITERS = {
    '.rs': ('{', '}'),
    '.py': None,  # Python uses indentation
    '.js': ('{', '}'),
    '.ts': ('{', '}'),
    '.sh': ('{', '}'),
    '.bash': ('{', '}'),
    '.toml': None,  # TOML uses sections
}


def get_file_extension(filepath: str) -> str:
    """Get the file extension from a filepath."""
    return Path(filepath).suffix.lower()


def run_git_command(args: list[str]) -> tuple[int, str, str]:
    """Run a git command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            ['git'] + args,
            capture_output=True,
            text=True
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, '', str(e)


def get_file_content(filepath: str, ref: Optional[str] = None) -> Optional[list[str]]:
    """Get file content from working directory or a git ref."""
    if ref:
        # Get content from git
        returncode, stdout, stderr = run_git_command(['show', f'{ref}:{filepath}'])
        if returncode != 0:
            return None
        return stdout.splitlines()
    else:
        # Get content from working directory
        try:
            full_path = Path(filepath)
            if not full_path.is_absolute():
                # Get git root
                _, git_root, _ = run_git_command(['rev-parse', '--show-toplevel'])
                full_path = Path(git_root.strip()) / filepath
            
            if full_path.exists():
                return full_path.read_text().splitlines()
        except Exception:
            pass
        return None


def parse_diff_header(line: str) -> Optional[tuple[int, int, int, int]]:
    """Parse a diff hunk header like @@ -1,5 +1,6 @@."""
    match = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
    if match:
        old_start = int(match.group(1))
        old_count = int(match.group(2)) if match.group(2) else 1
        new_start = int(match.group(3))
        new_count = int(match.group(4)) if match.group(4) else 1
        return old_start, old_count, new_start, new_count
    return None


def parse_git_diff(diff_output: str) -> list[FileDiff]:
    """Parse git diff output into structured FileDiff objects."""
    files = []
    current_file = None
    current_hunk = None
    
    lines = diff_output.splitlines()
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # New file diff header
        if line.startswith('diff --git'):
            if current_file and current_hunk:
                current_file.hunks.append(current_hunk)
            if current_file:
                files.append(current_file)
            
            # Parse file paths
            match = re.match(r'^diff --git a/(.*) b/(.*)$', line)
            if match:
                current_file = FileDiff(
                    old_path=match.group(1),
                    new_path=match.group(2),
                    hunks=[]
                )
            current_hunk = None
            
        elif line.startswith('new file mode'):
            if current_file:
                current_file.is_new = True
                
        elif line.startswith('deleted file mode'):
            if current_file:
                current_file.is_deleted = True
                
        elif line.startswith('Binary files'):
            if current_file:
                current_file.is_binary = True
                
        elif line.startswith('@@'):
            if current_file and current_hunk:
                current_file.hunks.append(current_hunk)
            
            parsed = parse_diff_header(line)
            if parsed:
                old_start, old_count, new_start, new_count = parsed
                current_hunk = DiffHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=[],
                    header=line
                )
                
        elif current_hunk is not None:
            if line.startswith('+') or line.startswith('-') or line.startswith(' ') or line == '':
                current_hunk.lines.append(line)
        
        i += 1
    
    # Don't forget the last hunk/file
    if current_file and current_hunk:
        current_file.hunks.append(current_hunk)
    if current_file:
        files.append(current_file)
    
    return files


def find_function_boundaries(lines: list[str], extension: str) -> list[FunctionInfo]:
    """Find all function/method boundaries in the file content."""
    functions = []
    patterns = FUNCTION_PATTERNS.get(extension, [])
    delimiters = BLOCK_DELIMITERS.get(extension)
    
    if not patterns:
        return functions
    
    compiled_patterns = [re.compile(p) for p in patterns]
    
    # For Rust, we need special handling to find fn definitions inside impl blocks
    if extension == '.rs':
        return find_rust_functions(lines)
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this line matches any function pattern
        matched = False
        for pattern in compiled_patterns:
            if pattern.match(line):
                matched = True
                break
        
        if matched:
            start_line = i + 1  # Convert to 1-indexed
            signature = line.strip()
            
            # Find the end of the function
            if delimiters:
                open_char, close_char = delimiters
                end_line = find_bracket_end(lines, i, open_char, close_char)
            elif extension == '.py':
                end_line = find_python_block_end(lines, i)
            elif extension == '.toml':
                end_line = find_toml_section_end(lines, i)
            else:
                # Default: assume function ends at next function or EOF
                end_line = len(lines)
                for j in range(i + 1, len(lines)):
                    for pattern in compiled_patterns:
                        if pattern.match(lines[j]):
                            end_line = j
                            break
                    if end_line != len(lines):
                        break
            
            # Extract function name from signature
            name = extract_function_name(signature, extension)
            
            functions.append(FunctionInfo(
                name=name,
                start_line=start_line,
                end_line=end_line,
                signature=signature
            ))
        
        i += 1
    
    return functions


def find_rust_functions(lines: list[str]) -> list[FunctionInfo]:
    """
    Find all Rust functions, including those inside impl blocks.
    This provides better granularity than just finding impl blocks.
    Also includes preceding comments, doc comments, and attributes as part of the function.
    For functions inside impl blocks, includes the impl signature for unique identification.
    """
    functions = []
    
    # Pattern to match fn definitions (including pub, async, etc.)
    fn_pattern = re.compile(r'^(\s*)(pub(\s*\([^)]*\))?\s+)?(async\s+)?fn\s+(\w+)')
    # Pattern to match impl blocks - including standalone 'impl' at line start
    impl_pattern = re.compile(r'^(\s*)impl\b')
    # Pattern to match attributes (including test, derive, etc.)
    attr_pattern = re.compile(r'^\s*#\[')
    # Pattern to match comments (// or ///)
    comment_pattern = re.compile(r'^\s*(//|/\*)')
    # Pattern for empty/whitespace lines
    empty_pattern = re.compile(r'^\s*$')
    
    # First, find all impl blocks and their ranges (handles multi-line impl declarations)
    impl_blocks = []  # List of (start_line, end_line, impl_signature)
    for i, line in enumerate(lines):
        if impl_pattern.match(line):
            # Collect the full impl signature (may span multiple lines until we find '{')
            impl_lines = [line.strip()]
            j = i
            # If the current line doesn't have '{', look at subsequent lines
            while '{' not in lines[j] and j < len(lines) - 1:
                j += 1
                next_line = lines[j].strip()
                if next_line and not next_line.startswith('//'):
                    impl_lines.append(next_line)
            
            # Join and clean up the impl signature
            full_impl_sig = ' '.join(impl_lines)
            # Extract just the impl part (remove leading 'impl' and trailing '{' and content after)
            if '{' in full_impl_sig:
                full_impl_sig = full_impl_sig.split('{')[0]
            full_impl_sig = full_impl_sig.strip()
            if full_impl_sig.startswith('impl'):
                full_impl_sig = full_impl_sig[4:].strip()  # Remove 'impl' prefix
            
            end_line = find_rust_fn_end(lines, i)
            impl_blocks.append((i + 1, end_line, full_impl_sig))  # 1-indexed
    
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check for fn definition
        fn_match = fn_pattern.match(line)
        if fn_match:
            fn_name = fn_match.group(5)
            fn_line_idx = i
            fn_line_1indexed = i + 1
            
            # Find containing impl block (if any)
            containing_impl = None
            for impl_start, impl_end, impl_sig in impl_blocks:
                if impl_start < fn_line_1indexed <= impl_end:
                    containing_impl = impl_sig
                    break
            
            # Create a unique identifier that includes impl context
            if containing_impl:
                unique_name = f"{fn_name}@impl {containing_impl}"
            else:
                unique_name = fn_name
            
            # Look backwards to find the start of associated comments/attributes
            start_idx = i
            j = i - 1
            while j >= 0:
                prev_line = lines[j]
                # Include attributes (#[...])
                if attr_pattern.match(prev_line):
                    start_idx = j
                    j -= 1
                    continue
                # Include doc comments (/// or //!) and regular comments
                if comment_pattern.match(prev_line):
                    start_idx = j
                    j -= 1
                    continue
                # Include empty lines that are between comments/attrs
                if empty_pattern.match(prev_line):
                    # Check if there's more relevant content above
                    if j > 0 and (attr_pattern.match(lines[j-1]) or comment_pattern.match(lines[j-1])):
                        start_idx = j
                        j -= 1
                        continue
                # Stop at anything else
                break
            
            start_line = start_idx + 1  # Convert to 1-indexed
            
            # Find the end of the function by matching braces
            end_line = find_rust_fn_end(lines, fn_line_idx)
            
            signature = line.strip()
            # Include impl context in signature for display
            if containing_impl:
                signature = f"{signature}  // in impl {containing_impl}"
            
            functions.append(FunctionInfo(
                name=unique_name,
                start_line=start_line,
                end_line=end_line,
                signature=signature
            ))
        
        i += 1
    
    return functions


def find_rust_containers(lines: list[str]) -> list[FunctionInfo]:
    """
    Find Rust container types: struct, enum, impl, trait, mod.
    These are used to provide context for changes outside of functions.
    Returns FunctionInfo objects (reusing the same structure for simplicity).
    For impl blocks, captures the full signature to differentiate between different impls.
    """
    containers = []
    
    # Patterns for container types
    struct_pattern = re.compile(r'^(\s*)(pub(\s*\([^)]*\))?\s+)?struct\s+(\w+)')
    enum_pattern = re.compile(r'^(\s*)(pub(\s*\([^)]*\))?\s+)?enum\s+(\w+)')
    # Capture the full impl signature including generics and trait bounds
    impl_pattern = re.compile(r'^(\s*)impl\s+(.+)$')
    trait_pattern = re.compile(r'^(\s*)(pub(\s*\([^)]*\))?\s+)?trait\s+(\w+)')
    mod_pattern = re.compile(r'^(\s*)(pub(\s*\([^)]*\))?\s+)?mod\s+(\w+)\s*\{')
    
    # Pattern for comments and attributes
    comment_pattern = re.compile(r'^\s*(//|/\*)')
    attr_pattern = re.compile(r'^\s*#\[')
    empty_pattern = re.compile(r'^\s*$')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for each container type
        match = None
        container_type = None
        name = None
        full_signature = None
        
        struct_match = struct_pattern.match(line)
        if struct_match:
            match = struct_match
            container_type = 'struct'
            name = struct_match.group(4)
            full_signature = f"struct {name}"
        
        if not match:
            enum_match = enum_pattern.match(line)
            if enum_match:
                match = enum_match
                container_type = 'enum'
                name = enum_match.group(4)
                full_signature = f"enum {name}"
        
        if not match:
            impl_match = impl_pattern.match(line)
            if impl_match:
                match = impl_match
                container_type = 'impl'
                # Get the full impl signature
                impl_sig = impl_match.group(2).strip()
                # Remove trailing { if present
                if impl_sig.endswith('{'):
                    impl_sig = impl_sig[:-1].strip()
                # For multi-line impl declarations, try to get more context
                full_impl_sig = impl_sig
                # If the line doesn't have a {, look at subsequent lines
                if '{' not in line:
                    for k in range(i + 1, min(i + 5, len(lines))):
                        next_line = lines[k].strip()
                        if '{' in next_line:
                            # Add content before the {
                            before_brace = next_line.split('{')[0].strip()
                            if before_brace:
                                full_impl_sig += ' ' + before_brace
                            break
                        elif next_line and not next_line.startswith('//'):
                            full_impl_sig += ' ' + next_line
                
                # Use the full signature as the name to uniquely identify impl blocks
                name = full_impl_sig
                full_signature = f"impl {full_impl_sig}"
        
        if not match:
            trait_match = trait_pattern.match(line)
            if trait_match:
                match = trait_match
                container_type = 'trait'
                name = trait_match.group(4)
                full_signature = f"trait {name}"
        
        if not match:
            mod_match = mod_pattern.match(line)
            if mod_match:
                match = mod_match
                container_type = 'mod'
                name = mod_match.group(4)
                full_signature = f"mod {name}"
        
        if match and name:
            container_line_idx = i
            
            # Look backwards to find the start of associated comments/attributes
            start_idx = i
            j = i - 1
            while j >= 0:
                prev_line = lines[j]
                if attr_pattern.match(prev_line):
                    start_idx = j
                    j -= 1
                    continue
                if comment_pattern.match(prev_line):
                    start_idx = j
                    j -= 1
                    continue
                if empty_pattern.match(prev_line):
                    if j > 0 and (attr_pattern.match(lines[j-1]) or comment_pattern.match(lines[j-1])):
                        start_idx = j
                        j -= 1
                        continue
                break
            
            start_line = start_idx + 1  # Convert to 1-indexed
            
            # Find the end of the container by matching braces
            end_line = find_rust_fn_end(lines, container_line_idx)
            
            containers.append(FunctionInfo(
                name=name,
                start_line=start_line,
                end_line=end_line,
                signature=full_signature or f"{container_type} {name}"
            ))
        
        i += 1
    
    return containers


def find_rust_fn_end(lines: list[str], start_idx: int) -> int:
    """Find the end of a Rust function by matching braces.
    
    Properly handles:
    - Line comments (//)
    - Block comments (/* */)
    - Regular strings ("...")
    - Raw strings (r#"..."#) including multi-line
    - Character literals ('x')
    - Lifetime annotations ('a, 'static)
    """
    depth = 0
    found_open = False
    in_string = False
    in_block_comment = False
    in_raw_string = False
    raw_string_hash_count = 0
    
    for i in range(start_idx, len(lines)):
        line = lines[i]
        j = 0
        
        while j < len(line):
            # Handle multi-line raw strings
            if in_raw_string:
                # Look for closing pattern: "# (with correct hash count)
                if line[j] == '"':
                    # Check for trailing hashes
                    matches = True
                    for h in range(raw_string_hash_count):
                        if j + 1 + h >= len(line) or line[j + 1 + h] != '#':
                            matches = False
                            break
                    if matches:
                        j += 1 + raw_string_hash_count
                        in_raw_string = False
                        continue
                j += 1
                continue
            
            # Handle block comments
            if in_block_comment:
                if j + 1 < len(line) and line[j:j+2] == '*/':
                    in_block_comment = False
                    j += 2
                    continue
                j += 1
                continue
            
            # Skip line comments
            if not in_string and j + 1 < len(line) and line[j:j+2] == '//':
                break  # Rest of line is comment
            
            # Start block comment
            if not in_string and j + 1 < len(line) and line[j:j+2] == '/*':
                in_block_comment = True
                j += 2
                continue
            
            char = line[j]
            
            # Handle string literals
            if char == '"' and not in_string:
                # Check for raw string r#"..."# or r"..."
                # Look backwards to see if preceded by 'r' and optional '#'s
                is_raw = False
                hash_count = 0
                if j > 0:
                    k = j - 1
                    while k >= 0 and line[k] == '#':
                        hash_count += 1
                        k -= 1
                    if k >= 0 and line[k] == 'r':
                        is_raw = True
                
                if is_raw:
                    # Start of raw string
                    in_raw_string = True
                    raw_string_hash_count = hash_count
                    j += 1  # Move past opening quote
                    continue
                else:
                    in_string = True
                    j += 1
                    continue
            
            if char == '"' and in_string:
                # Check if escaped
                num_backslashes = 0
                k = j - 1
                while k >= 0 and line[k] == '\\':
                    num_backslashes += 1
                    k -= 1
                if num_backslashes % 2 == 0:  # Not escaped
                    in_string = False
                j += 1
                continue
            
            # Skip content inside strings
            if in_string:
                j += 1
                continue
            
            # Handle character literals and lifetimes
            if char == "'":
                # Check if this is a lifetime annotation ('a, 'static, etc.)
                # Lifetimes: ' followed by identifier chars
                if j + 1 < len(line):
                    next_char = line[j + 1]
                    # If next char is alphanumeric or underscore, it's likely a lifetime
                    if next_char.isalpha() or next_char == '_':
                        # Skip the lifetime
                        j += 2
                        while j < len(line) and (line[j].isalnum() or line[j] == '_'):
                            j += 1
                        continue
                    # Character literal: 'x' or '\n' etc.
                    elif next_char == '\\':
                        # Escaped char like '\n', '\\'
                        j += 4 if j + 3 < len(line) and line[j + 3] == "'" else 3
                        continue
                    elif j + 2 < len(line) and line[j + 2] == "'":
                        # Simple char like 'a'
                        j += 3
                        continue
                j += 1
                continue
            
            # Count braces
            if char == '{':
                depth += 1
                found_open = True
            elif char == '}':
                depth -= 1
                if found_open and depth == 0:
                    return i + 1  # 1-indexed, inclusive
            
            j += 1
    
    return len(lines)


def find_bracket_end(lines: list[str], start_idx: int, open_char: str, close_char: str) -> int:
    """Find the end of a bracket-delimited block."""
    depth = 0
    found_open = False
    
    for i in range(start_idx, len(lines)):
        line = lines[i]
        # Simple bracket counting (doesn't handle strings/comments perfectly)
        for char in line:
            if char == open_char:
                depth += 1
                found_open = True
            elif char == close_char:
                depth -= 1
                if found_open and depth == 0:
                    return i + 1  # Convert to 1-indexed
    
    return len(lines)


def find_python_block_end(lines: list[str], start_idx: int) -> int:
    """Find the end of a Python indentation block."""
    if start_idx >= len(lines):
        return len(lines)
    
    # Get base indentation
    first_line = lines[start_idx]
    base_indent = len(first_line) - len(first_line.lstrip())
    
    # Find the first line with less or equal indentation (that's not empty)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            continue
        
        current_indent = len(line) - len(line.lstrip())
        
        # If we find a line with same or less indentation, the block ended
        if current_indent <= base_indent:
            return i  # Convert to 1-indexed is not needed here as we want line before
    
    return len(lines)


def find_toml_section_end(lines: list[str], start_idx: int) -> int:
    """Find the end of a TOML section."""
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        # Next section starts
        if re.match(r'^\[+[^\]]+\]+$', line):
            return i
    return len(lines)


def extract_function_name(signature: str, extension: str) -> str:
    """Extract the function/method name from a signature."""
    signature = signature.strip()
    
    if extension == '.rs':
        # Rust: fn function_name, impl Type, trait Name
        match = re.search(r'\bfn\s+(\w+)', signature)
        if match:
            return match.group(1)
        match = re.search(r'\bimpl(?:<[^>]+>)?\s+(\w+)', signature)
        if match:
            return f"impl {match.group(1)}"
        match = re.search(r'\btrait\s+(\w+)', signature)
        if match:
            return f"trait {match.group(1)}"
        match = re.search(r'\bstruct\s+(\w+)', signature)
        if match:
            return f"struct {match.group(1)}"
        match = re.search(r'\benum\s+(\w+)', signature)
        if match:
            return f"enum {match.group(1)}"
        match = re.search(r'\bmod\s+(\w+)', signature)
        if match:
            return f"mod {match.group(1)}"
            
    elif extension == '.py':
        match = re.search(r'\b(def|class)\s+(\w+)', signature)
        if match:
            return match.group(2)
            
    elif extension in ['.js', '.ts']:
        match = re.search(r'\bfunction\s+(\w+)', signature)
        if match:
            return match.group(1)
        match = re.search(r'\b(const|let|var)\s+(\w+)', signature)
        if match:
            return match.group(2)
        match = re.search(r'\bclass\s+(\w+)', signature)
        if match:
            return match.group(1)
    
    # Fallback: return first word-like part
    match = re.search(r'\b(\w+)', signature)
    return match.group(1) if match else "unknown"


def get_changed_line_numbers(hunk: DiffHunk, for_new_file: bool = True) -> list[int]:
    """Get the line numbers that were actually changed in a hunk."""
    changed_lines = []
    
    if for_new_file:
        line_num = hunk.new_start
    else:
        line_num = hunk.old_start
    
    for line in hunk.lines:
        if for_new_file:
            if line.startswith('+'):
                changed_lines.append(line_num)
                line_num += 1
            elif line.startswith('-'):
                continue  # Deleted lines don't exist in new file
            else:
                line_num += 1
        else:
            if line.startswith('-'):
                changed_lines.append(line_num)
                line_num += 1
            elif line.startswith('+'):
                continue  # Added lines don't exist in old file
            else:
                line_num += 1
    
    return changed_lines


def find_containing_functions(changed_lines: list[int], functions: list[FunctionInfo]) -> list[FunctionInfo]:
    """Find all functions that contain any of the changed lines."""
    containing = []
    
    for func in functions:
        for line_num in changed_lines:
            if func.start_line <= line_num <= func.end_line:
                if func not in containing:
                    containing.append(func)
                break
    
    return containing


def generate_context_diff_for_file(
    file_diff: FileDiff,
    old_ref: Optional[str] = None,
    new_ref: Optional[str] = None
) -> str:
    """Generate a context-aware diff for a single file."""
    output_lines = []
    
    filepath = file_diff.new_path if not file_diff.is_deleted else file_diff.old_path
    extension = get_file_extension(filepath)
    
    # File header
    output_lines.append(f"diff --git a/{file_diff.old_path} b/{file_diff.new_path}")
    
    if file_diff.is_new:
        output_lines.append("new file mode 100644")
    elif file_diff.is_deleted:
        output_lines.append("deleted file mode 100644")
    
    if file_diff.is_binary:
        output_lines.append(f"Binary files a/{file_diff.old_path} and b/{file_diff.new_path} differ")
        return '\n'.join(output_lines)
    
    output_lines.append(f"--- a/{file_diff.old_path}")
    output_lines.append(f"+++ b/{file_diff.new_path}")
    
    # Get file contents
    if file_diff.is_new:
        old_content = []
        new_content = get_file_content(file_diff.new_path, new_ref) or []
    elif file_diff.is_deleted:
        old_content = get_file_content(file_diff.old_path, old_ref) or []
        new_content = []
    else:
        old_content = get_file_content(file_diff.old_path, old_ref) or []
        new_content = get_file_content(file_diff.new_path, new_ref) or []
    
    # Find functions in the new content
    new_functions = find_function_boundaries(new_content, extension)
    old_functions = find_function_boundaries(old_content, extension)
    
    # Collect all regions that need to be shown
    regions_to_show = set()  # (start_line, end_line) tuples for new file
    old_regions_to_show = set()  # for deleted lines context
    
    for hunk in file_diff.hunks:
        # Get changed lines
        new_changed_lines = get_changed_line_numbers(hunk, for_new_file=True)
        old_changed_lines = get_changed_line_numbers(hunk, for_new_file=False)
        
        # Find containing functions in new file
        containing_new = find_containing_functions(new_changed_lines, new_functions)
        for func in containing_new:
            regions_to_show.add((func.start_line, func.end_line))
        
        # Find containing functions in old file (for deleted code context)
        containing_old = find_containing_functions(old_changed_lines, old_functions)
        for func in containing_old:
            old_regions_to_show.add((func.start_line, func.end_line))
        
        # If no functions found, at least show some context around the change
        if not containing_new and new_changed_lines:
            min_line = max(1, min(new_changed_lines) - 5)
            max_line = min(len(new_content), max(new_changed_lines) + 5)
            regions_to_show.add((min_line, max_line))
        
        if not containing_old and old_changed_lines:
            min_line = max(1, min(old_changed_lines) - 5)
            max_line = min(len(old_content), max(old_changed_lines) + 5)
            old_regions_to_show.add((min_line, max_line))
    
    # Merge overlapping regions
    regions_to_show = merge_regions(list(regions_to_show))
    old_regions_to_show = merge_regions(list(old_regions_to_show))
    
    # Generate unified diff for each region
    for hunk in file_diff.hunks:
        new_changed = set(get_changed_line_numbers(hunk, for_new_file=True))
        old_changed = set(get_changed_line_numbers(hunk, for_new_file=False))
        
        # Find the region this hunk belongs to
        hunk_region = None
        for start, end in regions_to_show:
            if any(start <= line <= end for line in new_changed):
                hunk_region = (start, end)
                break
        
        if not hunk_region and new_changed:
            hunk_region = (min(new_changed), max(new_changed))
        
        if not hunk_region:
            # Fallback for deletions
            for start, end in old_regions_to_show:
                if any(start <= line <= end for line in old_changed):
                    hunk_region = (start, end)
                    break
        
        if hunk_region:
            start, end = hunk_region
            
            # Build the expanded hunk
            expanded_lines = []
            
            # Add function header comment
            containing_funcs = find_containing_functions(list(new_changed), new_functions)
            if containing_funcs:
                func_names = ', '.join(f.name for f in containing_funcs)
                expanded_lines.append(f"@@ Function context: {func_names} @@")
            
            # Generate the actual diff lines for this region
            # This is a simplified version - for full accuracy we'd need a proper diff algorithm
            expanded_lines.append(f"@@ -{start},{end - start + 1} +{start},{end - start + 1} @@")
            
            # Show context before
            context_start = max(1, start - 3)
            for i in range(context_start, start):
                if i - 1 < len(new_content):
                    expanded_lines.append(f" {new_content[i - 1]}")
            
            # Show the actual changes with surrounding context
            for line in hunk.lines:
                expanded_lines.append(line if line else ' ')
            
            # Show context after 
            context_end = min(len(new_content), end + 3)
            for i in range(end + 1, context_end + 1):
                if i - 1 < len(new_content):
                    expanded_lines.append(f" {new_content[i - 1]}")
            
            output_lines.extend(expanded_lines)
    
    return '\n'.join(output_lines)


def merge_regions(regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent regions."""
    if not regions:
        return []
    
    sorted_regions = sorted(regions)
    merged = [sorted_regions[0]]
    
    for start, end in sorted_regions[1:]:
        last_start, last_end = merged[-1]
        
        # If overlapping or adjacent (within 5 lines), merge
        if start <= last_end + 5:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    
    return merged


def generate_enhanced_diff(
    diff_output: str,
    old_ref: Optional[str] = None,
    new_ref: Optional[str] = None
) -> str:
    """Generate an enhanced diff with full function context."""
    file_diffs = parse_git_diff(diff_output)
    
    if not file_diffs:
        return diff_output
    
    enhanced_parts = []
    
    # Add header explaining this is an enhanced diff
    enhanced_parts.append("# Enhanced Git Diff with Full Function Context")
    enhanced_parts.append("# This diff shows complete functions/methods that contain changes")
    enhanced_parts.append("# to help understand the full context of modifications.")
    enhanced_parts.append("")
    
    for file_diff in file_diffs:
        enhanced_parts.append(generate_context_diff_for_file(file_diff, old_ref, new_ref))
        enhanced_parts.append("")  # Blank line between files
    
    return '\n'.join(enhanced_parts)


def generate_readable_diff(
    diff_output: str,
    old_ref: Optional[str] = None,
    new_ref: Optional[str] = None
) -> str:
    """Generate a more readable diff format optimized for LLMs.
    
    This function:
    1. Parses the git diff to find exactly which lines changed
    2. For each changed line, finds the containing function
    3. Shows OLD VERSION and NEW VERSION for modified functions
    4. Shows DELETED for functions that were removed entirely
    5. Shows NEW for functions that were added entirely  
    6. Shows raw changes for code outside any function
    """
    file_diffs = parse_git_diff(diff_output)
    
    if not file_diffs:
        return "No changes detected."
    
    output_parts = []
    output_parts.append("=" * 80)
    output_parts.append("CONTEXT-AWARE GIT DIFF")
    output_parts.append("This diff shows complete functions/methods containing the changes.")
    output_parts.append("=" * 80)
    output_parts.append("")
    
    for file_diff in file_diffs:
        filepath = file_diff.new_path if not file_diff.is_deleted else file_diff.old_path
        extension = get_file_extension(filepath)
        
        output_parts.append("-" * 80)
        output_parts.append(f"FILE: {filepath}")
        
        if file_diff.is_new:
            output_parts.append("STATUS: New file")
        elif file_diff.is_deleted:
            output_parts.append("STATUS: Deleted file")
        else:
            output_parts.append("STATUS: Modified")
        
        if file_diff.is_binary:
            output_parts.append("(Binary file)")
            output_parts.append("")
            continue
        
        output_parts.append("-" * 80)
        output_parts.append("")
        
        # Get file contents
        if file_diff.is_new:
            old_content = []
            new_content = get_file_content(file_diff.new_path, new_ref) or []
        elif file_diff.is_deleted:
            old_content = get_file_content(file_diff.old_path, old_ref) or []
            new_content = []
        else:
            old_content = get_file_content(file_diff.old_path, old_ref) or []
            new_content = get_file_content(file_diff.new_path, new_ref) or []
        
        # Find functions in both versions
        new_functions = find_function_boundaries(new_content, extension)
        old_functions = find_function_boundaries(old_content, extension)
        
        # Get ALL changed lines from the diff
        all_new_changed_lines = set()  # Lines added/modified in new file
        all_old_changed_lines = set()  # Lines deleted/modified from old file
        
        for hunk in file_diff.hunks:
            all_new_changed_lines.update(get_changed_line_numbers(hunk, for_new_file=True))
            all_old_changed_lines.update(get_changed_line_numbers(hunk, for_new_file=False))
        
        # Track which functions contain changes
        # Key: function name, Value: (old_func or None, new_func or None, change_type)
        # change_type: 'modified', 'deleted', 'new'
        affected_functions = {}
        
        # Find NEW file functions that have added lines
        new_lines_in_functions = set()
        for func in new_functions:
            func_lines = set(range(func.start_line, func.end_line + 1))
            matching_lines = all_new_changed_lines & func_lines
            if matching_lines:
                new_lines_in_functions.update(matching_lines)
                # Check if this function existed in old file (by name)
                old_func = next((of for of in old_functions if of.name == func.name), None)
                if old_func:
                    affected_functions[func.name] = (old_func, func, 'modified')
                else:
                    affected_functions[func.name] = (None, func, 'new')
        
        # Find OLD file functions that have deleted lines
        old_lines_in_functions = set()
        for func in old_functions:
            func_lines = set(range(func.start_line, func.end_line + 1))
            matching_lines = all_old_changed_lines & func_lines
            if matching_lines:
                old_lines_in_functions.update(matching_lines)
                if func.name not in affected_functions:
                    # Check if this function exists in new file
                    new_func = next((nf for nf in new_functions if nf.name == func.name), None)
                    if new_func:
                        affected_functions[func.name] = (func, new_func, 'modified')
                    else:
                        affected_functions[func.name] = (func, None, 'deleted')
                elif affected_functions[func.name][0] is None:
                    # We found it in new but not old - update with old func
                    _, new_func, _ = affected_functions[func.name]
                    affected_functions[func.name] = (func, new_func, 'modified')
        
        # Find changes outside of any function
        new_lines_outside_functions = all_new_changed_lines - new_lines_in_functions
        old_lines_outside_functions = all_old_changed_lines - old_lines_in_functions
        
        # === Output the affected functions ===
        
        # Sort by the earliest line number (prefer new file line numbers)
        sorted_funcs = sorted(
            affected_functions.items(),
            key=lambda x: (x[1][1].start_line if x[1][1] else x[1][0].start_line if x[1][0] else 0)
        )
        
        for func_name, (old_func, new_func, change_type) in sorted_funcs:
            if change_type == 'modified':
                # Show both OLD and NEW versions
                if old_func:
                    output_parts.append(f"### Function: {func_name} [OLD VERSION]")
                    output_parts.append("")
                    for i in range(old_func.start_line - 1, min(old_func.end_line, len(old_content))):
                        output_parts.append(old_content[i] if i < len(old_content) else "")
                    output_parts.append("")
                
                if new_func:
                    output_parts.append(f"### Function: {func_name} [NEW VERSION]")
                    output_parts.append("")
                    for i in range(new_func.start_line - 1, min(new_func.end_line, len(new_content))):
                        output_parts.append(new_content[i] if i < len(new_content) else "")
                    output_parts.append("")
                    
            elif change_type == 'deleted':
                # Function was deleted - only show old version
                output_parts.append(f"### Function: {func_name} [DELETED]")
                output_parts.append("")
                for i in range(old_func.start_line - 1, min(old_func.end_line, len(old_content))):
                    output_parts.append(old_content[i] if i < len(old_content) else "")
                output_parts.append("")
                
            elif change_type == 'new':
                # Function was added - only show new version
                output_parts.append(f"### Function: {func_name} [NEW]")
                output_parts.append("")
                for i in range(new_func.start_line - 1, min(new_func.end_line, len(new_content))):
                    output_parts.append(new_content[i] if i < len(new_content) else "")
                output_parts.append("")
        
        # === Show changes outside of functions ===
        # Filter out empty/whitespace-only lines from outside function changes
        def has_content(line_num, content):
            if line_num - 1 < len(content):
                return content[line_num - 1].strip() != ''
            return False
        
        old_lines_outside_filtered = {ln for ln in old_lines_outside_functions if has_content(ln, old_content)}
        new_lines_outside_filtered = {ln for ln in new_lines_outside_functions if has_content(ln, new_content)}
        
        if old_lines_outside_filtered or new_lines_outside_filtered:
            # For Rust, try to find containing structs/impls/enums for these changes
            if extension == '.rs':
                # Find struct/impl/enum/trait blocks that contain these changes
                old_containers = find_rust_containers(old_content)
                new_containers = find_rust_containers(new_content)
                
                # Track which containers have changes
                affected_containers = {}
                
                # Check new file containers
                for container in new_containers:
                    container_lines = set(range(container.start_line, container.end_line + 1))
                    if container_lines & new_lines_outside_filtered:
                        old_container = next((oc for oc in old_containers if oc.name == container.name), None)
                        if old_container:
                            affected_containers[container.name] = (old_container, container, 'modified')
                        else:
                            affected_containers[container.name] = (None, container, 'new')
                
                # Check old file containers for deletions
                for container in old_containers:
                    container_lines = set(range(container.start_line, container.end_line + 1))
                    if container_lines & old_lines_outside_filtered:
                        if container.name not in affected_containers:
                            new_container = next((nc for nc in new_containers if nc.name == container.name), None)
                            if new_container:
                                affected_containers[container.name] = (container, new_container, 'modified')
                            else:
                                affected_containers[container.name] = (container, None, 'deleted')
                
                # Output affected containers
                for container_name, (old_cont, new_cont, change_type) in sorted(
                    affected_containers.items(),
                    key=lambda x: (x[1][1].start_line if x[1][1] else x[1][0].start_line if x[1][0] else 0)
                ):
                    if change_type == 'modified':
                        if old_cont:
                            output_parts.append(f"### {old_cont.signature.split()[0].title()}: {container_name} [OLD VERSION]")
                            output_parts.append("")
                            for i in range(old_cont.start_line - 1, min(old_cont.end_line, len(old_content))):
                                output_parts.append(old_content[i] if i < len(old_content) else "")
                            output_parts.append("")
                        
                        if new_cont:
                            output_parts.append(f"### {new_cont.signature.split()[0].title()}: {container_name} [NEW VERSION]")
                            output_parts.append("")
                            for i in range(new_cont.start_line - 1, min(new_cont.end_line, len(new_content))):
                                output_parts.append(new_content[i] if i < len(new_content) else "")
                            output_parts.append("")
                    
                    elif change_type == 'deleted':
                        output_parts.append(f"### {old_cont.signature.split()[0].title()}: {container_name} [DELETED]")
                        output_parts.append("")
                        for i in range(old_cont.start_line - 1, min(old_cont.end_line, len(old_content))):
                            output_parts.append(old_content[i] if i < len(old_content) else "")
                        output_parts.append("")
                    
                    elif change_type == 'new':
                        output_parts.append(f"### {new_cont.signature.split()[0].title()}: {container_name} [NEW]")
                        output_parts.append("")
                        for i in range(new_cont.start_line - 1, min(new_cont.end_line, len(new_content))):
                            output_parts.append(new_content[i] if i < len(new_content) else "")
                        output_parts.append("")
                
                # Find lines that are still not covered by any container
                covered_old = set()
                covered_new = set()
                for _, (old_cont, new_cont, _) in affected_containers.items():
                    if old_cont:
                        covered_old.update(range(old_cont.start_line, old_cont.end_line + 1))
                    if new_cont:
                        covered_new.update(range(new_cont.start_line, new_cont.end_line + 1))
                
                remaining_old = old_lines_outside_filtered - covered_old
                remaining_new = new_lines_outside_filtered - covered_new
                
                if remaining_old or remaining_new:
                    output_parts.append("### Other changes:")
                    output_parts.append("")
                    if remaining_old:
                        output_parts.append("#### OLD VERSION:")
                        output_parts.append("")
                        for line_num in sorted(remaining_old):
                            if line_num - 1 < len(old_content):
                                output_parts.append(old_content[line_num - 1])
                        output_parts.append("")
                    if remaining_new:
                        output_parts.append("#### NEW VERSION:")
                        output_parts.append("")
                        for line_num in sorted(remaining_new):
                            if line_num - 1 < len(new_content):
                                output_parts.append(new_content[line_num - 1])
                        output_parts.append("")
            else:
                # Non-Rust files: show old and new versions
                output_parts.append("### Changes outside of functions:")
                output_parts.append("")
                
                if old_lines_outside_filtered:
                    output_parts.append("#### OLD VERSION:")
                    output_parts.append("")
                    for line_num in sorted(old_lines_outside_filtered):
                        if line_num - 1 < len(old_content):
                            output_parts.append(old_content[line_num - 1])
                    output_parts.append("")
                
                if new_lines_outside_filtered:
                    output_parts.append("#### NEW VERSION:")
                    output_parts.append("")
                    for line_num in sorted(new_lines_outside_filtered):
                        if line_num - 1 < len(new_content):
                            output_parts.append(new_content[line_num - 1])
                    output_parts.append("")
    
    return '\n'.join(output_parts)


def main():
    parser = argparse.ArgumentParser(
        description='Generate context-aware git diffs with complete function context',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Diff of unstaged changes
  %(prog)s --staged                 # Diff of staged changes  
  %(prog)s HEAD~1                   # Diff from last commit
  %(prog)s main                     # Diff against main branch
  %(prog)s HEAD~3 HEAD              # Diff between commits
  %(prog)s -o context.diff          # Save to file
  %(prog)s --format=readable        # Human-readable format (default)
  %(prog)s --format=unified         # Traditional unified diff format
        """
    )
    
    parser.add_argument(
        'refs',
        nargs='*',
        help='Git refs to diff (commit, branch, HEAD~N, etc.)'
    )
    parser.add_argument(
        '--staged', '-s',
        action='store_true',
        help='Show staged changes'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['readable', 'unified'],
        default='readable',
        help='Output format (default: readable)'
    )
    parser.add_argument(
        '--no-color',
        action='store_true',
        help='Disable color output'
    )
    
    args = parser.parse_args()
    
    # Build git diff command
    git_args = ['diff']
    
    if args.staged:
        git_args.append('--cached')
    
    old_ref = None
    new_ref = None
    
    if len(args.refs) == 0:
        # Diff working directory vs index (unstaged) or index vs HEAD (staged)
        if args.staged:
            old_ref = 'HEAD'
        pass  # git diff with no args
    elif len(args.refs) == 1:
        # Diff against a single ref
        git_args.append(args.refs[0])
        old_ref = args.refs[0]
    elif len(args.refs) == 2:
        # Diff between two refs
        git_args.extend(args.refs)
        old_ref = args.refs[0]
        new_ref = args.refs[1]
    else:
        parser.error("Too many refs specified (max 2)")
    
    # Run git diff
    returncode, stdout, stderr = run_git_command(git_args)
    
    if returncode != 0:
        print(f"Error running git diff: {stderr}", file=sys.stderr)
        sys.exit(1)
    
    if not stdout.strip():
        print("No changes detected.")
        sys.exit(0)
    
    # Generate enhanced diff
    if args.format == 'readable':
        enhanced = generate_readable_diff(stdout, old_ref, new_ref)
    else:
        enhanced = generate_enhanced_diff(stdout, old_ref, new_ref)
    
    # Output
    if args.output:
        Path(args.output).write_text(enhanced)
        print(f"Enhanced diff written to: {args.output}")
    else:
        print(enhanced)


if __name__ == '__main__':
    main()
