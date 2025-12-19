from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


# Markers present in Prompt.md / prompt_template.md to locate individual templates.
SECTION_MARKERS = {
    "human_approach": "Prompt for getting Human Approach from real diff",
    "prompt1": "Prompt to get First Prompt",
    "prompt2": "Prompt to get Second and Third Prompt",
    "evaluation": "Summarize Claude Code's approach, compare with real changes, and decide Pass/Fail",
}


@dataclass
class Templates:
    human_approach: str
    prompt1: str
    prompt2: str
    prompt3: str
    evaluation: str


def _split_sections(raw: str) -> Dict[str, str]:
    """
    Split the Prompt.md file into named sections using marker headings.
    This keeps templates read-only and unchanged.
    """
    sections: Dict[str, str] = {}
    lines = raw.splitlines()
    indices: Dict[str, int] = {}

    for idx, line in enumerate(lines):
        for key, marker in SECTION_MARKERS.items():
            if marker in line and key not in indices:
                indices[key] = idx

    sorted_keys = sorted(indices.items(), key=lambda kv: kv[1])
    for i, (key, start_idx) in enumerate(sorted_keys):
        end_idx = sorted_keys[i + 1][1] if i + 1 < len(sorted_keys) else len(lines)
        # Keep the section text including the marker line for fidelity.
        section_text = "\n".join(lines[start_idx:end_idx]).strip()
        sections[key] = section_text

    return sections


def load_templates(template_path: Path) -> Templates:
    """
    Load templates from Prompt.md without modifying their content.
    """
    raw = template_path.read_text(encoding="utf-8")
    sections = _split_sections(raw)
    
    required = ["human_approach", "prompt1", "prompt2", "evaluation"]
    missing = [k for k in required if k not in sections]
    if missing:
        raise ValueError(f"Missing template sections: {missing}")

    # prompt2 and prompt3 share the same template (iterative improvement)
    prompt2_text = sections.get("prompt2", "")
    
    return Templates(
        human_approach=sections["human_approach"],
        prompt1=sections["prompt1"],
        prompt2=prompt2_text,
        prompt3=prompt2_text,  # Same template for both iterative prompts
        evaluation=sections["evaluation"],
    )


def fill_placeholders(template: str, replacements: Dict[str, str]) -> str:
    """
    Replace placeholders in two styles:
    - {{PLACEHOLDER}}
    - <PLACEHOLDER>
    If a placeholder is absent, it is left untouched to avoid mutating the template.
    """
    rendered = template

    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
        rendered = rendered.replace(f"<{key}>", value)

    # Clean up any double blank lines introduced by empty replacements.
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    return rendered.strip() + "\n"
