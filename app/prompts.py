"""
Prompt builder for legal fiction: continuous prose with Hero's Journey as internal structure.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

MODE_BLOCKS = {
    "dark": "Tone: fatalism, brutality, bleak atmosphere. The story should feel grim and inexorable.",
    "sympathetic_defendant": "Emphasize structural injustice and humanizing interiority of the defendant.",
    "victim_centered": "Center harm, fear, aftermath, and dignity of the victim(s).",
    "courtroom_focused": "Emphasize rhetoric, exchanges, and procedural tension in the courtroom.",
    "pamphlet_style": "Use a 17th-century moralizing pamphlet voice with occasional direct address to the Reader.",
}

FULL_TEXT_TRUNCATE = int(os.environ.get("PROMPT_FULL_TEXT_TRUNCATE", "5000"))

# Stage labels that should not appear in prose output (used for validation)
STAGE_LABELS = [
    "Ordinary World",
    "Call to Adventure",
    "Refusal of the Call",
    "Meeting the Mentor",
    "Crossing the Threshold",
    "Tests, Allies, Enemies",
    "Approach to the Inmost Cave",
    "Ordeal",
    "Reward (Seizing the Sword)",
    "The Road Back",
    "Resurrection",
    "Return with the Elixir",
]


def build_story_prompt(
    case_card: dict[str, Any],
    full_text: str,
    mode: str,
    target_length: str,
) -> str:
    """
    Build a prompt for legal fiction: continuous prose, Hero's Journey as internal arc.
    Output must be narrative only — no headings, bullets, or metadata.
    """
    year = case_card.get("year")
    year_int = int(year) if year is not None else None

    # 1) ROLE + CONSTRAINTS
    role_block = """You are a historical legal fiction writer and courtroom dramatist.
Your story MUST be grounded in the provided Old Bailey case facts.
You MUST NOT contradict: crime/offence, victim, verdict, punishment, or general event description.
Maintain period-appropriate tone:
"""
    if year_int is not None and year_int < 1700:
        role_block += "- Use early modern diction (no modern slang), period-appropriate legal vocabulary (arraignment, indictment, jury, verdict, Newgate, etc.).\n"
    elif year_int is not None and year_int < 1850:
        role_block += "- Use Georgian/Regency/Victorian-adjacent register (still not modern).\n"
    role_block += """- Do not include modern moralizing commentary or contemporary institutions.
- Do not reference "Old Bailey" unless it naturally appears in the source text.
"""

    # 2) OUTPUT FORMAT
    format_block = """Output ONLY continuous prose. No headings, bullet points, stage labels, or metadata in the final story.
Do not label Hero's Journey stages. Do not include an outline or provenance section.
"""

    # 3) NARRATIVE STRUCTURE (internal scaffolding)
    journey_block = """Structure your narrative internally using this arc (do not label stages in the output):
1. Ordinary World – setting and context (London, shop, street life)
2. Call to Adventure – the crime or disturbance occurs
3. Threshold – discovery or pursuit begins
4. Trials – witnesses, confusion, or chase
5. Ordeal – confrontation, arrest, or trial
6. Resolution – judgement and punishment
7. Reflection – moral or social observation

You may invent: dialogue, character motivations, bystanders, environmental description, internal thoughts, minor events between recorded facts.
"""

    # 4) LEGAL REALISM
    legal_block = """Include concrete procedural details consistent with the era: indictment read, plea, witnesses/evidence, jury deliberation, verdict, sentencing.
- If punishment = death: the reflection must be psychological/moral/spiritual.
- If punishment = transport: frame the ending as exile/inversion.
- If verdict = not guilty: preserve ambiguity; show tension and consequences despite acquittal.
"""

    # 5) LENGTH
    if target_length == "400-600":
        length_line = "Total length: 400–600 words."
    elif target_length == "1500-2500":
        length_line = "Total length: 1500–2500 words."
    else:
        length_line = "Total length: 800–1200 words."
    length_block = length_line + "\n"

    # 6) MODE
    mode_instruction = MODE_BLOCKS.get(mode, MODE_BLOCKS.get("courtroom_focused", ""))
    mode_block = f"Mode: {mode}\n{mode_instruction}\n"

    # 7) INPUT
    card_copy = {k: v for k, v in case_card.items() if k != "full_text"}
    card_json_str = json.dumps(card_copy, ensure_ascii=False, indent=2)

    full_text_trimmed = (full_text or "")[:FULL_TEXT_TRUNCATE]
    truncated_note = ""
    if len(full_text or "") > FULL_TEXT_TRUNCATE:
        truncated_note = "\n[Text truncated for length.]\n"
    full_block = f"```\n{full_text_trimmed}{truncated_note}\n```"

    input_block = f"""Case data as JSON:
```json
{card_json_str}
```

Full text of the case:
{full_block}
"""

    prompt = (
        role_block
        + "\n"
        + format_block
        + "\n"
        + journey_block
        + "\n"
        + legal_block
        + "\n"
        + length_block
        + "\n"
        + mode_block
        + "\n"
        + "---\n\n"
        + input_block
    )
    return prompt


def validate_story_prose(markdown: str) -> tuple[bool, str | None]:
    """
    Check that output appears to be continuous prose (no headings, bullets, stage labels).
    Returns (True, None) if prose-like, else (False, reason).
    """
    text = (markdown or "").strip()
    if not text:
        return (False, "Empty output")

    # Reject explicit stage headings
    for label in STAGE_LABELS:
        if re.search(r"#+\s*" + re.escape(label), text, re.IGNORECASE):
            return (False, f"Contains stage label: {label}")

    # Reject Markdown H2 headings
    if re.search(r"^##\s+", text, re.MULTILINE):
        return (False, "Contains Markdown H2 headings")

    # Reject bullet lists at line start
    if re.search(r"^\s*[-*]\s+", text, re.MULTILINE):
        return (False, "Contains bullet points")

    return (True, None)


def validate_story_has_twelve_stages(markdown: str) -> tuple[bool, list[str]]:
    """
    Deprecated: previously checked for 12-stage headings.
    Now delegates to validate_story_prose for backward compatibility.
    Returns (True, []) if prose-like, else (False, [reason]).
    """
    ok, reason = validate_story_prose(markdown)
    return (ok, [] if ok else [reason or "Failed prose validation"])
