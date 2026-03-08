"""
Strict prompt builder for legal fiction with Joseph Campbell 12-stage structure.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

REQUIRED_H2_HEADINGS = [
    "## 1. Ordinary World",
    "## 2. Call to Adventure",
    "## 3. Refusal of the Call",
    "## 4. Meeting the Mentor",
    "## 5. Crossing the Threshold",
    "## 6. Tests, Allies, Enemies",
    "## 7. Approach to the Inmost Cave",
    "## 8. Ordeal",
    "## 9. Reward (Seizing the Sword)",
    "## 10. The Road Back",
    "## 11. Resurrection",
    "## 12. Return with the Elixir",
]

THEMES_ALLOWED = [
    "Justice vs. Survival",
    "Reputation and Honor",
    "Poverty and Crime",
    "Gender and Power",
    "Authority and Resistance",
    "Fate vs. Agency",
    "Public Spectacle and Shame",
]

MODE_BLOCKS = {
    "dark": "Tone: fatalism, brutality, bleak atmosphere. The story should feel grim and inexorable.",
    "sympathetic_defendant": "Emphasize structural injustice and humanizing interiority of the defendant.",
    "victim_centered": "Center harm, fear, aftermath, and dignity of the victim(s).",
    "courtroom_focused": "Emphasize rhetoric, exchanges, and procedural tension in the courtroom.",
    "pamphlet_style": "Use a 17th-century moralizing pamphlet voice with occasional direct address to the Reader.",
}

FULL_TEXT_TRUNCATE = int(os.environ.get("PROMPT_FULL_TEXT_TRUNCATE", "5000"))


def build_story_prompt(
    case_card: dict[str, Any],
    full_text: str,
    mode: str,
    target_length: str,
) -> str:
    """
    Build a single prompt string containing role, constraints, hero rule, themes,
    12-stage headings, legal realism, length, mode, provenance, and input blocks.
    """
    year = case_card.get("year")
    year_int = int(year) if year is not None else None

    # 1) ROLE + CONSTRAINTS
    role_block = """You are a historical legal fiction writer and courtroom dramatist.
Your story MUST be grounded in the provided Old Bailey case facts.
You MUST NOT contradict: year, offence(s), verdict(s), punishment(s), places, or other explicit facts in the case data.
Maintain period-appropriate tone:
"""
    if year_int is not None and year_int < 1700:
        role_block += "- Use early modern diction (no modern slang), period-appropriate legal vocabulary (arraignment, indictment, jury, verdict, Newgate, etc.).\n"
    elif year_int is not None and year_int < 1850:
        role_block += "- Use Georgian/Regency/Victorian-adjacent register (still not modern).\n"
    role_block += """- Do not include modern moralizing commentary or contemporary institutions.
- Do not reference "Old Bailey" unless it naturally appears in the source text.
"""

    # 2) HERO SELECTION
    hero_block = """Choose exactly ONE protagonist: default = defendant; you may choose victim or witness if dramatically stronger.
Your output MUST begin with:
"Protagonist Chosen: …"
"Why This Perspective Works Dramatically: …" (2–3 sentences)
"""

    # 3) THEMES
    themes_list = "; ".join(THEMES_ALLOWED)
    themes_block = f"""You MUST explicitly name:
"Primary Theme: …"
"Secondary Undercurrent: …" (optional but encouraged)
Themes must be one or more of: {themes_list}.
"""

    # 4) 12-STAGE HERO'S JOURNEY
    headings_list = "\n".join(REQUIRED_H2_HEADINGS)
    journey_block = f"""Your output MUST contain the following headings EXACTLY (case-sensitive), in this order, as Markdown H2:
{headings_list}

Before the full narrative, include a compact outline section:
"Hero's Journey Outline:" followed by 12 bullet points, 1–2 sentences each, matching the stages above.
Then write the full narrative where each stage heading appears and contains the story prose for that stage.
"""

    # 5) LEGAL REALISM
    legal_block = """Include concrete procedural details consistent with the era: indictment read, plea, witnesses/evidence, jury deliberation, verdict, sentencing.
- If punishment = death: the "Resurrection" stage must be psychological/moral/spiritual, not literal survival.
- If punishment = transport: the "Return with the Elixir" must be framed as exile/inversion (what is "brought back" is lesson, letter, legend, or moral residue).
- If verdict = not guilty: preserve ambiguity; show tension and consequences despite acquittal.
"""

    # 6) LENGTH
    if target_length == "1500-2500":
        length_line = "Total length: 1500–2500 words."
    else:
        length_line = "Total length: 800–1200 words."
    length_block = length_line + "\n"

    # 7) MODE
    mode_instruction = MODE_BLOCKS.get(mode, MODE_BLOCKS.get("courtroom_focused", ""))
    mode_block = f"Mode: {mode}\n{mode_instruction}\n"

    # 8) PROVENANCE
    prov_block = """At the end of your output you MUST include a Markdown section:

## Provenance

Include: case_id, doc_id, year; offence categories/subcategories + offence text; verdict categories + verdict text; punishment categories + punishment text; places list; page_facsimiles list.
"Source excerpt:" include a direct excerpt from full_text, <= 120 words, quoted as a Markdown blockquote.
"Factual anchors used:" a bullet list of 6–12 specific facts drawn from the case data (names, locations, amounts, dates, verdict, etc.).
"""

    # 9) INPUT
    card_copy = {k: v for k, v in case_card.items() if k != "full_text"}
    card_json_str = json.dumps(card_copy, ensure_ascii=False, indent=2)

    full_text_trimmed = (full_text or "")[:FULL_TEXT_TRUNCATE]
    truncated_note = ""
    if len(full_text or "") > FULL_TEXT_TRUNCATE:
        truncated_note = "\n[Text truncated for length; use the excerpt in Provenance from the full source.]\n"
    full_block = f"```\n{full_text_trimmed}{truncated_note}\n```"

    input_block = f"""Case data as JSON:
```json
{card_json_str}
```

Full text of the case (use for facts and for Provenance excerpt):
{full_block}
"""

    prompt = (
        role_block
        + "\n"
        + hero_block
        + "\n"
        + themes_block
        + "\n"
        + journey_block
        + "\n"
        + legal_block
        + "\n"
        + length_block
        + "\n"
        + mode_block
        + "\n"
        + prov_block
        + "\n"
        + "---\n\n"
        + input_block
    )
    return prompt


def validate_story_has_twelve_stages(markdown: str) -> tuple[bool, list[str]]:
    """
    Check that the model output contains all 12 required H2 headings.
    Returns (True, []) if all present, else (False, list of missing headings).
    """
    missing: list[str] = []
    for heading in REQUIRED_H2_HEADINGS:
        # Require exact heading as a line (allowing optional trailing space)
        pattern = re.escape(heading) + r"\s*$"
        if not re.search(pattern, markdown, re.MULTILINE):
            missing.append(heading)
    return (len(missing) == 0, missing)
