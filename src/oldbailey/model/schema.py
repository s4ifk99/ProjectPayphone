from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Case(BaseModel):
    """
    Normalized case / trial record.

    Core fields: case_id, date_iso, year, court, offence_category, xml_path.
    All other XML-derived metadata goes into meta_json as a JSON string.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(..., min_length=1, description="Stable Old Bailey case identifier.")
    date_iso: str | None = Field(
        default=None, description="Best-effort ISO date string (YYYY-MM-DD) from source."
    )
    year: int | None = Field(default=None, description="Year extracted from date if available.")
    court: str | None = Field(default=None, description="Court name/code if present in XML.")
    offence_category: str | None = Field(
        default=None,
        description="Main offence category for the case (e.g. violentTheft, theft, burglary).",
    )
    xml_path: str = Field(default="", description="Absolute or relative path to source XML file.")
    meta_json: str = Field(
        default="{}",
        description="JSON string of non-core metadata (attributes, elements, etc.).",
    )
    source: str | None = Field(
        default=None, description="Source identifier (e.g., 'obo-xml')."
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra fields (legacy).")


class Speech(BaseModel):
    """
    Ordered speech/utterance text belonging to a case.

    TODO: When implementing the OBV2 and/or XML speech parser, decide whether
    `speech_no` corresponds to:
    - OBV2 utterance order within a case, or
    - XML speech segment order, or
    - a unified merged ordering strategy.
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(..., min_length=1, description="Foreign key to Case.case_id.")
    speech_no: int = Field(..., ge=1, description="1-based ordering within the case.")
    speaker_id: str | None = Field(default=None, description="Optional speaker identifier.")
    speaker_name: str | None = Field(default=None, description="Optional speaker name.")
    text: str = Field(..., min_length=1, description="Speech/utterance text.")
    source: str | None = Field(default=None, description="Source identifier (e.g., 'obv2').")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra fields.")

