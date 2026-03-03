from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from oldbailey.model.schema import Speech


def iter_speeches_from_obv2_zip(obv2_zip_path: str | Path) -> Iterable[Speech]:
    """
    Yield `Speech` records from an Old Bailey Voices OBV2 ZIP.

    OBV2 distributions commonly include TSVs representing either:
    - utterance-level rows (one row per speech/utterance), or
    - word-level rows (multiple rows per utterance, must be grouped/ordered).

    TODO:
    - Inspect ZIP members and identify the expected TSV(s).
    - Implement TSV parsing and grouping into ordered speeches per case_id.
    - Decide on a canonical mapping from OBV2 identifiers to `case_id`.
    """

    raise NotImplementedError("TODO: implement OBV2 ZIP parsing (iter_speeches_from_obv2_zip).")

