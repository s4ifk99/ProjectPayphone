"""
Parse Old Bailey Voices OBV2 TSV from a ZIP archive.

Handles two formats:
- A) One row per speech/utterance (text column exists)
- B) One row per token with utterance id (group by case_id, utt_id)
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from oldbailey.model.schema import Speech

logger = logging.getLogger(__name__)

# Candidate column names for heuristics (case-insensitive match)
CASE_ID_CANDIDATES = (
    "trial_id",
    "obo_id",
    "obo_trial",
    "case_id",
    "t",
    "case",
    "trial",
    "document_id",
    "doc_id",
)
SPEAKER_CANDIDATES = (
    "speaker",
    "who",
    "role",
    "speaker_name",
    "speaker_role",
    "person",
    "speaker_role_label",
)
TEXT_CANDIDATES = (
    "text",
    "utterance",
    "speech",
    "utterance_text",
    "speech_text",
    "content",
    "transcript",
)
TOKEN_CANDIDATES = (
    "token",
    "word",
    "w",
    "token_text",
    "word_text",
    "form",
)
UTT_ID_CANDIDATES = (
    "utt_id",
    "speech_id",
    "u",
    "utterance_id",
    "speech_no",
    "seq",
    "seq_no",
)
ROW_ID_CANDIDATES = (
    "id",
    "row_id",
    "rowid",
)


def _normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(header: list[str], candidates: tuple[str, ...]) -> str | None:
    """Return first header (original case) matching a candidate, or None."""
    normalized = {_normalize_header(h): h for h in header}
    for c in candidates:
        if _normalize_header(c) in normalized:
            return normalized[_normalize_header(c)]
    for h in header:
        if _normalize_header(h) in (c.lower() for c in candidates):
            return h
    return None


@dataclass
class ColumnMapping:
    """Describes the inferred column mapping for an OBV2 TSV."""

    case_id_col: str | None = None
    speaker_col: str | None = None
    text_col: str | None = None  # format A: full speech text
    token_col: str | None = None  # format B: word/token
    utt_id_col: str | None = None  # format B: utterance grouping
    row_id_col: str | None = None
    header: list[str] = field(default_factory=list)
    format: str = "unknown"  # "utterance" or "token"


def infer_column_mapping(header: list[str]) -> ColumnMapping:
    """
    Infer column mapping from TSV header using heuristics.

    Returns a ColumnMapping describing which columns map to case_id, speaker,
    text (utterance-level), token, and utt_id.
    """
    mapping = ColumnMapping(header=list(header))
    mapping.case_id_col = _find_column(header, CASE_ID_CANDIDATES)
    mapping.speaker_col = _find_column(header, SPEAKER_CANDIDATES)
    mapping.text_col = _find_column(header, TEXT_CANDIDATES)
    mapping.token_col = _find_column(header, TOKEN_CANDIDATES)
    mapping.utt_id_col = _find_column(header, UTT_ID_CANDIDATES)
    mapping.row_id_col = _find_column(header, ROW_ID_CANDIDATES)

    if mapping.text_col:
        mapping.format = "utterance"
    elif mapping.token_col and mapping.utt_id_col:
        mapping.format = "token"
    else:
        mapping.format = "unknown"

    return mapping


def infer_case_id(row: dict[str, str], mapping: ColumnMapping | None = None) -> str:
    """
    Best-effort case_id from row using plausible columns.

    If mapping is provided, uses it; else tries candidate keys directly.
    """
    if mapping and mapping.case_id_col and mapping.case_id_col in row:
        v = row[mapping.case_id_col]
        if v and str(v).strip():
            return str(v).strip()
    for key in CASE_ID_CANDIDATES:
        for k, v in row.items():
            if _normalize_header(k) == _normalize_header(key) and v and str(v).strip():
                return str(v).strip()
    return ""


def infer_speaker(row: dict[str, str], mapping: ColumnMapping | None = None) -> str | None:
    """
    Best-effort speaker name/role from row using plausible columns.

    Returns combined speaker+role if both exist, else whichever is present.
    """
    parts: list[str] = []
    if mapping and mapping.speaker_col and mapping.speaker_col in row:
        v = row[mapping.speaker_col]
        if v and str(v).strip():
            parts.append(str(v).strip())
    if not parts:
        for key in SPEAKER_CANDIDATES:
            for k, v in row.items():
                if _normalize_header(k) == _normalize_header(key) and v and str(v).strip():
                    parts.append(str(v).strip())
                    break
    return " ".join(parts) if parts else None


def _get_source_ref(row: dict[str, str], mapping: ColumnMapping, utt_id: str | None = None) -> str:
    """Row id or utterance id for source_ref."""
    if utt_id:
        return str(utt_id)
    if mapping.row_id_col and mapping.row_id_col in row:
        v = row[mapping.row_id_col]
        if v:
            return str(v)
    return ""


def iter_speeches_from_obv2_zip(zip_path: str | Path) -> Iterator[Speech]:
    """
    Yield Speech records from an OBV2 ZIP containing a TSV.

    Streaming: reads the TSV line-by-line, does not load the whole file.
    Finds the first .tsv member in the ZIP.

    Produces Speech with:
    - case_id, speech_no (monotonic per case), speaker_name if available,
    - text, source="OBV2", metadata["source_ref"] = row/utt id
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        tsv_members = [m for m in zf.namelist() if m.lower().endswith(".tsv")]
        if not tsv_members:
            raise ValueError(f"No .tsv file found in ZIP: {zip_path}")
        tsv_name = tsv_members[0]
        logger.info("Using TSV member: %s", tsv_name)

        with zf.open(tsv_name) as f:
            # Decode as text
            content = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            reader = csv.DictReader(content, delimiter="\t")
            header = reader.fieldnames or []
            mapping = infer_column_mapping(header)

            if mapping.format == "unknown":
                raise ValueError(
                    f"Cannot infer format from columns {header}. "
                    "Need either a text column (utterance-level) or token+utt_id (token-level)."
                )

            case_seq: dict[str, int] = {}

            if mapping.format == "utterance":
                for row in reader:
                    case_id = infer_case_id(row, mapping)
                    if not case_id:
                        continue
                    text_val = row.get(mapping.text_col or "", "").strip()
                    if not text_val:
                        continue
                    case_seq[case_id] = case_seq.get(case_id, 0) + 1
                    speaker = infer_speaker(row, mapping)
                    source_ref = _get_source_ref(row, mapping)
                    yield Speech(
                        case_id=case_id,
                        speech_no=case_seq[case_id],
                        speaker_name=speaker,
                        text=text_val,
                        source="OBV2",
                        metadata={"source_ref": source_ref} if source_ref else {},
                    )
            else:
                # Token-level: group by (case_id, utt_id)
                buf: list[tuple[str, str | None, str]] = []  # (case_id, speaker, token)
                prev_key: tuple[str, str] | None = None
                tokens: list[str] = []

                for row in reader:
                    case_id = infer_case_id(row, mapping)
                    if not case_id:
                        continue
                    utt_id = row.get(mapping.utt_id_col or "", "").strip()
                    if not utt_id:
                        utt_id = str(len(buf))  # fallback: treat each row as own utterance
                    key = (case_id, utt_id)
                    token_val = row.get(mapping.token_col or "", "").strip()

                    if prev_key is not None and key != prev_key:
                        # Emit previous utterance
                        if tokens:
                            text = " ".join(tokens)
                            if text:
                                case_seq[prev_key[0]] = case_seq.get(prev_key[0], 0) + 1
                                speaker = buf[0][1] if buf else None
                                yield Speech(
                                    case_id=prev_key[0],
                                    speech_no=case_seq[prev_key[0]],
                                    speaker_name=speaker,
                                    text=text,
                                    source="OBV2",
                                    metadata={"source_ref": prev_key[1]},
                                )
                        buf = []
                        tokens = []

                    speaker = infer_speaker(row, mapping)
                    buf.append((case_id, speaker, token_val))
                    tokens.append(token_val)
                    prev_key = key

                # Emit last utterance
                if tokens:
                    text = " ".join(tokens)
                    if text and prev_key:
                        case_seq[prev_key[0]] = case_seq.get(prev_key[0], 0) + 1
                        speaker = buf[0][1] if buf else None
                        yield Speech(
                            case_id=prev_key[0],
                            speech_no=case_seq[prev_key[0]],
                            speaker_name=speaker,
                            text=text,
                            source="OBV2",
                            metadata={"source_ref": prev_key[1]},
                        )
