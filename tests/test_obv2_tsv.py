"""Tests for OBV2 TSV parsing using in-memory zip fixtures."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from oldbailey.io.obv2_tsv import (
    ColumnMapping,
    infer_case_id,
    infer_column_mapping,
    infer_speaker,
    iter_speeches_from_obv2_zip,
)
from oldbailey.model.schema import Speech


def _make_zip(tsv_content: str, tsv_name: str = "data.tsv") -> Path:
    """Create an in-memory zip with a TSV, write to temp file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(tsv_name, tsv_content.encode("utf-8"))
    return buf


def _zip_to_file(buf: io.BytesIO, path: Path) -> Path:
    """Write zip buffer to file and return path."""
    path.write_bytes(buf.getvalue())
    return path


# Format A: one row per speech/utterance
FORMAT_A_TSV = """case_id\tspeaker\ttext\trow_id
t17800628-1\tJudge\tYou are charged with theft.\t1
t17800628-1\tDefendant\tI plead not guilty.\t2
t17800628-2\tProsecutor\tThe evidence shows otherwise.\t3"""

# Format B: one row per token with utt_id
FORMAT_B_TSV = """trial_id\tutt_id\ttoken\tspeaker
t17800628-1\tu1\tThe\tJudge
t17800628-1\tu1\tprisoner\tJudge
t17800628-1\tu1\tpleads\tJudge
t17800628-1\tu1\tnot\tJudge
t17800628-1\tu1\tguilty\tJudge
t17800628-1\tu2\tVery\tClerk
t17800628-1\tu2\twell\tClerk"""


def test_infer_column_mapping_utterance():
    """Format A: text column detected -> utterance format."""
    header = ["case_id", "speaker", "text", "row_id"]
    m = infer_column_mapping(header)
    assert m.format == "utterance"
    assert m.text_col == "text"
    assert m.case_id_col == "case_id"
    assert m.speaker_col == "speaker"


def test_infer_column_mapping_token():
    """Format B: token + utt_id detected -> token format."""
    header = ["trial_id", "utt_id", "word", "speaker"]
    m = infer_column_mapping(header)
    assert m.format == "token"
    assert m.token_col == "word"
    assert m.utt_id_col == "utt_id"
    assert m.case_id_col == "trial_id"


def test_infer_case_id():
    """infer_case_id picks from candidate columns."""
    assert infer_case_id({"case_id": "t1"}, None) == "t1"
    assert infer_case_id({"trial_id": "t2"}, None) == "t2"
    assert infer_case_id({"obo_id": "t3"}, None) == "t3"
    m = ColumnMapping(case_id_col="t")
    assert infer_case_id({"t": "t4"}, m) == "t4"


def test_infer_speaker():
    """infer_speaker picks from candidate columns."""
    assert infer_speaker({"speaker": "Judge"}, None) == "Judge"
    assert infer_speaker({"role": "Defendant"}, None) == "Defendant"
    assert infer_speaker({"who": "Clerk"}, None) == "Clerk"
    m = ColumnMapping(speaker_col="person")
    assert infer_speaker({"person": "Witness"}, m) == "Witness"


def test_iter_speeches_format_a(tmp_path):
    """Format A: one row per speech, yields Speech per row."""
    zip_buf = _make_zip(FORMAT_A_TSV)
    zip_path = tmp_path / "obv2_a.zip"
    _zip_to_file(zip_buf, zip_path)

    speeches = list(iter_speeches_from_obv2_zip(zip_path))
    assert len(speeches) == 3

    assert speeches[0].case_id == "t17800628-1"
    assert speeches[0].speech_no == 1
    assert speeches[0].speaker_name == "Judge"
    assert speeches[0].text == "You are charged with theft."
    assert speeches[0].source == "OBV2"
    assert speeches[0].metadata.get("source_ref") == "1"

    assert speeches[1].case_id == "t17800628-1"
    assert speeches[1].speech_no == 2
    assert speeches[1].speaker_name == "Defendant"
    assert speeches[1].text == "I plead not guilty."

    assert speeches[2].case_id == "t17800628-2"
    assert speeches[2].speech_no == 1
    assert speeches[2].text == "The evidence shows otherwise."


def test_iter_speeches_format_b(tmp_path):
    """Format B: token rows grouped by (case_id, utt_id)."""
    zip_buf = _make_zip(FORMAT_B_TSV)
    zip_path = tmp_path / "obv2_b.zip"
    _zip_to_file(zip_buf, zip_path)

    speeches = list(iter_speeches_from_obv2_zip(zip_path))
    assert len(speeches) == 2

    assert speeches[0].case_id == "t17800628-1"
    assert speeches[0].speech_no == 1
    assert speeches[0].speaker_name == "Judge"
    assert speeches[0].text == "The prisoner pleads not guilty"
    assert speeches[0].source == "OBV2"
    assert speeches[0].metadata.get("source_ref") == "u1"

    assert speeches[1].case_id == "t17800628-1"
    assert speeches[1].speech_no == 2
    assert speeches[1].speaker_name == "Clerk"
    assert speeches[1].text == "Very well"
    assert speeches[1].metadata.get("source_ref") == "u2"


def test_iter_speeches_alternate_column_names(tmp_path):
    """Alternate column names (obo_id, word, utterance_id) work."""
    tsv = "obo_id\tutterance_id\tword\twho\nt1\tu1\tHello\ttest\nt1\tu1\tworld\ttest"
    zip_buf = _make_zip(tsv)
    zip_path = tmp_path / "alt.zip"
    _zip_to_file(zip_buf, zip_path)

    speeches = list(iter_speeches_from_obv2_zip(zip_path))
    assert len(speeches) == 1
    assert speeches[0].case_id == "t1"
    assert speeches[0].text == "Hello world"
    assert speeches[0].speaker_name == "test"


def test_iter_speeches_no_tsv_raises(tmp_path):
    """ZIP without .tsv raises ValueError."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", b"no tsv here")
    zip_path = tmp_path / "empty.zip"
    zip_path.write_bytes(buf.getvalue())

    with pytest.raises(ValueError, match="No .tsv file"):
        list(iter_speeches_from_obv2_zip(zip_path))


def test_iter_speeches_unknown_format_raises(tmp_path):
    """TSV with neither text nor token+utt_id raises."""
    tsv = "col1\tcol2\nx\ty"
    zip_buf = _make_zip(tsv)
    zip_path = tmp_path / "bad.zip"
    _zip_to_file(zip_buf, zip_path)

    with pytest.raises(ValueError, match="Cannot infer format"):
        list(iter_speeches_from_obv2_zip(zip_path))


def test_iter_speeches_file_not_found():
    """Non-existent ZIP raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="ZIP not found"):
        list(iter_speeches_from_obv2_zip(Path("/nonexistent/obv2.zip")))
