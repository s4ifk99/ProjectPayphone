"""Tests for Old Bailey XML parsing using inline fixtures (no external files)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oldbailey.io.obo_xml import iter_cases_and_speeches_from_xml_dir, iter_cases_from_xml_dir
from oldbailey.model.schema import Case, Speech


# Minimal sessions paper: one trialAccount div1
SESSIONS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI.2>
  <text><body>
    <div0 type="sessionsPaper" id="16740429">
      <interp inst="16740429" type="date" value="16740429"/>
      <interp inst="16740429" type="year" value="1674"/>
      <div1 type="trialAccount" id="t16740429-1">
        <interp inst="t16740429-1" type="date" value="16740429"/>
        <interp inst="t16740429-1" type="year" value="1674"/>
        <interp inst="t16740429-1" type="uri" value="sessionsPapers/16740429"/>
      </div1>
      <div1 type="trialAccount" id="t16740429-2">
        <interp inst="t16740429-2" type="date" value="16740429"/>
      </div1>
    </div0>
  </body></text>
</TEI.2>"""

# Ordinary account: single div0 with offenceCategory
ORDINARY_SAMPLE = """<?xml version="1.0" encoding="ISO-8859-1"?>
<TEI.2>
  <text><body>
    <div0 type="ordinarysAccount" id="OA16760517">
      <interp inst="OA16760517" type="date" value="16760517"/>
      <interp inst="OA16760517" type="offenceCategory" value="violentTheft"/>
      <interp inst="OA16760517" type="uri" value="ordinarysAccounts/OA16760517"/>
    </div0>
  </body></text>
</TEI.2>"""

# Unknown schema: no div0/div1 with id, use file stem
UNKNOWN_SAMPLE = """<?xml version="1.0"?>
<root>
  <interp type="date" value="16900115"/>
  <interp type="year" value="1690"/>
</root>"""


def test_iter_cases_sessions_paper(tmp_path):
    """Sessions paper with trialAccount div1s yields one Case per div1."""
    xml_dir = tmp_path / "sessions"
    xml_dir.mkdir()
    (xml_dir / "16740429.xml").write_text(SESSIONS_SAMPLE, encoding="utf-8")

    cases = list(iter_cases_from_xml_dir(xml_dir))
    assert len(cases) == 2
    assert cases[0].case_id == "t16740429-1"
    assert cases[0].date_iso == "1674-04-29"
    assert cases[0].year == 1674
    # No offenceCategory in this sample
    assert cases[0].offence_category is None
    assert "16740429.xml" in cases[0].xml_path or "16740429" in cases[0].xml_path
    # meta_json holds non-core metadata (interps, attributes)
    meta = json.loads(cases[0].meta_json)
    assert "interp" in meta or "attributes" in meta

    assert cases[1].case_id == "t16740429-2"
    assert cases[1].date_iso == "1674-04-29"
    assert cases[1].year == 1674


def test_iter_cases_ordinary_account(tmp_path):
    """Ordinary account with single div0 yields one Case."""
    xml_dir = tmp_path / "ordinary"
    xml_dir.mkdir()
    (xml_dir / "OA16760517.xml").write_text(ORDINARY_SAMPLE, encoding="utf-8")

    cases = list(iter_cases_from_xml_dir(xml_dir))
    assert len(cases) == 1
    assert cases[0].case_id == "OA16760517"
    assert cases[0].date_iso == "1676-05-17"
    assert cases[0].year == 1676
    assert cases[0].offence_category == "violentTheft"


def test_iter_cases_unknown_schema(tmp_path):
    """Unknown schema falls back to file stem as case_id."""
    xml_dir = tmp_path / "unknown"
    xml_dir.mkdir()
    (xml_dir / "mystery.xml").write_text(UNKNOWN_SAMPLE, encoding="utf-8")

    cases = list(iter_cases_from_xml_dir(xml_dir))
    assert len(cases) == 1
    assert cases[0].case_id == "mystery"
    assert cases[0].date_iso == "1690-01-15"
    assert cases[0].year == 1690


def test_iter_cases_empty_dir(tmp_path):
    """Empty directory yields no cases."""
    cases = list(iter_cases_from_xml_dir(tmp_path))
    assert cases == []


def test_iter_cases_not_directory(tmp_path):
    """Raises NotADirectoryError when root is not a directory."""
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        list(iter_cases_from_xml_dir(f))


def test_iter_cases_recursive(tmp_path):
    """Recurses into subdirectories."""
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    sample = ORDINARY_SAMPLE.replace("OA16760517", "OA99999999")
    (sub / "deep.xml").write_text(sample, encoding="utf-8")

    cases = list(iter_cases_from_xml_dir(tmp_path))
    assert len(cases) == 1
    assert cases[0].case_id == "OA99999999"


# Speech extraction fixtures
SAMPLE_WITH_SPEAKER = """<?xml version="1.0" encoding="UTF-8"?>
<TEI.2>
  <text><body>
    <div1 type="trialAccount" id="t18440000-1">
      <interp inst="t18440000-1" type="date" value="18440101"/>
      <p><hi rend="italic">Prisoner</hi>. He asked me if I was the lad; I said I was not.</p>
      <p><hi rend="italic">Witness</hi>. He did not say so.</p>
      <p><hi rend="smallCaps">COURT</hi>. Consider your answer.</p>
    </div1>
  </body></text>
</TEI.2>"""

SAMPLE_WITH_LB_NOTE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI.2>
  <text><body>
    <div1 type="trialAccount" id="t17290000-1">
      <interp inst="t17290000-1" type="date" value="17291203"/>
      <p>Ste<lb/>phen Hunter was found guilty. <note>[Transportation. See summary.]</note></p>
    </div1>
  </body></text>
</TEI.2>"""


def test_speech_extraction_with_speaker(tmp_path):
    """Paragraphs with hi speaker labels extract speaker_name."""
    xml_dir = tmp_path / "speaker"
    xml_dir.mkdir()
    (xml_dir / "speaker.xml").write_text(SAMPLE_WITH_SPEAKER, encoding="utf-8")

    items = list(iter_cases_and_speeches_from_xml_dir(xml_dir))
    assert len(items) == 1
    case, speeches = items[0]
    assert case.case_id == "t18440000-1"
    assert len(speeches) == 3
    assert speeches[0].speaker_name == "Prisoner"
    assert "He asked me" in speeches[0].text
    assert speeches[1].speaker_name == "Witness"
    assert speeches[2].speaker_name == "Court"  # normalized from COURT


def test_speech_extraction_lb_note_cleaned(tmp_path):
    """lb becomes space, note content stripped, whitespace normalized."""
    xml_dir = tmp_path / "clean"
    xml_dir.mkdir()
    (xml_dir / "clean.xml").write_text(SAMPLE_WITH_LB_NOTE, encoding="utf-8")

    items = list(iter_cases_and_speeches_from_xml_dir(xml_dir))
    assert len(items) == 1
    case, speeches = items[0]
    assert len(speeches) == 1
    # lb: "Ste<lb/>phen" -> "Stephen" (lb joins without space)
    assert "Stephen" in speeches[0].text
    # note stripped
    assert "[Transportation" not in speeches[0].text
    assert "See summary" not in speeches[0].text


SAMPLE_WITH_SUBTITLES = """<?xml version="1.0" encoding="UTF-8"?>
<TEI.2>
  <text><body>
    <div1 type="trialAccount" id="t16740000-1">
      <interp inst="t16740000-1" type="date" value="16740429"/>
      <interp inst="t16740000-1" type="offenceCategory" value="theft"/>
      <interp inst="t16740000-1" type="offenceSubcategory" value="burglary"/>
      <p>There were three
        <persName type="defendantName">others</persName>
        tryed for
        <rs type="offenceDescription">breaking the House of one
          <persName type="victimName"><rs type="occupation">Mr.</rs> Bradbourn</persName>
          , in the Parish of
          <placeName>St. Giles in the Fields</placeName>
          , having stole some Silver Spoons</rs>
        , the Jury found them
        <rs type="verdictDescription">Guilty</rs>
        , whereupon
        <rs type="punishmentDescription">Sentence was past</rs>.</p>
    </div1>
  </body></text>
</TEI.2>"""


def test_subtitles_in_meta_json(tmp_path):
    """Subtitles (defendants, victims, place, offence, verdict, punishment) appear in meta_json."""
    xml_dir = tmp_path / "subtitles"
    xml_dir.mkdir()
    (xml_dir / "sub.xml").write_text(SAMPLE_WITH_SUBTITLES, encoding="utf-8")

    items = list(iter_cases_and_speeches_from_xml_dir(xml_dir))
    assert len(items) == 1
    case, _ = items[0]
    meta = json.loads(case.meta_json)
    assert "subtitles" in meta
    st = meta["subtitles"]
    assert st.get("defendants") == "others"
    assert "Bradbourn" in st.get("victims", "")
    assert "St. Giles" in st.get("place", "")
    assert "breaking" in st.get("offence_description", "")
    assert st.get("verdict") == "Guilty"
    assert "Sentence" in st.get("punishment", "")
