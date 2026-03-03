#!/usr/bin/env python3
"""
TEI XML ingestion pipeline for Old Bailey sessions papers.

Recursively scans ./xml_files for TEI XML, extracts document and case data
from div0/div1, inserts into old_bailey.db, and writes case_cards.jsonl.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from lxml import etree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

XML_DIR = Path("./xml_files")
DB_PATH = Path("./old_bailey.db")
JSONL_PATH = Path("./case_cards.jsonl")


def _local_tag(elem) -> str:
    """Return tag name without namespace."""
    tag = getattr(elem, "tag", None) or ""
    if isinstance(tag, str) and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _element_text(elem) -> str:
    """Recursively collect text; lb joins without space, note/xptr stripped, whitespace normalized."""
    parts: list[str] = [elem.text or ""]
    for child in elem:
        tag = _local_tag(child)
        if tag == "lb":
            if parts:
                parts[-1] = (parts[-1] or "") + (child.tail or "")
        elif tag in ("note", "xptr"):
            parts.append(child.tail or "")
        else:
            parts.append(_element_text(child))
            parts.append(child.tail or "")
    raw = " ".join(p for p in parts if p)
    return re.sub(r"\s+", " ", raw).strip()


def _gather_interps_by_inst(scope_elem) -> dict[str, dict[str, str]]:
    """Build {elem_id: {interp_type: value}} from interp elements with inst attr."""
    result: dict[str, dict[str, str]] = {}
    for interp in scope_elem.iter():
        if _local_tag(interp) != "interp":
            continue
        inst = interp.get("inst")
        typ = interp.get("type")
        val = interp.get("value")
        if inst and typ and val is not None:
            if inst not in result:
                result[inst] = {}
            result[inst][typ] = val
    return result


def _iter_elements(scope, tag: str, attrs: dict[str, str] | None = None):
    """Yield elements with local tag and optional attribute filters."""
    for elem in scope.iter():
        if _local_tag(elem) != tag:
            continue
        if attrs:
            match = all(elem.get(k) == v for k, v in attrs.items())
            if not match:
                continue
        yield elem


def _extract_document(div0, source_file: str) -> dict[str, Any] | None:
    """Extract document metadata from div0 type='sessionsPaper'."""
    doc_id = div0.get("id")
    if not doc_id:
        return None
    interps = _gather_interps_by_inst(div0)
    doc_interps = interps.get(doc_id, {})
    return {
        "doc_id": doc_id,
        "collection": doc_interps.get("collection"),
        "year": _parse_year(doc_interps.get("year") or doc_interps.get("date")),
        "uri": doc_interps.get("uri"),
        "date_raw": doc_interps.get("date"),
        "source_file": source_file,
    }


def _parse_year(val: str | None) -> int | None:
    """Extract year from value (e.g. '1674' or '16740429')."""
    if not val:
        return None
    m = re.match(r"^(\d{4})", str(val).strip())
    return int(m.group(1)) if m else None


def _extract_full_text(div1) -> str:
    """Concatenate cleaned text from all <p> inside div1."""
    parts: list[str] = []
    for p in _iter_elements(div1, "p"):
        t = _element_text(p)
        if t:
            parts.append(t)
    return " ".join(parts).strip() if parts else ""


def _extract_page_facsimiles(div1) -> list[str]:
    """Collect xptr[@type='pageFacsimile'] @doc values."""
    out: list[str] = []
    for xptr in _iter_elements(div1, "xptr", {"type": "pageFacsimile"}):
        doc_val = xptr.get("doc")
        if doc_val:
            out.append(doc_val)
    return out


def _extract_offences(div1, interps_by_inst: dict) -> list[dict]:
    """Extract offences from rs type='offenceDescription'."""
    offences: list[dict] = []
    for rs in _iter_elements(div1, "rs", {"type": "offenceDescription"}):
        txt = _element_text(rs).strip()
        rid = rs.get("id") or ""
        ip = interps_by_inst.get(rid, {})
        offences.append({
            "offence_text": txt or None,
            "offenceCategory": ip.get("offenceCategory"),
            "offenceSubcategory": ip.get("offenceSubcategory"),
        })
    return offences


def _extract_defendants(div1, interps_by_inst: dict) -> list[dict]:
    """Extract defendants from persName type='defendantName'."""
    out: list[dict] = []
    for pn in _iter_elements(div1, "persName", {"type": "defendantName"}):
        display_name = _element_text(pn).strip()
        pid = pn.get("id") or ""
        ip = interps_by_inst.get(pid, {})
        out.append({
            "display_name": display_name or None,
            "gender": ip.get("gender"),
        })
    return out


def _extract_victims(div1, interps_by_inst: dict) -> list[dict]:
    """Extract victims from persName type='victimName'; include occupation from child rs."""
    out: list[dict] = []
    for pn in _iter_elements(div1, "persName", {"type": "victimName"}):
        display_name = _element_text(pn).strip()
        pid = pn.get("id") or ""
        ip = interps_by_inst.get(pid, {})
        occupations: list[str] = []
        for rs in pn.iter():
            if _local_tag(rs) == "rs" and rs.get("type") == "occupation":
                t = _element_text(rs).strip()
                if t:
                    occupations.append(t)
        out.append({
            "display_name": display_name or None,
            "gender": ip.get("gender"),
            "occupation_labels": occupations,
        })
    return out


def _extract_verdicts(div1, interps_by_inst: dict) -> list[dict]:
    """Extract verdicts from rs type='verdictDescription'."""
    out: list[dict] = []
    for rs in _iter_elements(div1, "rs", {"type": "verdictDescription"}):
        txt = _element_text(rs).strip()
        rid = rs.get("id") or ""
        ip = interps_by_inst.get(rid, {})
        out.append({
            "verdict_text": txt or None,
            "verdictCategory": ip.get("verdictCategory"),
        })
    return out


def _extract_punishments(div1, interps_by_inst: dict) -> list[dict]:
    """Extract punishments from rs type='punishmentDescription'."""
    out: list[dict] = []
    for rs in _iter_elements(div1, "rs", {"type": "punishmentDescription"}):
        txt = _element_text(rs).strip()
        rid = rs.get("id") or ""
        ip = interps_by_inst.get(rid, {})
        out.append({
            "punishment_text": txt or None,
            "punishmentCategory": ip.get("punishmentCategory"),
        })
    return out


def _extract_places(div1, interps_by_inst: dict) -> list[dict]:
    """Extract places from placeName elements."""
    out: list[dict] = []
    for pn in _iter_elements(div1, "placeName"):
        place_text = _element_text(pn).strip()
        pid = pn.get("id") or ""
        ip = interps_by_inst.get(pid, {})
        place_type = ip.get("type") or ip.get("placeName")
        out.append({
            "place_text": place_text or None,
            "place_type": place_type,
        })
    return out


def _extract_case(div1, doc_id: str, year: int | None, seq: int) -> dict[str, Any] | None:
    """Extract full case card from div1 type='trialAccount'."""
    case_id = div1.get("id")
    if not case_id:
        return None
    interps = _gather_interps_by_inst(div1)
    card = {
        "case_id": case_id,
        "doc_id": doc_id,
        "year": year,
        "offences": _extract_offences(div1, interps),
        "defendants": _extract_defendants(div1, interps),
        "victims": _extract_victims(div1, interps),
        "verdicts": _extract_verdicts(div1, interps),
        "punishments": _extract_punishments(div1, interps),
        "places": _extract_places(div1, interps),
        "full_text": _extract_full_text(div1),
        "page_facsimiles": _extract_page_facsimiles(div1),
    }
    return card


def _iter_elements_by_tag(scope, tag: str):
    """Yield elements with local tag (no attr filter)."""
    for elem in scope.iter():
        if _local_tag(elem) == tag:
            yield elem


def _iter_div0_sessions(root):
    """Yield div0 elements with type='sessionsPaper'."""
    for div0 in _iter_elements_by_tag(root, "div0"):
        if div0.get("type") == "sessionsPaper":
            yield div0


def _iter_div1_trial_accounts(div0):
    """Yield div1 elements with type='trialAccount' in document order."""
    for div1 in div0.iter():
        if _local_tag(div1) == "div1" and div1.get("type") == "trialAccount":
            yield div1


def _skip_path(p: Path) -> bool:
    """Skip macOS metadata and hidden files."""
    parts = p.parts
    if "__MACOSX" in parts:
        return True
    if any(part.startswith("._") for part in parts):
        return True
    return False


def _iter_xml_files() -> list[Path]:
    """Recursively find XML files in xml_files, excluding skip paths."""
    if not XML_DIR.exists():
        return []
    out: list[Path] = []
    for p in sorted(XML_DIR.rglob("*.xml")):
        if _skip_path(p):
            continue
        out.append(p)
    return out


def _init_db(conn: sqlite3.Connection) -> None:
    """Create tables (idempotent: drop if exist)."""
    conn.execute("DROP TABLE IF EXISTS cases")
    conn.execute("DROP TABLE IF EXISTS documents")
    conn.execute("""
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            collection TEXT,
            year INTEGER,
            uri TEXT,
            date_raw TEXT,
            source_file TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE cases (
            case_id TEXT PRIMARY KEY,
            doc_id TEXT,
            sequence_in_doc INTEGER,
            full_text TEXT,
            page_facsimiles TEXT,
            card_json TEXT,
            FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
        )
    """)


def main() -> None:
    xml_files = _iter_xml_files()
    if not xml_files:
        logger.warning("No XML files found in %s", XML_DIR.resolve())
        return

    conn = sqlite3.connect(DB_PATH)
    _init_db(conn)
    documents: list[tuple] = []
    cases: list[tuple] = []
    cards: list[dict] = []

    for xml_path in xml_files:
        try:
            tree = etree.parse(
                str(xml_path),
                parser=etree.XMLParser(remove_blank_text=True, recover=True),
            )
            root = tree.getroot()
        except Exception as e:
            logger.warning("Parse error %s: %s", xml_path, e)
            continue

        rel_path = str(xml_path.relative_to(XML_DIR)) if xml_path.is_relative_to(XML_DIR) else str(xml_path)

        for div0 in _iter_div0_sessions(root):
            doc = _extract_document(div0, rel_path)
            if not doc:
                continue
            documents.append((
                doc["doc_id"],
                doc.get("collection"),
                doc.get("year"),
                doc.get("uri"),
                doc.get("date_raw"),
                doc["source_file"],
            ))
            doc_id = doc["doc_id"]
            year = doc.get("year")

            for seq, div1 in enumerate(_iter_div1_trial_accounts(div0), start=1):
                card = _extract_case(div1, doc_id, year, seq)
                if not card:
                    continue
                cards.append(card)
                cases.append((
                    card["case_id"],
                    doc_id,
                    seq,
                    card["full_text"],
                    json.dumps(card["page_facsimiles"]),
                    json.dumps(card, ensure_ascii=False),
                ))

    for row in documents:
        conn.execute(
            "INSERT INTO documents (doc_id, collection, year, uri, date_raw, source_file) VALUES (?, ?, ?, ?, ?, ?)",
            row,
        )
    for row in cases:
        conn.execute(
            "INSERT INTO cases (case_id, doc_id, sequence_in_doc, full_text, page_facsimiles, card_json) VALUES (?, ?, ?, ?, ?, ?)",
            row,
        )
    conn.commit()

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")

    conn.close()
    logger.info("Ingested %d documents, %d cases -> %s, %s", len(documents), len(cases), DB_PATH, JSONL_PATH)


if __name__ == "__main__":
    main()
