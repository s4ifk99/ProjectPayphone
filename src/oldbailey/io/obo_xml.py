"""
Parse Old Bailey XML files into Case records.

Uses lxml if available, else xml.etree.ElementTree.
Streaming/memory-safe: one file at a time, no full corpus in memory.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree as ET

from oldbailey.model.schema import Case, Speech

logger = logging.getLogger(__name__)


def _local_tag(elem) -> str:
    """Return tag name without namespace (e.g. '{uri}lb' -> 'lb')."""
    tag = getattr(elem, "tag", None) or ""
    if isinstance(tag, str) and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _element_text(elem) -> str:
    """
    Recursively collect all text and tail from element and descendants.
    Cleaned: lb joins without space (Ste<lb/>phen -> Stephen), note/xptr stripped,
    whitespace normalized.
    """
    parts: list[str] = [elem.text or ""]
    for child in elem:
        tag = _local_tag(child)
        if tag == "lb":
            # Line break: "Ste<lb/>phen" -> "Stephen"; append tail to last part
            if parts:
                parts[-1] = (parts[-1] or "") + (child.tail or "")
        elif tag == "note":
            parts.append(child.tail or "")
        elif tag == "xptr":
            parts.append(child.tail or "")
        else:
            parts.append(_element_text(child))
            parts.append(child.tail or "")
    raw = " ".join(p for p in parts if p)
    return re.sub(r"\s+", " ", raw).strip()


def _extract_speaker_from_p(p_elem) -> str | None:
    """
    Extract speaker label from paragraph start if present.
    OBO uses <hi rend="italic">Prisoner</hi>. or <hi rend="smallCaps">COURT</hi>. at paragraph lead.
    Only hi elements; persName at start is often defendant in narrative, not a speaker.
    """
    for child in p_elem:
        tag = _local_tag(child)
        rend = child.get("rend", "") if hasattr(child, "get") else ""
        if tag == "hi" and rend in ("italic", "smallCaps"):
            label = _element_text(child).strip()
            if label:
                return "Court" if label.upper() == "COURT" else label
    return None


def _speeches_from_div(div_elem, case_id: str) -> list[Speech]:
    """Extract paragraph text from a trial div (div1 or div0) as Speech records."""
    speeches: list[Speech] = []
    for i, p in enumerate(div_elem.iter("p"), start=1):
        text = _element_text(p)
        if not text:
            continue
        speaker_name = _extract_speaker_from_p(p)
        speeches.append(
            Speech(
                case_id=case_id,
                speech_no=i,
                speaker_id=None,
                speaker_name=speaker_name,
                text=text,
                source="OBO_XML",
                metadata={},
            )
        )
    return speeches

try:
    import lxml.etree as LET

    _HAS_LXML = True
except ImportError:
    _HAS_LXML = False


def _parse_date_to_iso(value: str | None) -> str | None:
    """Best-effort parse to ISO date string (YYYY-MM-DD)."""
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    # YYYYMMDD (8 digits) - common in Old Bailey
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", value)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo}-{d}"
    # YYYY-MM-DD already
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value
    # YYYYMM (6 digits) - use first day
    m = re.match(r"^(\d{4})(\d{2})$", value)
    if m:
        return f"{m.group(1)}-{m.group(2)}-01"
    # YYYY only
    m = re.match(r"^(\d{4})$", value)
    if m:
        return f"{m.group(1)}-01-01"
    return value  # return raw as best-effort


def _year_from_date_iso(date_iso: str | None) -> int | None:
    """Extract year from ISO date string."""
    if not date_iso or len(date_iso) < 4:
        return None
    try:
        return int(date_iso[:4])
    except ValueError:
        return None


def _gather_interps(parent) -> dict[str, list[str]]:
    """Collect interp elements: {type: [value, ...]}."""
    out: dict[str, list[str]] = {}
    for interp in parent.iter("interp"):
        if interp is parent:
            continue
        typ = interp.get("type")
        val = interp.get("value")
        if typ and val is not None:
            out.setdefault(typ, []).append(val)
    return out


def _extract_court(interps: dict, root) -> str | None:
    """Best-effort court from interps or root."""
    if "court" in interps and interps["court"]:
        return interps["court"][0]
    # Some schemas use different keys
    for k in ("courtName", "place", "location"):
        if k in interps and interps[k]:
            return interps[k][0]
    return None


def _extract_case_subtitles(div_elem) -> dict:
    """
    Extract structured entities from trial div for subtitle display.

    Collects: defendants, victims, offence_description, place, verdict, punishment.
    """
    out: dict[str, str | list[str]] = {}
    defendants: list[str] = []
    victims: list[str] = []
    offence_desc: str | None = None
    place: str | None = None
    verdict: str | None = None
    punishment: str | None = None

    for elem in div_elem.iter():
        tag = _local_tag(elem)
        ptype = elem.get("type", "") if hasattr(elem, "get") else ""

        if tag == "persName":
            text = _element_text(elem).strip()
            if not text:
                continue
            if ptype == "defendantName":
                defendants.append(text)
            elif ptype == "victimName":
                victims.append(text)

        elif tag == "rs":
            text = _element_text(elem).strip()
            if not text:
                continue
            if ptype == "offenceDescription" and offence_desc is None:
                offence_desc = text
            elif ptype == "verdictDescription" and verdict is None:
                verdict = text
            elif ptype == "punishmentDescription" and punishment is None:
                punishment = text

        elif tag == "placeName":
            text = _element_text(elem).strip()
            if text and place is None:
                place = text

    subtitles: dict[str, str] = {}
    if defendants:
        subtitles["defendants"] = "; ".join(defendants)
    if victims:
        subtitles["victims"] = "; ".join(victims)
    if offence_desc:
        subtitles["offence_description"] = offence_desc
    if place:
        subtitles["place"] = place
    if verdict:
        subtitles["verdict"] = verdict
    if punishment:
        subtitles["punishment"] = punishment

    return subtitles


def _build_meta_json(interps: dict, elem_attrib: dict, extra: dict | None = None) -> str:
    """Build meta_json from interps, attributes, extra, and optional subtitles."""
    meta: dict = {}
    if interps:
        meta["interp"] = {k: v[0] if len(v) == 1 else v for k, v in interps.items()}
    if elem_attrib:
        meta["attributes"] = dict(elem_attrib)
    if extra:
        meta.update(extra)
    return json.dumps(meta, ensure_ascii=False)


def _pick_offence_category(interps_local: dict, interps_global: dict) -> str | None:
    """
    Choose a single main offenceCategory value for a case.

    Rule: prefer the first offenceCategory in the local interps (trial/div),
    otherwise fall back to the first offenceCategory in the global interps.
    """
    if interps_local.get("offenceCategory"):
        return interps_local["offenceCategory"][0]
    if interps_global.get("offenceCategory"):
        return interps_global["offenceCategory"][0]
    return None


def _iter_cases_etree(
    root_elem, xml_path: str, file_stem: str, with_speeches: bool = False
) -> Iterator[Case] | Iterator[tuple[Case, list[Speech]]]:
    """Yield Case (and optionally speeches) from ElementTree root. Handles trialAccount div1s and single div0."""
    interps_global = _gather_interps(root_elem)
    date_iso = None
    year = None
    for interp in root_elem.iter("interp"):
        if interp.get("type") == "date" and interp.get("value"):
            date_iso = _parse_date_to_iso(interp.get("value"))
            year = _year_from_date_iso(date_iso)
            break
    if date_iso is None and interps_global.get("date"):
        date_iso = _parse_date_to_iso(interps_global["date"][0])
        year = _year_from_date_iso(date_iso)
    if year is None and interps_global.get("year"):
        try:
            year = int(interps_global["year"][0])
        except (ValueError, IndexError):
            pass
    court = _extract_court(interps_global, root_elem)

    # Prefer div1 type="trialAccount" (sessions papers) - each is a case
    yielded = False
    for div1 in root_elem.iter("div1"):
        if div1.get("type") == "trialAccount":
            cid = div1.get("id")
            if not cid:
                continue
            local_interps = _gather_interps(div1)
            d = date_iso
            y = year
            if local_interps.get("date"):
                d = _parse_date_to_iso(local_interps["date"][0])
                y = _year_from_date_iso(d)
            if y is None and local_interps.get("year"):
                try:
                    y = int(local_interps["year"][0])
                except (ValueError, IndexError):
                    pass
            c = _extract_court(local_interps, div1) or court
            offence_category = _pick_offence_category(local_interps, interps_global)
            subtitles = _extract_case_subtitles(div1)
            meta = _build_meta_json(
                local_interps, dict(div1.attrib), extra={"subtitles": subtitles} if subtitles else None
            )
            case = Case(
                case_id=cid,
                date_iso=d,
                year=y,
                court=c,
                offence_category=offence_category,
                xml_path=xml_path,
                meta_json=meta,
                source="obo-xml",
            )
            if with_speeches:
                yield (case, _speeches_from_div(div1, cid))
            else:
                yield case
            yielded = True

    if yielded:
        return

    # Fallback: div0 with id (e.g. ordinarysAccounts, or single-case sessions)
    for div0 in root_elem.iter("div0"):
        cid = div0.get("id")
        if cid:
            local_interps = _gather_interps(div0)
            d = date_iso
            y = year
            if local_interps.get("date"):
                d = _parse_date_to_iso(local_interps["date"][0])
                y = _year_from_date_iso(d)
            if y is None and local_interps.get("year"):
                try:
                    y = int(local_interps["year"][0])
                except (ValueError, IndexError):
                    pass
            c = _extract_court(local_interps, div0) or court
            offence_category = _pick_offence_category(local_interps, interps_global)
            subtitles = _extract_case_subtitles(div0)
            meta = _build_meta_json(
                local_interps, dict(div0.attrib), extra={"subtitles": subtitles} if subtitles else None
            )
            case = Case(
                case_id=cid,
                date_iso=d,
                year=y,
                court=c,
                offence_category=offence_category,
                xml_path=xml_path,
                meta_json=meta,
                source="obo-xml",
            )
            if with_speeches:
                yield (case, _speeches_from_div(div0, cid))
            else:
                yield case
            return

    # No div0/div1 with id: use file stem as case_id
    meta = _build_meta_json(interps_global, {})
    offence_category = _pick_offence_category({}, interps_global)
    case = Case(
        case_id=file_stem,
        date_iso=date_iso,
        year=year,
        court=court,
        offence_category=offence_category,
        xml_path=xml_path,
        meta_json=meta,
        source="obo-xml",
    )
    if with_speeches:
        yield (case, [])
    else:
        yield case


def _parse_file(
    file_path: Path, with_speeches: bool = False
) -> Iterator[Case] | Iterator[tuple[Case, list[Speech]]]:
    """Parse a single XML file and yield Case records (and optionally speeches). Memory-safe: one file at a time."""
    if _HAS_LXML:
        try:
            tree = LET.parse(str(file_path))
            root = tree.getroot()
            yield from _iter_cases_lxml(root, str(file_path), file_path.stem, with_speeches)
            return
        except Exception as e:
            logger.warning("lxml parse failed for %s: %s; falling back to ElementTree", file_path, e)

    tree = ET.parse(str(file_path))
    root = tree.getroot()
    yield from _iter_cases_etree(root, str(file_path), file_path.stem, with_speeches)


def _iter_cases_lxml(
    root, xml_path: str, file_stem: str, with_speeches: bool = False
) -> Iterator[Case] | Iterator[tuple[Case, list[Speech]]]:
    """Yield Case (and optionally speeches) from lxml root. Same logic as _iter_cases_etree."""
    interps_global = _gather_interps(root)
    date_iso = None
    year = None
    for interp in root.iter("interp"):
        if interp.get("type") == "date" and interp.get("value"):
            date_iso = _parse_date_to_iso(interp.get("value"))
            year = _year_from_date_iso(date_iso)
            break
    if date_iso is None and interps_global.get("date"):
        date_iso = _parse_date_to_iso(interps_global["date"][0])
        year = _year_from_date_iso(date_iso)
    if year is None and interps_global.get("year"):
        try:
            year = int(interps_global["year"][0])
        except (ValueError, IndexError):
            pass
    court = _extract_court(interps_global, root)

    yielded = False
    for div1 in root.iter("div1"):
        if div1.get("type") == "trialAccount":
            cid = div1.get("id")
            if not cid:
                continue
            local_interps = _gather_interps(div1)
            d = date_iso
            y = year
            if local_interps.get("date"):
                d = _parse_date_to_iso(local_interps["date"][0])
                y = _year_from_date_iso(d)
            if y is None and local_interps.get("year"):
                try:
                    y = int(local_interps["year"][0])
                except (ValueError, IndexError):
                    pass
            c = _extract_court(local_interps, div1) or court
            offence_category = _pick_offence_category(local_interps, interps_global)
            subtitles = _extract_case_subtitles(div1)
            meta = _build_meta_json(
                local_interps, dict(div1.attrib), extra={"subtitles": subtitles} if subtitles else None
            )
            case = Case(
                case_id=cid,
                date_iso=d,
                year=y,
                court=c,
                offence_category=offence_category,
                xml_path=xml_path,
                meta_json=meta,
                source="obo-xml",
            )
            if with_speeches:
                yield (case, _speeches_from_div(div1, cid))
            else:
                yield case
            yielded = True

    if yielded:
        return

    for div0 in root.iter("div0"):
        cid = div0.get("id")
        if cid:
            local_interps = _gather_interps(div0)
            d = date_iso
            y = year
            if local_interps.get("date"):
                d = _parse_date_to_iso(local_interps["date"][0])
                y = _year_from_date_iso(d)
            if y is None and local_interps.get("year"):
                try:
                    y = int(local_interps["year"][0])
                except (ValueError, IndexError):
                    pass
            c = _extract_court(local_interps, div0) or court
            offence_category = _pick_offence_category(local_interps, interps_global)
            subtitles = _extract_case_subtitles(div0)
            meta = _build_meta_json(
                local_interps, dict(div0.attrib), extra={"subtitles": subtitles} if subtitles else None
            )
            case = Case(
                case_id=cid,
                date_iso=d,
                year=y,
                court=c,
                offence_category=offence_category,
                xml_path=xml_path,
                meta_json=meta,
                source="obo-xml",
            )
            if with_speeches:
                yield (case, _speeches_from_div(div0, cid))
            else:
                yield case
            return

    meta = _build_meta_json(interps_global, {})
    offence_category = _pick_offence_category({}, interps_global)
    case = Case(
        case_id=file_stem,
        date_iso=date_iso,
        year=year,
        court=court,
        offence_category=offence_category,
        xml_path=xml_path,
        meta_json=meta,
        source="obo-xml",
    )
    if with_speeches:
        yield (case, [])
    else:
        yield case


def iter_cases_from_xml_dir(root: Path) -> Iterator[Case]:
    """
    Yield Case records from a directory tree of Old Bailey XML files.

    Walks root recursively, processes each .xml file once. Memory-safe:
    one file at a time, no full corpus in memory.

    - case_id: from XML id attribute if present (div1 trialAccount or div0),
      else file stem
    - date_iso: best-effort parse from interp type="date" to YYYY-MM-DD
    - year: extracted from date if available
    - court: from interp if present, else null
    - xml_path: path to the file (as passed)
    - meta_json: JSON string of non-core metadata (interps, attributes)
    """
    root = Path(root.resolve())
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    def _skip_macos_metadata(p: Path) -> bool:
        # Skip __MACOSX and macOS resource-fork files (._*)
        parts = p.parts
        if "__MACOSX" in parts:
            return True
        if p.name.startswith("._"):
            return True
        return False

    xml_files = sorted(
        fp for fp in root.rglob("*.xml") if not _skip_macos_metadata(fp)
    )
    logger.info("Found %d XML files under %s", len(xml_files), root)

    for i, fp in enumerate(xml_files):
        try:
            for case in _parse_file(fp):
                yield case
        except Exception as e:
            logger.exception("Failed to parse %s: %s", fp, e)
            continue


def iter_cases_and_speeches_from_xml_dir(
    root: Path,
) -> Iterator[tuple[Case, list[Speech]]]:
    """
    Yield (Case, list[Speech]) from a directory tree of Old Bailey XML files.

    Same as iter_cases_from_xml_dir but also extracts paragraph text from each
    trial div as Speech records (source="OBO_XML"). One parse per file.
    """
    root = Path(root.resolve())
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    def _skip_macos_metadata(p: Path) -> bool:
        if "__MACOSX" in p.parts or p.name.startswith("._"):
            return True
        return False

    xml_files = sorted(
        fp for fp in root.rglob("*.xml") if not _skip_macos_metadata(fp)
    )
    logger.info("Found %d XML files under %s", len(xml_files), root)

    for fp in xml_files:
        try:
            for item in _parse_file(fp, with_speeches=True):
                yield item
        except Exception as e:
            logger.exception("Failed to parse %s: %s", fp, e)
            continue
