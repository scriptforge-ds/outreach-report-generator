"""
Regex-first metadata extraction with optional LLM backfill (via BaseLLM).

Design:
- First pass: deterministic regex extraction (fast, stable).
- Second pass (optional): LLM fills ONLY missing keys; never overwrites regex hits.
- Final pass: strong post-processing to remove markdown/noise and to split
  combined clauses like:
    "under **Gardi Pharid Panchayat** in **Shri Chamkaur Sahib Block**"

This module is tuned for translated English transcripts produced by Ajsal's pipeline
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pipeline.extraction.base_llm import BaseLLM

log = logging.getLogger(__name__)

# =============================================================================
# SCHEMA
# =============================================================================

SCHEMA: Dict[str, Any] = {
    "date": None,
    "day": None,
    "village": None,
    "sarpanch_name": None,
    "panchayat": None,
    "block": None,
    "phone_number": None,
    "event_location": None,
    "district": None,
    "farmers_attended_total": None,
    "coordinator_name": None,
    "reporting_manager_name": None,
    "female_farmers_count": None,
    "male_farmers_count": None,
    "event_start_time": None,
    "event_end_time": None,
}

NUMWORDS = {
    "zero": 0, "nil": 0, "none": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

HONORIFICS_RE = re.compile(r"^\s*(shri|smt|mr|mrs|ms|miss|dr)\.?\s+", re.I)


# =============================================================================
# TEXT HELPERS
# =============================================================================

def normalize_text(t: str) -> str:
    t = (t or "")
    t = t.replace("\u2019", "'").replace("\u2018", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\\ u", "\\u")               # fix broken escapes
    t = re.sub(r"\\\s*u0A3C", "", t)           # remove marker seen in your data
    t = re.sub(r"\s+", " ", t).strip()
    return t

def weekday_from_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    s = str(date_str).strip()
    # expected "YYYY-MM-DD"
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%A")
    except Exception:
        return None

def pick_relevant_window(text: str, window: int = 7000) -> str:
    """
    Metadata typically appears early in your narration.
    We pick a window around the first occurrence of any key phrase.
    """
    t = (text or "")
    tn = t.lower()
    keys = [
        "today", "date", "day",
        "village", "panchayat", "block", "district",
        "coordinator", "reporting manager", "sarpanch",
        "phone",
        "event start", "event end", "meeting location", "event location",
        "farmers", "male farmers", "female farmers", "total farmers",
    ]
    idxs = [tn.find(k) for k in keys if tn.find(k) != -1]
    if not idxs:
        return t[:window]
    i = max(min(idxs) - window // 2, 0)
    return t[i:i + window]


def first_match(patterns: List[str], text: str, flags=re.I) -> Optional[str]:
    for p in patterns:
        m = re.search(p, text, flags)
        if m:
            return (m.group(1) or "").strip()
    return None


def clean_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    s = re.sub(r"[,\.;:\-]+$", "", s).strip()
    return s if s else None


def to_int_maybe(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s.isdigit():
        return int(s)
    return NUMWORDS.get(s)


# =============================================================================
# SANITIZERS (markdown/noise, people, places, time)
# =============================================================================

def strip_markdown(s: str) -> str:
    s = (s or "")
    s = re.sub(r"\*\*", "", s)   # **bold**
    s = re.sub(r"__", "", s)     # __underline__
    s = s.replace("`", "")       # `code`
    return s


def clean_place(s: Any) -> Optional[str]:
    if s is None:
        return None
    s = strip_markdown(str(s))
    s = normalize_text(s)
    s = s.strip(" ,.;:-")
    if not s or s in ("*", "**"):
        return None
    # remove leading glue words
    s = re.sub(r"^(under|in|at)\s+", "", s, flags=re.I).strip()
    return s or None


def clean_person_name(s: Any, drop_honorifics: bool = True) -> Optional[str]:
    if s is None:
        return None
    s = strip_markdown(str(s))
    s = normalize_text(s).strip(" ,.;:-")
    if not s:
        return None
    if drop_honorifics:
        s = HONORIFICS_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def normalize_phone(s: Any) -> Optional[str]:
    if s is None:
        return None
    digits = re.sub(r"\D", "", str(s))
    if len(digits) >= 10:
        return digits[-10:]
    return None


def normalize_time(t: Any) -> Optional[str]:
    """
    Convert formats like:
      "11AM", "11:00AM", "11:00 AM", "1PM", "01:00 pm"
    to:
      "11:00 AM", "01:00 PM"
    """
    if t is None:
        return None
    s = strip_markdown(str(t))
    s = normalize_text(s).upper().replace(" ", "")
    if not s:
        return None

    m = re.match(r"^(\d{1,2})(?::?(\d{2}))?(AM|PM)$", s)
    if not m:
        return clean_value(t)

    hh = int(m.group(1))
    mm = int(m.group(2) or "00")
    ap = m.group(3)
    return f"{hh:02d}:{mm:02d} {ap}"


def _split_combined_places(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix combined clauses often produced by LLM or run-on transcript:
      village: "under Gardi Pharid Panchayat in Shri Chamkaur Sahib Block"
    """
    m = dict(meta)

    village = m.get("village")
    panchayat = m.get("panchayat")
    block = m.get("block")

    # If village contains "panchayat" or "block", split it.
    if isinstance(village, str):
        v = clean_place(village) or ""
        v2 = re.sub(r"^\s*under\s+", "", v, flags=re.I).strip()
        vl = v2.lower()

        # Extract block name "... <BLOCK> Block"
        mb = re.search(r"([A-Za-z][A-Za-z\s]+?)\s+block\b", v2, flags=re.I)
        if mb and (not block or block in ("*", "**")):
            m["block"] = mb.group(1).strip()

        # Extract panchayat name "... <PANCHAYAT> Panchayat"
        mp = re.search(r"([A-Za-z][A-Za-z\s]+?)\s+panchayat\b", v2, flags=re.I)
        if mp and not panchayat:
            m["panchayat"] = mp.group(1).strip()

        # Extract village candidate = before "Panchayat" if present
        if "panchayat" in vl:
            before_pan = re.split(r"\bpanchayat\b", v2, flags=re.I)[0].strip()
            before_pan = re.sub(r"^\s*village\s+name\s*(?:is)?\s*", "", before_pan, flags=re.I).strip()
            if before_pan:
                m["village"] = before_pan

    # Clean panchayat if it contains "in ... block"
    if isinstance(m.get("panchayat"), str):
        p = clean_place(m["panchayat"])
        if p:
            p = re.sub(r"\s+in\s+.*?\bblock\b.*$", "", p, flags=re.I).strip()
            p = re.sub(r"\bpanchayat\b", "", p, flags=re.I).strip()
            m["panchayat"] = p or None
        else:
            m["panchayat"] = None

    # Clean block
    if isinstance(m.get("block"), str):
        b = clean_place(m["block"])
        if b:
            b = re.sub(r"\bblock\b", "", b, flags=re.I).strip()
            m["block"] = b or None
        else:
            m["block"] = None

    # Event location: remove trailing "village"
    if isinstance(m.get("event_location"), str):
        el = clean_place(m["event_location"])
        if el:
            el = re.sub(r"\bvillage\b$", "", el, flags=re.I).strip()
            m["event_location"] = el or None
        else:
            m["event_location"] = None

    # if event_location missing, fallback to village
    if not m.get("event_location") and m.get("village"):
        m["event_location"] = m["village"]

    return m


def postprocess_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final cleanup step — removes markdown/noise and normalizes types/formatting.
    """
    m = dict(meta)

    # places
    for k in ("village", "panchayat", "block", "district", "event_location"):
        m[k] = clean_place(m.get(k))

    # split combined clause (critical for your issue)
    m = _split_combined_places(m)

    # names
    m["sarpanch_name"] = clean_person_name(m.get("sarpanch_name"), drop_honorifics=True)
    m["coordinator_name"] = clean_person_name(m.get("coordinator_name"), drop_honorifics=True)
    m["reporting_manager_name"] = clean_person_name(m.get("reporting_manager_name"), drop_honorifics=True)

    # phone
    m["phone_number"] = normalize_phone(m.get("phone_number"))

    # # counts -> int
    # for k in ("farmers_attended_total", "female_farmers_count", "male_farmers_count"):
    #     if m.get(k) is None:
    #         continue
    #     try:
    #         m[k] = int(str(m[k]).strip())
    #     except Exception:
    #         m[k] = to_int_maybe(m[k])

    # ------------------------------------------------------------------
    # CONSISTENCY FIX FOR FARMER COUNTS
    # ------------------------------------------------------------------
    
    total = m.get("farmers_attended_total")
    male = m.get("male_farmers_count")
    female = m.get("female_farmers_count")
    
    # Ensure ints or None
    total = int(total) if isinstance(total, int) else total
    male = int(male) if isinstance(male, int) else male
    female = int(female) if isinstance(female, int) else female
    
    # Case 1: total + male known → derive female
    if total is not None and male is not None:
        derived = max(total - male, 0)
        if female is None:
            m["female_farmers_count"] = derived
    
    # Case 2: total + female known → derive male
    elif total is not None and female is not None:
        derived = max(total - female, 0)
        if male is None:
            m["male_farmers_count"] = derived
    
    # Case 3: female still None → default to 0
    if m.get("female_farmers_count") is None:
        m["female_farmers_count"] = 0
    
    # Case 4: male still None but total exists → assume all male
    if m.get("male_farmers_count") is None and total is not None:
        m["male_farmers_count"] = total

    # times
    m["event_start_time"] = normalize_time(m.get("event_start_time"))
    m["event_end_time"] = normalize_time(m.get("event_end_time"))

    # If day missing but date present, compute it
    if not m.get("day") and m.get("date"):
        m["day"] = weekday_from_date(m["date"])
    # day capitalization
    if m.get("day"):
        m["day"] = clean_value(m["day"]).capitalize()

    # remove junk markers
    for k, v in list(m.items()):
        if isinstance(v, str) and v.strip() in ("", "*", "**"):
            m[k] = None

    return m


# =============================================================================
# REGEX EXTRACTOR
# =============================================================================

def extract_meta_regex(text: str) -> Dict[str, Any]:
    text = normalize_text(text)
    out = dict(SCHEMA)

    # Stop at comma/period/end OR when the next field label starts
    STOP = r"(?=\s*(?:,|\.|$|\bvillage\b|\bpanchayat\b|\bblock\b|\bdistrict\b|\bcoordinator\b|\breporting\b|\bsarpanch\b|\bevent\b|\bphone\b))"

    # date
    out["date"] = clean_value(first_match([
        rf"(?:today'?s\s+date\s*(?:is)?\s*)(.*?){STOP}",
        rf"(?:\bdate\b\s*)(\d{{4}}-\d{{2}}-\d{{2}}){STOP}",
        rf"(?:\bdate\b\s*)(\d{{1,2}}[\/\-]\d{{1,2}}[\/\-]\d{{2,4}}){STOP}",
        rf"(?:\bdate\b\s*)([A-Za-z]+\s+\d{{1,2}},\s*\d{{4}}){STOP}",
    ], text))

    # day
    # out["day"] = clean_value(first_match([
    #     rf"(?:\bday\b\s*(?:is)?\s*)([A-Za-z]+){STOP}",
    # ], text))
    out["day"] = clean_value(first_match([
        rf"(?:\bday\b\s*(?:is)?\s*)([A-Za-z]+){STOP}",
        rf"(?:\btoday\b\s*(?:is)?\s*)(monday|tuesday|wednesday|thursday|friday|saturday|sunday){STOP}",
    ], text))

    # village / panchayat / block
    out["village"] = clean_value(first_match([
        rf"(?:village\s+name\s*(?:is)?\s*)(.*?){STOP}",
        rf"(?:\bvillage\b\s*)(.*?){STOP}",
    ], text))

    out["panchayat"] = clean_value(first_match([
        rf"(?:panchayat\s+name\s*(?:is)?\s*)(.*?){STOP}",
        rf"(?:\bpanchayat\b\s*)(.*?){STOP}",
    ], text))

    # out["block"] = clean_value(first_match([
    #     rf"(?:\bblock\b\s*(?:is)?\s*)(.*?){STOP}",
    # ], text))
    # out["block"] = out["block"] or clean_value(first_match([
    #     rf"(?:\bblock\b\s*)([A-Za-z][A-Za-z\s]+?){STOP}",
    # ], text))
    # Block (IndicTrans2 often gives: "Block Shri Chamkaur Sahib Coordinator ...")
    out["block"] = clean_value(first_match([
        rf"\bblock\b\s*(?:name\s*)?(?:is\s*)?([A-Za-z][A-Za-z\s]+?){STOP}",
        rf"\bblock\b\s*[:\-]?\s*([A-Za-z][A-Za-z\s]+?){STOP}",
        rf"\bblok\b\s*(?:nem|name)?\s*(?:is\s*)?([A-Za-z][A-Za-z\s]+?){STOP}",  # "Blok Nem"
    ], text))

    # names
    out["coordinator_name"] = clean_value(first_match([
        rf"(?:coordinator\s+name\s*(?:is)?\s*)(.*?){STOP}",
        rf"(?:name\s+of\s+the\s+coordinator\s*)(.*?){STOP}",
    ], text))

    out["reporting_manager_name"] = clean_value(first_match([
        rf"(?:reporting\s+manager\s*(?:name)?\s*(?:is)?\s*)(.*?){STOP}",
        rf"(?:name\s+of\s+the\s+reporting\s+manager\s*)(.*?){STOP}",
    ], text))

    out["sarpanch_name"] = clean_value(first_match([
        rf"(?:sarpanch\s+name\s*(?:is)?\s*)(.*?){STOP}",
        rf"(?:name\s+of\s+the\s+sarpanch\s*)(.*?){STOP}",
    ], text))

    # location
    out["event_location"] = clean_value(first_match([
        rf"(?:meeting\s+location\s*)(.*?){STOP}",
        rf"(?:event\s+location\s*)(.*?){STOP}",
    ], text))

    # district
    out["district"] = clean_value(first_match([
        rf"(?:\bdistrict\b\s*)(.*?){STOP}",
    ], text))

    # phone
    phone_raw = first_match([
        r"(?:phone\s+number\s*(?:is)?\s*)(\+?\d[\d\s]{8,}\d)",
        r"(?:\bphone\b\s*(?:no|number)?\s*(?:is|:)?\s*)(\+?\d[\d\s]{8,}\d)",
    ], text)
    out["phone_number"] = normalize_phone(phone_raw)

    # total farmers
    tf = first_match([
        r"number\s+of\s+total\s+farmers\s*\.\s*([A-Za-z]+|\d+)",
        r"number\s+of\s+total\s+farmers\s*[:\-]?\s*([A-Za-z]+|\d+)",
        r"\btotal\s+farmers\b\s*[:\-]?\s*([A-Za-z]+|\d+)",
        r"\bno\s+of\s+farmers\s+attended\b\s*[:\-]?\s*([A-Za-z]+|\d+)",
    ], text)
    out["farmers_attended_total"] = to_int_maybe(tf)

    # female / male farmers
    female_raw = first_match([r"female\s+farmers\s*,?\s*(nil|none|\d+|[A-Za-z]+)"], text)
    male_raw   = first_match([r"male\s+farmers\s*,?\s*(nil|none|\d+|[A-Za-z]+)"], text)

    f_int = to_int_maybe(female_raw)
    m_int = to_int_maybe(male_raw)

    # Special-case: "male farmers, nil ... eight" -> choose last numeric token
    m_clause = re.search(r"male\s+farmers([^\.]{0,80})", text, re.I)
    if m_clause:
        tail = m_clause.group(1).lower()
        tokens = re.findall(
            r"\b(nil|none|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)\b",
            tail,
        )
        if tokens:
            last_int = to_int_maybe(tokens[-1])
            if tokens[0] in ("nil", "none", "zero") and last_int not in (None, 0):
                m_int = last_int

    out["female_farmers_count"] = f_int
    out["male_farmers_count"] = m_int

    # start time
    ms = re.search(
        r"(?:event\s+start(?:\s+time)?\s*(?:is)?\s*)(\d{1,2})(?::(\d{2}))?\s*(am|pm)",
        text,
        re.I,
    )
    if ms:
        hh, mm, ap = ms.group(1), ms.group(2), ms.group(3).upper()
        out["event_start_time"] = f"{hh}{(':' + mm) if mm else ''}{ap}"

    # end time (sometimes "approximately")
    me = re.search(
        r"(?:event\s+end(?:\s+time)?\s*(?:is)?\s*(?:approximately)?\s*)(\d{1,2})(?::(\d{2}))?\s*(am|pm)",
        text,
        re.I,
    )
    if me:
        hh, mm, ap = me.group(1), me.group(2), me.group(3).upper()
        out["event_end_time"] = f"{hh}{(':' + mm) if mm else ''}{ap}"

    # normalize times into consistent display
    out["event_start_time"] = normalize_time(out.get("event_start_time"))
    out["event_end_time"] = normalize_time(out.get("event_end_time"))

    return out


# =============================================================================
# LLM BACKFILL PROMPT
# =============================================================================

def build_fill_prompt(en_text: str, current: Dict[str, Any]) -> str:
    return f"""
Fill ONLY the missing fields for this JSON. Do not change existing values.
Return ONLY one JSON object with the same keys as the schema.
If still not present, keep null.
Do NOT use markdown formatting (no **bold**, no bullets). Return plain text values only.

Schema keys:
{list(SCHEMA.keys())}

Current JSON:
{json.dumps(current, ensure_ascii=False)}

Transcript:
\"\"\"{en_text[:8000]}\"\"\"

Return JSON only.
""".strip()


# =============================================================================
# METADATA EXTRACTOR
# =============================================================================

class MetadataExtractor(BaseLLM):
    """
    Regex-first metadata extractor with optional LLM backfill.
    """

    def __init__(self, base: Optional[BaseLLM] = None, device: Optional[str] = None):
        if base is not None:
            # share loaded model/tokenizer/device
            self.model = base.model
            self.tokenizer = base.tokenizer
            self.device = base.device
        else:
            super().__init__(device=device)

    def extract(self, english_text: str, use_llm: bool = True, window: int = 7000) -> Dict[str, Any]:
        win = pick_relevant_window(english_text, window=window)
        base = extract_meta_regex(win)

        if use_llm and any(base.get(k) is None for k in SCHEMA.keys()):
            try:
                base = self._llm_fill_missing(win, base)
            except Exception:
                log.exception("LLM backfill failed; returning regex-only metadata.")

        # Always sanitize final output (fixes panchayat/block/location noise)
        base = postprocess_metadata(base)
        return base

    def _llm_fill_missing(self, en_text: str, current: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_fill_prompt(en_text, current)
        messages = [{"role": "user", "content": prompt}]

        raw = self._run_inference(messages, max_new_tokens=450)
        obj = self._safe_json(raw, fallback={})

        merged = dict(current)
        for k in SCHEMA.keys():
            if merged.get(k) is None and obj.get(k) is not None:
                merged[k] = obj.get(k)

        return merged