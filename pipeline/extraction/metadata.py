"""
pipeline/extraction/metadata.py

LLM-first metadata extraction with lightweight deterministic post-processing.

Goal
----
metadata is a small, well-defined schema (date, village, block, counts, etc.).
Regex-only extraction is brittle because transcripts vary a lot. This extractor
uses the LLM as the primary parser, and uses regex only as a *fallback* for
missing fields.

Behavior
--------
1) LLM extracts STRICT JSON for the schema from a relevant text window.
2) Post-processing:
   - normalize phone/time
   - compute 'day' from 'date' if missing
   - derive female_count = total - male when possible (and vice versa)
   - split combined phrases where a field includes multiple places
3) Fallback regex fills only missing keys (never overwrites LLM values).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pipeline.extraction.base_llm import BaseLLM
from pipeline.transcript.utils import format_transcript

log = logging.getLogger(__name__)

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

HONORIFICS_RE = re.compile(r"^\s*(shri|smt|mr|mrs|ms|miss|dr)\.?\s+", re.I)

NUMWORDS = {
    "zero": 0, "nil": 0, "none": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _ws(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").replace("\n", " ")).strip()

def strip_markdown(s: str) -> str:
    s = s or ""
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s)      # **bold**
    s = re.sub(r"`([^`]+)`", r"\1", s)          # inline code
    s = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", s) # markdown links
    return _ws(s)

def clean_value(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = strip_markdown(str(v))
    s = s.strip(" :-,")
    s = _ws(s)
    return s or None

def clean_place(v: Any) -> Optional[str]:
    s = clean_value(v)
    if not s:
        return None
    # remove leading connectors
    s = re.sub(r"^(in|at|near|under|within)\s+", "", s, flags=re.I).strip()
    # remove trailing generic words
    s = re.sub(r"\b(village|district|block|panchayat)\b$", "", s, flags=re.I).strip()
    return _ws(s) or None

def clean_person_name(v: Any) -> Optional[str]:
    s = clean_value(v)
    if not s:
        return None
    s = HONORIFICS_RE.sub("", s).strip()
    # remove label tails
    s = re.sub(r"\b(phone|mobile|number)\b.*$", "", s, flags=re.I).strip()
    # title-case (simple)
    s = " ".join(w.capitalize() for w in s.split())
    return s or None

def normalize_phone(v: Any) -> Optional[str]:
    if v is None:
        return None
    digits = re.sub(r"\D", "", str(v))
    if len(digits) < 10:
        return None
    return digits[-10:]

def to_int_maybe(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    s = str(v).strip().lower()
    if not s:
        return None
    # word numbers
    if s in NUMWORDS:
        return NUMWORDS[s]
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None

def normalize_time(v: Any) -> Optional[str]:
    """
    Normalize times to HH:MM (24h) when possible; else return cleaned string.
    """
    s = clean_value(v)
    if not s:
        return None

    # Common: "2:30 pm", "14:30", "2 pm"
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", s, re.I)
    if not m:
        return s

    hh = int(m.group(1))
    mm = int(m.group(2) or "00")
    ampm = (m.group(3) or "").lower()

    if ampm == "pm" and hh < 12:
        hh += 12
    if ampm == "am" and hh == 12:
        hh = 0

    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return f"{hh:02d}:{mm:02d}"
    return s

def weekday_from_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    try:
        # accept YYYY-MM-DD
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%A")
    except Exception:
        return None

def pick_relevant_window(text: str, window: int = 7500) -> str:
    """
    Metadata usually appears near the beginning. Keep a window that includes
    the start + any explicit "metadata / details" section if present.
    """
    t = text or ""
    t = t.strip()
    if len(t) <= window:
        return t

    anchors = [
        r"\bmetadata\b",
        r"\bmeeting details\b",
        r"\bevent details\b",
        r"\bdate\b",
        r"\bvillage\b",
        r"\bpanchayat\b",
        r"\bblock\b",
        r"\bdistrict\b",
    ]
    for a in anchors:
        m = re.search(a, t, re.I)
        if m and m.start() < len(t):
            start = max(0, m.start() - 1200)
            return t[start : start + window]
    return t[:window]

# ---------------------------------------------------------------------
# Fallback regex (fills only missing keys)
# ---------------------------------------------------------------------

def _first_match(patterns: List[str], text: str) -> Optional[str]:
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            return m.group(1).strip()
    return None

def extract_meta_regex(text: str) -> Dict[str, Any]:
    t = text or ""
    out: Dict[str, Any] = {}

    date = _first_match([
        r"\bdate\b\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{2}/\d{2}/\d{4})\b",
    ], t)
    if date and "/" in date:
        try:
            d = datetime.strptime(date, "%d/%m/%Y").strftime("%Y-%m-%d")
            date = d
        except Exception:
            pass
    if date:
        out["date"] = date

    out["village"] = _first_match([r"\bvillage\b\s*[:\-]?\s*([A-Za-z][A-Za-z\s]+)"], t)
    out["panchayat"] = _first_match([r"\bpanchayat\b\s*[:\-]?\s*([A-Za-z][A-Za-z\s]+)"], t)
    out["block"] = _first_match([r"\bblock\b\s*[:\-]?\s*([A-Za-z][A-Za-z\s]+)"], t)
    out["district"] = _first_match([r"\bdistrict\b\s*[:\-]?\s*([A-Za-z][A-Za-z\s]+)"], t)

    phone = _first_match([r"\b(phone|mobile)\b\s*(?:number)?\s*[:\-]?\s*([+()\d\s\-]{8,})"], t)
    if phone:
        out["phone_number"] = phone

    total = _first_match([r"\b(total|farmers attended)\b.*?(\d{1,4})\b"], t)
    if total:
        out["farmers_attended_total"] = total

    male = _first_match([r"\bmale\b.*?(\d{1,4})\b"], t)
    if male:
        out["male_farmers_count"] = male

    female = _first_match([r"\bfemale\b.*?(\d{1,4})\b"], t)
    if female:
        out["female_farmers_count"] = female

    coord = _first_match([r"\bcoordinator\b\s*[:\-]?\s*([A-Za-z][A-Za-z\s]+)"], t)
    if coord:
        out["coordinator_name"] = coord

    mgr = _first_match([r"\b(reporting manager|manager)\b\s*[:\-]?\s*([A-Za-z][A-Za-z\s]+)"], t)
    if mgr:
        out["reporting_manager_name"] = mgr

    sar = _first_match([r"\bsarpanch\b\s*(?:name)?\s*[:\-]?\s*([A-Za-z][A-Za-z\s]+)"], t)
    if sar:
        out["sarpanch_name"] = sar

    return out

# ---------------------------------------------------------------------
# Postprocess
# ---------------------------------------------------------------------

def _split_combined_places(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sometimes one field contains a combined clause:
      "Gardi Pharid Panchayat in Shri Chamkaur Sahib Block"
    Split gently if obvious.
    """
    p = meta.get("panchayat")
    b = meta.get("block")

    if p and (not b) and re.search(r"\bblock\b", p, re.I):
        # attempt split: "... Panchayat ... Block ..."
        m = re.search(r"(.+?)\b(block)\b\s*(.+)$", p, re.I)
        if m:
            left = clean_place(m.group(1))
            right = clean_place(m.group(3))
            if left:
                meta["panchayat"] = left
            if right:
                meta["block"] = right

    if b and re.search(r"\bpanchayat\b", b, re.I) and not p:
        m = re.search(r"(.+?)\b(panchayat)\b\s*(.+)$", b, re.I)
        if m:
            left = clean_place(m.group(1))
            right = clean_place(m.group(3))
            if left:
                meta["block"] = left
            if right:
                meta["panchayat"] = right

    return meta

def postprocess_metadata(meta: Dict[str, Any], evidence_text: str = "") -> Dict[str, Any]:
    out = dict(SCHEMA)
    out.update(meta or {})

    # --- normalize evidence text once ---
    ev = (evidence_text or "")

    # basic cleaning
    out["date"] = clean_value(out.get("date"))
    out["day"] = clean_value(out.get("day"))
    out["village"] = clean_place(out.get("village"))
    out["panchayat"] = clean_place(out.get("panchayat"))
    out["block"] = clean_place(out.get("block"))
    out["district"] = clean_place(out.get("district"))
    out["event_location"] = clean_place(out.get("event_location"))

    out["sarpanch_name"] = clean_person_name(out.get("sarpanch_name"))
    out["coordinator_name"] = clean_person_name(out.get("coordinator_name"))
    out["reporting_manager_name"] = clean_person_name(out.get("reporting_manager_name"))

    out["phone_number"] = normalize_phone(out.get("phone_number"))

    out["farmers_attended_total"] = to_int_maybe(out.get("farmers_attended_total"))
    out["female_farmers_count"] = to_int_maybe(out.get("female_farmers_count"))
    out["male_farmers_count"] = to_int_maybe(out.get("male_farmers_count"))

    out["event_start_time"] = normalize_time(out.get("event_start_time"))
    out["event_end_time"] = normalize_time(out.get("event_end_time"))

    # ------------------------------------------------------------------
    # Evidence-based fixes (only fill missing values; never overwrite)
    # ------------------------------------------------------------------

    # 1) DAY extraction when transcript contains "DAY THURSDAY" etc.
    if (not out.get("day")) and ev:
        m = re.search(r"\bday\b\s+([A-Za-z]+)\b", ev, flags=re.I)
        if m:
            out["day"] = m.group(1).strip().capitalize()

    # 2) If date is still missing and you have a natural-date parser, use it
    #    (safe: only fills when missing)
    if (not out.get("date")) and ev:
        try:
            # parse_natural_date should return "YYYY-MM-DD" or None
            d = parse_natural_date(ev)  # <-- must exist in your file; if not, remove this block
            if d:
                out["date"] = d
        except NameError:
            pass

    # compute day from date if missing
    if not out.get("day") and out.get("date"):
        out["day"] = weekday_from_date(out["date"])

    # 3) Explicit "no female" / "female nil/null/zero" => female_farmers_count = 0
    #    only if still None (don’t override actual numbers)
    if out.get("female_farmers_count") is None and ev:
        if re.search(r"\b(no female|female\s+(nil|null|zero))\b", ev, flags=re.I):
            out["female_farmers_count"] = 0

    # derive missing gender counts when possible
    total = out.get("farmers_attended_total")
    male = out.get("male_farmers_count")
    female = out.get("female_farmers_count")
    if total is not None:
        if female is None and male is not None:
            out["female_farmers_count"] = max(total - male, 0)
        if male is None and female is not None:
            out["male_farmers_count"] = max(total - female, 0)
    # if both missing but total exists, keep as None (do not guess)
    # Case 1: total + male known → derive female
    if total is not None and male is not None:
        derived = max(total - male, 0)
        if female is None:
            out["female_farmers_count"] = derived
    
    # Case 2: total + female known → derive male
    elif total is not None and female is not None:
        derived = max(total - female, 0)
        if male is None:
            out["male_farmers_count"] = derived
    
    # Case 3: female still None → default to 0
    if out.get("female_farmers_count") is None:
        out["female_farmers_count"] = 0
    
    # Case 4: male still None but total exists → assume all male
    if out.get("male_farmers_count") is None and total is not None:
        out["male_farmers_count"] = total


    out = _split_combined_places(out)
    return out

# =============================================================================
# LLM FILL PROMPT
# =============================================================================
def build_fill_prompt(en_text: str, current: Dict[str, Any]) -> str:
    return f"""
You extract structured meeting metadata from a transcript.

STRICT RULES:
- Use ONLY facts explicitly present in the text.
- If a value is unknown / not stated, return null.
- Do NOT invent names/places/counts.
- Do NOT include extra keys.
- Prefer ISO date format YYYY-MM-DD when the date is explicit.
- Phone number must be digits only (10 digits if possible); otherwise null.
- If both male/female counts are not stated, keep them null (do not compute).
- Return ONLY one JSON object with the same keys as the schema.

Schema keys:
{list(SCHEMA.keys())}

Current JSON:
{json.dumps(current, ensure_ascii=False)}

Transcript:
\"\"\"{en_text[:8000]}\"\"\"

Return JSON only.
""".strip()


# ---------------------------------------------------------------------
# LLM-first extractor
# ---------------------------------------------------------------------

class MetadataExtractor(BaseLLM):

    async def extract(self, entries: List[Dict], use_llm: bool = True, **kwargs) -> Dict[str, Any]:
        if isinstance(entries, str):
            transcript = entries
        else:
            transcript = format_transcript(entries)

        window = pick_relevant_window(transcript, window=9000)

        # IMPORTANT: seed "current" with the schema so LLM keeps exact keys
        llm_meta = dict(SCHEMA)

        if use_llm:
            llm_meta = self._extract_llm(window, current=llm_meta)

        llm_meta = postprocess_metadata(llm_meta, evidence_text=window)

        # fallback regex fills only missing keys
        regex_meta = extract_meta_regex(window)
        for k, v in regex_meta.items():
            if llm_meta.get(k) in (None, "", []):
                llm_meta[k] = v

        llm_meta = postprocess_metadata(llm_meta, evidence_text=window)
        return llm_meta

    def _extract_llm(self, text: str, current: Dict[str, Any]) -> Dict[str, Any]:
        prompt = build_fill_prompt(text, current=current)

        messages = [{"role": "user", "content": prompt}]
        decoded = self._run_inference(messages, max_new_tokens=550)

        # Always coerce to schema keys
        return self._safe_json(decoded, dict(SCHEMA))