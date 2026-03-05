"""
pipeline/extraction/participants.py

LLM-first farmer participant extraction with deterministic post-processing.

Why this exists
---------------
ASR/translation transcripts often contain semi-structured enumerations like:
  "First farmer Surinder Singh, phone 9143..., land 3 acres..."
and also lots of noisy tokens:
  "Karamjit Singh phone", "Main Crops", "Qualification", "Animals", etc.

This extractor follows the same style as insights.py / terminology.py:
  1) Chunk transcript and ask the LLM to output STRICT JSON for participants.
  2) Strong normalization + validation (phones, land, names).
  3) De-duplication + ordinal stabilization.

Returned schema (list):
[
  {
    "ordinal": "1",
    "name": "Surinder Singh",
    "phone_number": "9143294053",
    "total_land_acres": 3,
    "qualification": "10th",
    "animals": "Cow, Buffalo",
    "main_crops": "Wheat, Paddy",
    "notes": "Progressive farmer"
  },
  ...
]
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pipeline.extraction.base_llm import BaseLLM
from pipeline.transcript.utils import chunk_entries, format_transcript

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------

STOPWORDS_IN_NAME = {
    "phone", "number", "mobile", "contact", "qualification", "animals", "animal",
    "main", "crops", "crop", "notes", "note", "land", "acre", "acres", "hectare",
    "farmer", "participant",
}

QUALI_RE = re.compile(r"\b(illiterate|literate|primary|middle|matric|10th|12th|graduate|post\s*graduate|diploma)\b", re.I)

def _ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip()

def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")

def normalize_phone(x: Any) -> Optional[str]:
    p = _digits_only(str(x or ""))
    if len(p) < 10:
        return None
    # keep last 10 digits (Indian numbers sometimes include +91 / 0)
    p = p[-10:]
    return p if len(p) == 10 else None

def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
        
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    s = s.replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def _clean_listish(v: Any) -> Optional[str]:
    """
    Accepts list/str and returns a comma-separated title-cased string.
    """
    if v is None:
        return None
    items: List[str] = []
    if isinstance(v, list):
        for x in v:
            if x is None:
                continue
            s = _ws(str(x))
            if s:
                items.append(s)
    else:
        s = _ws(str(v))
        if s:
            # split on common separators
            parts = re.split(r"[;,/]| and ", s, flags=re.I)
            items.extend([_ws(p) for p in parts if _ws(p)])

    # normalize capitalization and dedupe while preserving order
    seen = set()
    out: List[str] = []
    for it in items:
        it = it.strip(" .:-")
        if not it:
            continue
        it = " ".join(w.capitalize() for w in it.split())
        key = it.lower()
        if key not in seen:
            seen.add(key)
            out.append(it)
    return ", ".join(out) if out else None

def clean_name(name: Any) -> Optional[str]:
    if name is None:
        return None
    s = _ws(str(name))
    if not s:
        return None

    # Remove obvious label tails, e.g. "Karamjit Singh phone", "Jasdeep Singh Phone"
    s = re.sub(r"\b(phone|mobile|number|contact)\b.*$", "", s, flags=re.I).strip()
    s = re.sub(r"\b(main\s*crops?|qualification|animals?|notes?)\b.*$", "", s, flags=re.I).strip()

    # Strip punctuation noise
    s = s.strip(" -:,.|/")

    # Drop if it's clearly a field label, not a person name
    lower = s.lower()
    if lower in {"main crops", "qualification", "animals", "notes"}:
        return None

    # Remove digits embedded in the name
    s = re.sub(r"\d+", "", s).strip()
    s = _ws(s)

    # Drop if too short or contains only stopwords
    toks = [t for t in re.split(r"\s+", s) if t]
    if len(toks) < 2:
        return None
    if all(t.lower() in STOPWORDS_IN_NAME for t in toks):
        return None

    # Title-case (but keep common particles)
    s = " ".join(w.capitalize() for w in s.split())
    return s or None

def looks_like_person_name(s: str) -> bool:
    s = _ws(s or "")
    if not s:
        return False
    if len(s) < 5:
        return False
    if re.search(r"\d", s):
        return False
    if any(tok.lower() in STOPWORDS_IN_NAME for tok in s.split()):
        # allow "Singh" etc; reject if it contains many labels
        label_hits = sum(1 for tok in s.split() if tok.lower() in STOPWORDS_IN_NAME)
        if label_hits >= 2:
            return False
    # at least 2 alpha tokens
    alpha_tokens = [t for t in s.split() if re.fullmatch(r"[A-Za-z]+", t)]
    return len(alpha_tokens) >= 2

# ---------------------------------------------------------------------
# LLM-first extractor
# ---------------------------------------------------------------------

DEFAULT_ROW = {
    "ordinal": None,
    "name": None,
    "phone_number": None,
    "total_land_acres": None,
    "qualification": None,
    "animals": None,
    "main_crops": None,
    "notes": None,
}

class ParticipantExtractor(BaseLLM):
    """
    Extracts ONLY farmer participant records from the transcript.
    """

    async def extract(self, entries, use_llm: bool = True, **kwargs) -> List[Dict]:
        # Accept either raw text (str) or list[dict] entries
        if isinstance(entries, str):
            chunks = [entries]   # treat as one chunk of transcript text
        else:
            # filter only dicts to avoid 'str has no get' type crashes
            safe_entries = [e for e in (entries or []) if isinstance(e, dict)]
            chunks = chunk_entries(safe_entries, max_chars=8000, text_key="translated_text")
    
        log.info(f"ParticipantExtractor: {len(chunks)} chunk(s)...")
    
        all_rows: List[Dict[str, Any]] = []
    
        for i, chunk in enumerate(chunks):
            # chunk is either a string transcript OR a list[dict]
            transcript = chunk if isinstance(chunk, str) else format_transcript(chunk)
            if len((transcript or "").strip()) < 80:
                continue
    
            log.info(f"  Chunk {i+1}/{len(chunks)}")
    
            # Optional: if use_llm is False, skip LLM and return empty list (or keep rule-based fallback)
            if not use_llm:
                continue
    
            rows = self._extract_from_chunk(transcript).get("participants", [])
            if isinstance(rows, list):
                all_rows.extend(rows)


        # Post-process + validate
        cleaned: List[Dict[str, Any]] = []
        for r in all_rows:
            row = dict(DEFAULT_ROW)
            if isinstance(r, dict):
                row.update(r)

            row["ordinal"] = _ws(str(row.get("ordinal") or "")) or None
            row["name"] = clean_name(row.get("name"))
            row["phone_number"] = normalize_phone(row.get("phone_number"))
            row["total_land_acres"] = _coerce_float(row.get("total_land_acres"))
            row["qualification"] = _ws(str(row.get("qualification") or "")) or None
            row["animals"] = _clean_listish(row.get("animals"))
            row["main_crops"] = _clean_listish(row.get("main_crops"))
            row["notes"] = _ws(str(row.get("notes") or "")) or None

            # If LLM forgot to move qualification into the field, try to infer from notes/name text
            if not row["qualification"]:
                blob = " ".join([str(r.get("qualification") or ""), str(r.get("notes") or ""), str(r.get("name") or "")])
                m = QUALI_RE.search(blob or "")
                if m:
                    row["qualification"] = m.group(1)

            # Final sanity: require a plausible name OR phone number
            if row["name"] and not looks_like_person_name(row["name"]):
                row["name"] = None

            if not row["name"] and not row["phone_number"]:
                continue

            cleaned.append(row)

        cleaned = self._dedupe(cleaned)
        cleaned = self._stabilize_ordinals(cleaned)

        return {
        "total_count": len(cleaned),
        "farmers": cleaned,   # matches your old extractor keys
    }

    # ------------------------------------------------------------------

    def _extract_from_chunk(self, transcript: str) -> Dict[str, Any]:
        """
        Ask the LLM to extract participant cards in STRICT JSON.
        """
        messages = [
            {
                "role": "user",
                "content": f"""
You are extracting farmer participant details from a rural meeting transcript.

STRICT RULES:
- Output ONLY farmers / participants (NOT officers, coordinators, sarpanch, interviewers).
- A participant record is typically introduced with ordinal words (first/second/third) OR a clear participant card
  (name + phone/land/qualification/animals/crops).
- DO NOT treat field labels as names: "Main Crops", "Qualification", "Animals", "Notes" are NOT participant names.
- If a line says: "Karamjit Singh phone number ..." → name MUST be "Karamjit Singh" (remove the word "phone").
- Convert spoken phone numbers to digits if present; otherwise use null.
- If a field is unknown, set it to null (do NOT guess).
- Do NOT invent participants.

Transcript:
{transcript}

Return STRICT JSON only:
{{
  "participants": [
    {{
      "ordinal": "1",
      "name": "",
      "phone_number": "",
      "total_land_acres": 0,
      "qualification": "",
      "animals": "",
      "main_crops": "",
      "notes": ""
    }}
  ]
}}
""",
            }
        ]

        decoded = self._run_inference(messages, max_new_tokens=900)
        return self._safe_json(decoded, {"participants": []})

    # ------------------------------------------------------------------

    def _dedupe(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge duplicates using phone number as primary key, else name.
        Keeps the most complete record.
        """
        def score(r: Dict[str, Any]) -> int:
            return sum(1 for k in DEFAULT_ROW.keys() if r.get(k) not in (None, "", []))

        by_key: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []

        for r in rows:
            phone = r.get("phone_number")
            name = (r.get("name") or "").lower().strip()
            key = phone or (f"name:{name}" if name else None)

            if not key:
                continue

            if key not in by_key:
                by_key[key] = r
                order.append(key)
                continue

            # merge: keep non-null fields; prefer higher score
            cur = by_key[key]
            better = r if score(r) > score(cur) else cur
            worse  = cur if better is r else r

            merged = dict(better)
            for k in DEFAULT_ROW.keys():
                if merged.get(k) in (None, "", []):
                    merged[k] = worse.get(k)
            by_key[key] = merged

        return [by_key[k] for k in order]

    def _stabilize_ordinals(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure ordinals are sequential strings if missing / duplicated.
        Preserve existing numeric ordinals when possible.
        """
        # Parse numeric ordinals where possible
        parsed: List[Tuple[int, Dict[str, Any]]] = []
        unparsed: List[Dict[str, Any]] = []

        for r in rows:
            o = r.get("ordinal")
            if o is None:
                unparsed.append(r)
                continue
            m = re.search(r"\d+", str(o))
            if m:
                parsed.append((int(m.group(0)), r))
            else:
                unparsed.append(r)

        # Sort those with ordinals
        parsed.sort(key=lambda x: x[0])

        # Assign sequential for unparsed, starting after max existing
        max_o = parsed[-1][0] if parsed else 0
        for i, r in enumerate(unparsed, start=max_o + 1):
            r["ordinal"] = str(i)

        # Also normalize parsed ordinals to strings
        for o, r in parsed:
            r["ordinal"] = str(o)

        return [r for _, r in parsed] + unparsed
