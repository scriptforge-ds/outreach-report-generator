"""
Extracts ONLY farmer participant details from the transcript.

Returns farmers with fields:
- ordinal
- name
- phone_number
- total_land_acres
- qualification
- animals
- main_crops
- notes
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pipeline.extraction.base_llm import BaseLLM
from pipeline.transcript.utils import format_transcript

log = logging.getLogger(__name__)

# =========================
# 0) Utilities
# =========================
def _ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip()

def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")

def valid_phone(p: Any) -> Optional[str]:
    p = _digits_only(str(p or ""))
    return p if 10 <= len(p) <= 13 else None

def normalize_phone(x: Any) -> Optional[str]:
    p = valid_phone(x)
    if not p:
        return None
    return p[-10:]  # keep last 10 digits

# =========================
# 1) Normalization helpers
# =========================
NUM_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
}

def normalize_text(t: str) -> str:
    t = (t or "").replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = re.sub(r"\s+", " ", str(name)).strip()
    s = re.sub(r"\s+(Farmer|Phone|Phone Number|Number|No\.?|Contact)\b.*$", "", s, flags=re.I).strip()
    s = re.sub(r"^\s*farmer\s+", "", name, flags=re.I).strip()
    s = re.sub(r"[,\.;:\-]+$", "", s).strip()
    return s if s else None

BAD_NAME_PHRASES = {
    "names and", "name and", "farmers names", "farmers name", "phone numbers",
    "phone number", "numbers are", "are listed", "names are", "farmers' names",
    "farmers’ names", "farmers", "farmer", "total farmers", "male farmers", "female farmers",
    "meeting schedule", "event start", "event end", "day monday", "day tuesday",
}

COMMON_SURNAMES = re.compile(
    r"\b(singh|kaur|devi|kumar|lal|ram|das|begum|ali|shaikh|sheikh)\b", re.I
)

def looks_like_person_name(name: str) -> bool:
    if not name:
        return False

    n = normalize_text(name).lower()
    n = re.sub(r"[^a-z\s']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()

    if n in BAD_NAME_PHRASES:
        return False

    # reject meta words unless surname marker exists
    if any(w in n.split() for w in ["names", "name", "numbers", "number", "listed", "farmers", "phone", "schedule"]):
        if not COMMON_SURNAMES.search(n):
            return False

    parts = n.split()
    if len(parts) < 2:
        return False

    if COMMON_SURNAMES.search(n):
        return True

    generic = {"names", "name", "numbers", "number", "listed", "farmers", "phone", "schedule", "meeting", "event", "total", "male", "female"}
    if any(p in generic for p in parts):
        return False

    return True

def spoken_to_digits(s: str) -> str:
    """
    Convert spoken number phrases into digits.
    Handles: 'double nine', 'eight five seven one seven', etc.
    Keeps existing digits too.
    """
    s0 = (s or "").lower()
    s0 = re.sub(r"[-/,\.]", " ", s0)
    s0 = re.sub(r"\s+", " ", s0).strip()

    tokens = s0.split()
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok in ("double", "doubble") and i + 1 < len(tokens) and tokens[i + 1] in NUM_WORDS:
            out.append(NUM_WORDS[tokens[i + 1]] * 2)
            i += 2
            continue

        if tok == "triple" and i + 1 < len(tokens) and tokens[i + 1] in NUM_WORDS:
            out.append(NUM_WORDS[tokens[i + 1]] * 3)
            i += 2
            continue

        if tok.isdigit():
            out.append(tok)
            i += 1
            continue

        if tok in NUM_WORDS and len(NUM_WORDS[tok]) == 1:
            out.append(NUM_WORDS[tok])
            i += 1
            continue

        i += 1

    return "".join(out)

# =========================
# 2) Field extractors
# =========================
def extract_phone(block: str) -> Optional[str]:
    """
    Robust phone extraction:
    - prefers phone cues (phone/ph/mobile/contact)
    - handles spaced digits and spoken digits
    - returns last 10 digits if >=10 found
    """
    b = (block or "")
    bl = b.lower()

    # A) explicit phone cue, digits/spaced digits nearby
    m = re.search(r"\b(?:phone|ph|mobile|contact)\b(?:\s*(?:no|number))?\s*(?:is|:)?\s*([0-9][0-9\s]{7,})", bl)
    if m:
        digits = re.sub(r"\D", "", m.group(1))
        if len(digits) >= 10:
            return digits[-10:]

    # B) explicit phone cue, spoken digits window
    m2 = re.search(r"\b(?:phone|ph|mobile|contact)\b(?:\s*(?:no|number))?\s*(?:is|:)?\s*(.{0,80})", bl)
    if m2:
        digits = spoken_to_digits(m2.group(1))
        if len(digits) >= 10:
            return digits[-10:]

    # C) any 10+ digit run anywhere (fallback)
    m3 = re.search(r"\b(\d(?:[\d\s]{8,}\d))\b", b)
    if m3:
        digits = re.sub(r"\D", "", m3.group(1))
        if len(digits) >= 10:
            return digits[-10:]

    return None

def extract_land_acres(block: str) -> Optional[float]:
    patterns = [
        r"total\s+land[^0-9]{0,30}(\d+)\s*acres?",
        r"total\s+land\s+of\s+(\d+)\s*acres?",
        r"has\s+a\s+total\s+land\s+of\s+(\d+)\s*acres?",
        r"\bland[^0-9]{0,20}(\d+)\s*acres?",
        r"land\s+each\s+(\d+)\s*acres?",
        r"each\s+(\d+)\s*acres\s+of\s+each",
    ]
    for p in patterns:
        m = re.search(p, block, re.I)
        if m:
            return float(m.group(1))
    return None

def extract_qualification(block: str) -> Optional[str]:
    b = (block or "").lower()

    m = re.search(r"(?:educational\s+)?qualification\s*(?:is\s*)?(\d{1,2})(?:st|nd|rd|th)?\b", b)
    if m:
        return f"{m.group(1)}th"

    word_map = {
        "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
        "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th"
    }

    m2 = re.search(r"(?:educational\s+)?qualification\s*(?:is\s*)?([a-z]+)\b", b)
    if m2 and m2.group(1) in word_map:
        return word_map[m2.group(1)]

    m3 = re.search(r"\b([a-z]+)\s+is\s+his\s+qualification\b", b)
    if m3 and m3.group(1) in word_map:
        return word_map[m3.group(1)]

    return None

def extract_animals(block: str) -> Optional[str]:
    animals = re.findall(r"\b(buffalo|cow|goat|sheep)\b", block, re.I)
    if not animals:
        return None
    uniq = []
    for a in animals:
        a = a.lower()
        if a not in uniq:
            uniq.append(a)
    return ", ".join([u.title() for u in uniq])

def extract_crops(block: str) -> Optional[str]:
    crop_words = ["wheat", "paddy", "rice", "sugarcane", "potato", "cauliflower", "poplar", "maize", "mustard"]
    found = []
    for cw in crop_words:
        if re.search(rf"\b{re.escape(cw)}\b", block, re.I):
            found.append(cw.title() if cw != "sugarcane" else "Sugarcane")
    return ", ".join(found) if found else None

def extract_notes(block: str) -> Optional[str]:
    notes = []
    if re.search(r"milk\s+dairy", block, re.I):
        notes.append("Runs a milk dairy")
    if re.search(r"sabzi\s+mandi", block, re.I):
        notes.append("Goes to Sabzi Mandi")
    if re.search(r"lease", block, re.I):
        notes.append("Takes land on lease")
    if re.search(r"progressive\s+farmer", block, re.I):
        notes.append("Progressive farmer")
    if re.search(r"both\s+are\s+brothers|they\s+are\s+brothers", block, re.I):
        notes.append("Brothers (with another participant)")
    if re.search(r"meeting\s+in\s+their\s+house", block, re.I):
        notes.append("Meeting held at their house")
    return "; ".join(notes) if notes else None

def extract_name(block: str) -> Optional[str]:
    b = (block or "").strip()

    # "Mr. First Last" or "Mr First Last Middle"
    m = re.search(r"\bMr\.?\s+([A-Za-z]+(?:\s+[A-Za-z]+){1,2})\b", b)
    if m:
        return clean_name(m.group(1))

    # "X Singh, phone number ..."
    m2 = re.search(r"\b([A-Za-z]+(?:\s+[A-Za-z]+){1,2})\s*,?\s*(?:phone|ph|mobile|contact)\b", b, re.I)
    if m2:
        return clean_name(m2.group(1))

    # "farmer, First Last"
    m3 = re.search(r"\bfarm(?:er)?\b\s*,?\s*([A-Za-z]+(?:\s+[A-Za-z]+){1,2})\b", b, re.I)
    if m3:
        return clean_name(m3.group(1))

    return None

# =========================
# 3) Block splitters
# =========================
def split_blocks_by_ordinals(text: str) -> List[Tuple[str, str]]:
    t = normalize_text(text)

    t = re.sub(r"\bnumber\s+one\b", "ORD1", t, flags=re.I)
    t = re.sub(r"\bnumber\s+two\b", "ORD2", t, flags=re.I)
    t = re.sub(r"\bsecond\s+one\b", "ORD2", t, flags=re.I)
    t = re.sub(r"\bthe\s+third\s+one\b", "ORD3", t, flags=re.I)
    t = re.sub(r"\bthird\s+one\b", "ORD3", t, flags=re.I)
    t = re.sub(r"\bfourth\s+one\b", "ORD4", t, flags=re.I)
    t = re.sub(r"\bfifth\s+one\b", "ORD5", t, flags=re.I)
    t = re.sub(r"\bsix\s+one\b", "ORD6", t, flags=re.I)
    t = re.sub(r"\bseventh\b", "ORD7", t, flags=re.I)
    t = re.sub(r"\beighth\b", "ORD8", t, flags=re.I)
    t = re.sub(r"\bninth\b", "ORD9", t, flags=re.I)

    m = re.search(r"\bORD[1-9]\b", t)
    if not m:
        return []
    t = t[m.start():]

    parts = re.split(r"\b(ORD[1-9])\b", t)
    blocks: List[Tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        ord_tag = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        blocks.append((ord_tag, content.strip()))
    return blocks

def _extract_anchor_spans(text: str) -> List[Dict[str, Any]]:
    t = normalize_text(text)

    # collapse spaced digits like "9 7 8 1 5 8 0 1 4 2" into "9781580142"
    t = re.sub(r"\b(?:\d\s+){9,12}\d\b", lambda m: re.sub(r"\s+", "", m.group(0)), t)

    name_re = r"([A-Za-z]+(?:\s+[A-Za-z]+){1,2})"
    phone_label = r"(?:phone|ph|mobile|contact|number)(?:\s*(?:no|number))?"
    pat = re.compile(
        rf"{name_re}\s*,?\s*{phone_label}\s*(?:is|:)?\s*([A-Za-z0-9\s]{{1,30}})",
        flags=re.I
    )

    spans = []
    for m in pat.finditer(t):
        nm = clean_name(m.group(1))
        if not nm or not looks_like_person_name(nm):
            continue
        spans.append({
            "start": m.start(),
            "end": m.end(),
            "name": nm,
            "raw_phone": m.group(2),
        })
    return spans

def split_blocks_by_anchors(text: str) -> List[Tuple[str, str]]:
    t = normalize_text(text)
    spans = _extract_anchor_spans(t)
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: x["start"])
    blocks: List[Tuple[str, str]] = []
    for i, s in enumerate(spans):
        start = s["start"]
        end = spans[i + 1]["start"] if i + 1 < len(spans) else len(t)
        block_text = t[start:end].strip()
        blocks.append((f"ORD{i + 1}", block_text))
    return blocks

def split_participant_blocks_any(text: str) -> List[Tuple[str, str]]:
    blocks = split_blocks_by_ordinals(text)
    if blocks:
        return blocks
    return split_blocks_by_anchors(text)

# =========================
# 4) LLM helpers (repo-style)
# =========================
PARTICIPANT_KEYS = ["name", "phone_number", "total_land_acres", "qualification", "animals", "main_crops", "notes"]

def _coerce_land_acres(v: Any) -> Optional[float]:
    if v is None:
        return None
    m = re.search(r"\d+(?:\.\d+)?", str(v))
    return float(m.group(0)) if m else None

def _normalize_qualification(v: Any) -> Optional[str]:
    if v is None:
        return None
    q = str(v).strip()
    if not q:
        return None
    ql = q.lower()
    word_map = {"fifth":"5th","sixth":"6th","seventh":"7th","eighth":"8th","ninth":"9th","tenth":"10th"}
    return word_map.get(ql, q)

# =========================
# 5) MAIN: rules first, then optional LLM patch
# =========================
class ParticipantExtractor(BaseLLM):
    """
    Uses the old Kaggle deterministic extractor (split + regex extractors),
    with an optional LLM patch to fill only missing fields.
    """

    # async def extract(self, entries: List[Dict]) -> Dict:
    #     transcript = format_transcript(entries)[:20000]
    #     farmers = self._extract_farmers_from_asr(transcript, use_llm_patch=True)

    #     return {
    #         "total_count": len(farmers),
    #         "farmers": farmers,
    #     }
    async def extract(self, entries: List[Dict]) -> Dict:
        # Build transcript from translated English text
        lines = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            txt = (e.get("translated_text") or "").strip()
            if txt:
                lines.append(txt)
    
        transcript = " ".join(lines)[:20000]
    
        # Optional: log first chars for debugging
        log.info("Participants: transcript (translated_text) sample: %s", transcript[:250])
    
        farmers = self._extract_farmers_from_asr(transcript, use_llm_patch=True)
    
        return {"total_count": len(farmers), "farmers": farmers}

    # ------------------------------------------------------------------

    def _extract_farmers_from_asr(self, text: str, use_llm_patch: bool = True) -> List[Dict[str, Any]]:
        text = text or ""
        blocks = split_participant_blocks_any(text)

        # If no blocks found: do a one-shot LLM extraction of ONLY farmers
        if not blocks:
            return self._llm_one_shot_farmers(text)

        rows: List[Dict[str, Any]] = []
        for ord_tag, block in blocks:
            name = extract_name(block)
            if not name:
                continue

            row = {
                "ordinal": ord_tag.replace("ORD", ""),
                "name": clean_name(name),
                "phone_number": normalize_phone(extract_phone(block)),
                "total_land_acres": extract_land_acres(block),
                "qualification": extract_qualification(block),
                "animals": extract_animals(block),
                "main_crops": extract_crops(block),
                "notes": extract_notes(block),
            }
            rows.append(row)

        # Filter junk names (critical for "names and"/"main crops"/etc.)
        rows = [r for r in rows if looks_like_person_name(r.get("name") or "")]

        # Optional LLM patch per block: fill ONLY missing fields
        if use_llm_patch and rows:
            enriched: List[Dict[str, Any]] = []
            for (ord_tag, block), row in zip(blocks, rows):
                # only patch the ones we kept
                if not looks_like_person_name(row.get("name") or ""):
                    continue

                current = {k: row.get(k) for k in PARTICIPANT_KEYS}
                filled = self._llm_patch_one_farmer(block_text=block, current=current)

                row2 = dict(row)
                row2.update(filled)

                # re-normalize
                row2["name"] = clean_name(row2.get("name"))
                row2["phone_number"] = normalize_phone(row2.get("phone_number"))
                row2["total_land_acres"] = _coerce_land_acres(row2.get("total_land_acres"))
                row2["qualification"] = _normalize_qualification(row2.get("qualification"))

                if looks_like_person_name(row2.get("name") or ""):
                    enriched.append(row2)

            return enriched

        # normalize land acres to float to match your example-like dataframe printouts
        for r in rows:
            r["total_land_acres"] = _coerce_land_acres(r.get("total_land_acres"))
            r["qualification"] = _normalize_qualification(r.get("qualification"))

        return rows

    # ------------------------------------------------------------------
    # LLM paths (repo-native: _run_inference + _safe_json)

    def _llm_one_shot_farmers(self, text: str) -> List[Dict[str, Any]]:
        if not (text or "").strip():
            return []

        prompt = f"""
Extract ONLY farmer participants from the text.

Return STRICT JSON as an object with key "farmers" holding a list of objects,
each with keys:
["ordinal","name","phone_number","total_land_acres","qualification","animals","main_crops","notes"]

Rules:
- Exclude sarpanch/coordinator/reporting manager/facilitator/staff.
- phone_number digits only or null.
- Do not invent values. If not stated, use null.
- total_land_acres must be a number (acres) or null.
Text:
\"\"\"{text}\"\"\"

Return JSON only.
""".strip()

        decoded = self._run_inference([{"role": "user", "content": prompt}], max_new_tokens=900)
        obj = self._safe_json(decoded, {"farmers": []})

        out: List[Dict[str, Any]] = []
        for i, r in enumerate(obj.get("farmers", []) or [], start=1):
            if not isinstance(r, dict):
                continue
            row = {
                "ordinal": str(r.get("ordinal") or i),
                "name": clean_name(r.get("name")),
                "phone_number": normalize_phone(r.get("phone_number")),
                "total_land_acres": _coerce_land_acres(r.get("total_land_acres")),
                "qualification": _normalize_qualification(r.get("qualification")),
                "animals": r.get("animals") if isinstance(r.get("animals"), str) else None,
                "main_crops": r.get("main_crops") if isinstance(r.get("main_crops"), str) else None,
                "notes": r.get("notes") if r.get("notes") else None,
            }
            if looks_like_person_name(row.get("name") or ""):
                out.append(row)

        return out

    def _llm_patch_one_farmer(self, block_text: str, current: Dict[str, Any]) -> Dict[str, Any]:
        missing = [k for k in PARTICIPANT_KEYS if current.get(k) is None]
        if not missing:
            return current

        prompt = f"""
You extract structured farmer participant data from noisy ASR text.

Return ONLY one JSON object with EXACTLY these keys:
{PARTICIPANT_KEYS}

Rules:
- Do NOT invent. If not stated, use null.
- name must be only the person's name (no labels like Phone/Number).
- phone_number must be digits only or null.
- If phone is spoken like "double nine ..." convert to digits.
- total_land_acres must be a number (acres) or null.
- qualification should be like "5th", "7th", "10th" if present.
- animals and main_crops should be comma-separated Title Case strings (e.g., "Cow, Buffalo", "Wheat, Paddy").
- notes: short phrases only.
- Only fill fields that are currently null. Keep existing values unchanged.

Current JSON:
{json.dumps({k: current.get(k) for k in PARTICIPANT_KEYS}, ensure_ascii=False)}

Block text:
\"\"\"{block_text}\"\"\"

Return JSON only.
""".strip()

        decoded = self._run_inference([{"role": "user", "content": prompt}], max_new_tokens=450)
        obj = self._safe_json(decoded, {k: None for k in PARTICIPANT_KEYS})

        merged = dict(current)
        for k in PARTICIPANT_KEYS:
            if merged.get(k) is None and obj.get(k) is not None:
                merged[k] = obj.get(k)

        # normalize basics (more will be normalized by caller)
        merged["name"] = clean_name(merged.get("name"))
        merged["phone_number"] = normalize_phone(merged.get("phone_number"))
        merged["total_land_acres"] = _coerce_land_acres(merged.get("total_land_acres"))
        merged["qualification"] = _normalize_qualification(merged.get("qualification"))

        return merged


# """
# Extracts participant names and roles from the transcript and structures
# them by role category.
# """        
# class ParticipantExtractor(BaseLLM):

#     async def extract(self, entries: List[Dict]) -> Dict:
#         transcript = format_transcript(entries)[:20000]
#         result     = self._extract_from_transcript(transcript)
#         return self._structure_roles(result.get("participants", []))

#     # ------------------------------------------------------------------

#     def _extract_from_transcript(self, transcript: str) -> Dict:
#         messages = [
#             {
#                 "role": "system",
#                 "content": "Extract participant names and roles from transcript.",
#             },
#             {
#                 "role": "user",
#                 "content": f"""
# Extract all real participants.

# Rules:
# - Ignore Speaker labels.
# - Do not assume farmer.
# - If unclear role → Unknown.

# Transcript:
# {transcript}

# Return JSON:
# {{
#   "participants": [
#     {{"name": "", "role": "", "village": "", "phone_number": ""}}
#   ]
# }}
# """,
#             },
#         ]
#         decoded = self._run_inference(messages, max_new_tokens=600)
#         return self._safe_json(decoded, {"participants": []})

#     def _structure_roles(self, participants: List[Dict]) -> Dict:
#         by_role: Dict[str, List] = {
#             "coordinators":       [],
#             "reporting_managers": [],
#             "sarpanch":           [],
#             "farmers":            [],
#             "other_officials":    [],
#         }
#         for p in participants:
#             role = (p.get("role") or "").lower()
#             if "coordinator" in role:
#                 by_role["coordinators"].append(p)
#             elif role in ("reporting manager", "rm"):
#                 by_role["reporting_managers"].append(p)
#             elif "sarpanch" in role:
#                 by_role["sarpanch"].append(p)
#             elif "farmer" in role:
#                 by_role["farmers"].append(p)
#             else:
#                 by_role["other_officials"].append(p)

#         return {
#             "total_count":           len(participants),
#             "participants_by_role":  by_role,
#             "detailed_participants": participants,
#         }

#     @staticmethod
#     def _validate_phone(phone: Optional[str]) -> Optional[str]:
#         if not phone:
#             return None
#         phone = re.sub(r"\D", "", phone)
#         return phone if re.match(r"^\d{10}$", phone) else None