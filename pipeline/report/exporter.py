"""
pipeline/report/exporter.py

Generates a formatted PDF report from the assembled report dict.

Expected report dict shape (from pipeline/report/assembler.py):
{
    "summary":          str,
    "narration":        {"summary": str, "narration": str},
    "terminology":      [{"Crop": "", "Local Name": "", "Standard Name": "",
                          "Scientific Name": "", "Language": ""}, ...],
    "farmer_questions": [str, ...],
    "challenges":       [{"category": str, "challenges": [str, ...]}, ...],
    "participants": {
        "total_count":           int,
        "participants_by_role":  {...},
        "detailed_participants": [{"name": "", "role": "", "village": "",
                                   "phone_number": ""}, ...]
    },
    "metadata": {...}   ← optional, populated by zoho_client / caller
}

Usage:
    from pipeline.report.exporter import PDFReportGenerator
    gen = PDFReportGenerator()
    gen.create_report(report, "./outputs/report.pdf")

Dependencies:
    pip install reportlab pandas
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    LongTable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# =============================================================================
# HELPERS
# =============================================================================

def safe(val: Any) -> str:
    """Return a clean string or empty string for None / 'None' values."""
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() == "none" else s


def strip_markdown(text: str) -> str:
    """Remove common markdown symbols so they don't bleed into PDF text."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # bold
    text = re.sub(r"\*(.*?)\*",     r"\1", text)   # italic
    text = re.sub(r"^#{1,6}\s+",    "",    text, flags=re.MULTILINE)  # headings
    text = re.sub(r"`(.*?)`",       r"\1", text)   # inline code
    text = re.sub(r"^\s*[-*]\s+",   "",    text, flags=re.MULTILINE)  # bullets
    return text.strip()


def _escape_for_para(text: str) -> str:
    """Escape characters that break ReportLab Paragraph XML parsing."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def normalize_time(val: str) -> str:
    """Normalise time strings to HH:MM format where possible."""
    val = safe(val)
    match = re.search(r"(\d{1,2}):(\d{2})", val)
    if match:
        return f"{int(match.group(1)):02d}:{match.group(2)}"
    return val


def extract_parenthetical(val: Any) -> str:
    """
    Some LLM outputs wrap values in parentheses or quotes — strip them.
    e.g. "(Ludhiana)" → "Ludhiana"
    """
    s = safe(val)
    match = re.match(r"^\(?(.*?)\)?$", s)
    return match.group(1).strip() if match else s


def coerce_list(val: Any) -> List:
    """Ensure value is a list; wrap single items, return [] for None."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        # e.g. {"items": [...]}
        for k in ("items", "rows", "data", "list"):
            if k in val and isinstance(val[k], list):
                return val[k]
        return [val]
    return [val]


def participants_to_df(participants: Any) -> pd.DataFrame:
    """
    Convert participants output to a DataFrame.

    Supported shapes:
      - New farmer-only shape:
          {"total_count": int, "farmers": [ {...}, ... ]}
      - Old role-based shape:
          {"detailed_participants": [ {...}, ... ], "total_count": int, ...}
      - A plain list of dicts
    """
    if participants is None:
        return pd.DataFrame()

    rows: Any = None

    if isinstance(participants, dict):
        if isinstance(participants.get("farmers"), list):
            rows = participants.get("farmers", [])
        elif isinstance(participants.get("detailed_participants"), list):
            rows = participants.get("detailed_participants", [])
        elif isinstance(participants.get("participants"), list):
            rows = participants.get("participants", [])
        else:
            rows = []
    elif isinstance(participants, list):
        rows = participants
    else:
        rows = []

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Be defensive about column names
    if "total_land_acre" in df.columns and "total_land_acres" not in df.columns:
        df = df.rename(columns={"total_land_acre": "total_land_acres"})

    return df


_SENT_END = re.compile(r"(?s)(.*?[.!?])(\s+|$)")

def paragraphize_long_text(text: str, style: ParagraphStyle, *, max_chunk_chars: int = 1400) -> List[Paragraph]:
    """
    - Keeps original paragraph breaks (\n\n)
    - Only chunks very long paragraphs
    - When chunking, prefers sentence boundaries so you don't get 'A' / half sentences.
    """
    text = safe(text)
    if not text:
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    out: List[Paragraph] = []

    for block in blocks:
        # chunk only if a single paragraph is huge
        while len(block) > max_chunk_chars:
            window = block[:max_chunk_chars]

            # Prefer last sentence ending inside the window
            cut = None
            matches = list(_SENT_END.finditer(window))
            if matches:
                cut = matches[-1].end(1)

            # Fallback: last space
            if not cut or cut < 300:
                sp = window.rfind(" ")
                cut = sp if sp > 300 else max_chunk_chars

            part = block[:cut].strip()
            if part:
                out.append(Paragraph(_escape_for_para(part), style))
            block = block[cut:].strip()

        if block:
            out.append(Paragraph(_escape_for_para(block), style))

    return out
# =============================================================================
# PDF REPORT GENERATOR
# =============================================================================

class PDFReportGenerator:
    """Generate a formatted A4 PDF report from the assembled report dict."""

    def __init__(self, title: str = "Project Ajrasakha - Farmers Outreach Project"):
        self.title = title
    
        # Prefer NotoSans (better Unicode coverage than FreeSans)
        regular = Path("assets/fonts/NotoSans-Regular.ttf")
        bold    = Path("assets/fonts/NotoSans-Bold.ttf")
    
        try:
            if regular.exists():
                pdfmetrics.registerFont(TTFont("NotoSans", str(regular)))
                self.font = "NotoSans"
            else:
                # fallback to FreeSans 
                fs = Path("assets/fonts/FreeSans.ttf")
                if fs.exists():
                    pdfmetrics.registerFont(TTFont("FreeSans", str(fs)))
                    self.font = "FreeSans"
                else:
                    self.font = "Helvetica"
        except Exception:
            self.font = "Helvetica"
    
        # Bold font registration
        try:
            if bold.exists():
                pdfmetrics.registerFont(TTFont("NotoSans-Bold", str(bold)))
                self.header_font = "NotoSans-Bold"
            else:
                # if no bold file, at least use same font for headers
                self.header_font = self.font
        except Exception:
            self.header_font = "Helvetica-Bold"

    # ------------------------------------------------------------------
    # LAYOUT PRIMITIVES
    # ------------------------------------------------------------------

    def _boxed_header(self, text: str, width: float = 7.3 * inch) -> Table:
        return Table(
            [[text]],
            colWidths=[width],
            style=TableStyle([
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, -1), colors.lightgrey),
                ("FONTNAME",   (0, 0), (-1, -1), self.header_font),
                ("FONTSIZE",   (0, 0), (-1, -1), 10),
                ("PADDING",    (0, 0), (-1, -1), 6),
            ]),
        )

    def _boxed_body(self, flowables: List, width: float = 7.3 * inch) -> Table:
        return Table(
            [[flowables]],
            colWidths=[width],
            style=TableStyle([
                ("GRID",    (0, 0), (-1, -1), 0.5, colors.black),
                ("VALIGN",  (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]),
        )

    def _boxed_body_splittable(
        self, flowables: List, width: float = 7.3 * inch
    ) -> LongTable:
        """Boxed container that splits across pages (one flowable per row)."""
        if not flowables:
            # flowables = [Paragraph("No data available.", getSampleStyleSheet()["BodyText"])]
            fallback_style = ParagraphStyle(
                "FallbackBody",
                parent=getSampleStyleSheet()["BodyText"],
                fontName=self.font,
                fontSize=10,
                leading=12,
            )
            flowables = [Paragraph("No data available.", fallback_style)]

        # t = LongTable([[f] for f in flowables], colWidths=[width], splitByRow=1)
        # t.setStyle(TableStyle([
        #     ("BOX",            (0, 0), (-1, -1), 0.5, colors.black),
        #     ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        #     ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        #     ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
        #     ("TOPPADDING",     (0, 0), (-1, -1), 3),
        #     ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
        # ]))
        t = LongTable([[f] for f in flowables], colWidths=[width], splitByRow=1)
        t.setStyle(TableStyle([
            # vertical borders always
            ("LINEBEFORE",     (0, 0), (0, -1), 0.5, colors.black),
            ("LINEAFTER",      (-1, 0), (-1, -1), 0.5, colors.black),
        
            # IMPORTANT: draw top border at start of EACH split fragment
            ("LINEABOVE",      (0, 0), (-1, 0), 0.5, colors.black),
        
            # IMPORTANT: draw bottom border at end of EACH split fragment
            ("LINEBELOW",      (0, -1), (-1, -1), 0.5, colors.black),
        
            ("VALIGN",         (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",    (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
            ("TOPPADDING",     (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
        ]))
        return t

    def _make_bullets(
        self, items: List[Any], body_style: ParagraphStyle
    ) -> List[Any]:
        """Render a list of strings or dicts as bullet Paragraphs."""
        if not items:
            return [Paragraph("No data available.", body_style)]

        out = []
        preferred_keys = ["challenge", "issue", "text", "question", "crop", "details", "notes"]

        for it in items:
            if isinstance(it, dict):
                parts = [
                    f"{k}: {it[k]}"
                    for k in preferred_keys
                    if k in it and it[k] not in (None, "", "None")
                ]
                s = " | ".join(parts) if parts else json.dumps(it, ensure_ascii=False)
            else:
                s = str(it)

            s = _escape_for_para(strip_markdown(s))
            out.append(Paragraph(f"• {s}", body_style))

        return out

    def _table_from_rows(
        self,
        rows: List[Dict[str, Any]],
        columns: List[str],
        doc: SimpleDocTemplate,
        styles,
        *,
        header_title_map: Optional[Dict[str, str]] = None,
    ) -> LongTable:
        """Build a wrapped, page-splittable table using LongTable + Paragraph cells."""
        body_style = ParagraphStyle(
            "Cell",
            parent=styles["BodyText"],
            fontName=self.font,
            fontSize=9,
            leading=11,
            alignment=TA_LEFT,
            spaceBefore=0,
            spaceAfter=0,
        )
        hdr_style = ParagraphStyle(
            "HeaderCell",
            parent=body_style,
            fontName=self.header_font,
        )

        header_title_map = header_title_map or {}
        header_row = [
            Paragraph(_escape_for_para(header_title_map.get(c, c)), hdr_style)
            for c in columns
        ]
        data = [header_row]

        for r in rows:
            row = []
            for c in columns:
                v = r.get(c, "")
                if v is None:
                    v = ""
                if isinstance(v, (list, dict)):
                    v = json.dumps(v, ensure_ascii=False)
                s = str(v).replace("\n", " ").strip()
                s = "" if s.lower() == "none" else s
                row.append(Paragraph(_escape_for_para(s), body_style))
            data.append(row)

        usable_width = A4[0] - doc.leftMargin - doc.rightMargin
        heavy = {"question", "symptoms", "notes", "scientific_name", "standard_name",
                 "Scientific Name", "Standard Name"}
        weights    = [2.2 if c in heavy else 1.0 for c in columns]
        col_widths = [usable_width * (w / sum(weights)) for w in weights]

        t = LongTable(data, colWidths=col_widths, repeatRows=1, splitByRow=1)
        t.setStyle(TableStyle([
            ("GRID",           (0, 0), (-1, -1), 0.5, colors.black),
            ("BOX",            (0, 0), (-1, -1), 0.8, colors.black),
            ("BACKGROUND",     (0, 0), (-1,  0), colors.lightgrey),
            ("VALIGN",         (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",    (0, 0), (-1, -1), 4),
            ("RIGHTPADDING",   (0, 0), (-1, -1), 4),
            ("TOPPADDING",     (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 3),
        ]))
        return t

    # ------------------------------------------------------------------
    # MAIN ENTRY
    # ------------------------------------------------------------------

    def create_report(
        self,
        data: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> Path:
        """
        Build the PDF from the assembled report dict and write to output_path.

        Args:
            data:        Report dict from pipeline/report/assembler.assemble()
            output_path: Destination .pdf path

        Returns:
            Path to the written PDF file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=0.5 * inch,
            rightMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
        )
        styles    = getSampleStyleSheet()
        body_style = ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName=self.font,
            leading=14,
            fontSize=10,
            spaceAfter=6,
        )

        # ── Unpack report dict ───────────────────────────────────────────────

        meta: Dict[str, Any] = data.get("metadata") or {}

        # narration is {"summary": str, "narration": str} in our schema
        narration_block = data.get("narration") or {}
        if isinstance(narration_block, str):
            narration_text = narration_block
            summary_text   = safe(data.get("summary", ""))
        else:
            narration_text = safe(narration_block.get("narration", ""))
            summary_text   = safe(
                data.get("summary")
                or narration_block.get("summary", "")
            )

        # # challenges → flatten category groups into bullet strings
        # key_challenges = flatten_challenges(data.get("challenges", []))

        # challenges → keep grouped by category for table rendering
        key_challenges_groups = coerce_list(data.get("challenges", []))

        # farmer_questions → list of strings
        questions = coerce_list(data.get("farmer_questions", []))

        # terminology → list of dicts with our schema keys
        terminology_rows = coerce_list(data.get("terminology", []))

        # participants → DataFrame from detailed_participants
        participants_df = participants_to_df(data.get("participants"))

        # participant counts
        participants_meta = data.get("participants") or {}
        total  = safe(meta.get("farmers_attended_total") or participants_meta.get("total_count"))
        female = safe(meta.get("female_farmers_count", ""))
        male   = safe(meta.get("male_farmers_count", ""))
        if not total and len(participants_df) > 0:
            total = str(len(participants_df))

        # ── Build story ──────────────────────────────────────────────────────

        # story = [
        #     Paragraph(self.title, styles["Title"]),
        #     Spacer(1, 0.2 * inch),
        # ]
        title_style = ParagraphStyle(
            "MyTitle",
            parent=styles["Title"],
            fontName=self.header_font,
        )
        story = [
            Paragraph(_escape_for_para(self.title), title_style),
            Spacer(1, 0.2 * inch),
        ]

        # ── Metadata table ──────────────────────────────────────────────────
        #   - columns sized to fit the usable page width
        #   - Paragraph cells so text wraps (no overflow)
        #   - splitByRow so the table can break across pages cleanly
        key_style = ParagraphStyle(
            "MetaKey",
            parent=styles["BodyText"],
            fontName=self.header_font,
            fontSize=9,
            leading=11,
            alignment=TA_LEFT,
            spaceBefore=0,
            spaceAfter=0,
        )
        val_style = ParagraphStyle(
            "MetaVal",
            parent=styles["BodyText"],
            fontName=self.font,
            fontSize=9,
            leading=11,
            alignment=TA_LEFT,
            spaceBefore=0,
            spaceAfter=0,
        )

        def _p(s: Any, style: ParagraphStyle) -> Paragraph:
            return Paragraph(_escape_for_para(safe(s)), style)

        meta_rows = [
            ["Date",                         safe(meta.get("date", "")),
             "Day",                          safe(meta.get("day", ""))],
            ["Village",                      extract_parenthetical(meta.get("village")),
             "Name of the Sarpanch",         extract_parenthetical(meta.get("sarpanch_name"))],
            ["Panchayat",                    extract_parenthetical(meta.get("panchayat")),
             "Phone Number",                 safe(meta.get("phone_number", ""))],
            ["Block",                        extract_parenthetical(meta.get("block")),
             "Event Location",               extract_parenthetical(meta.get("event_location"))],
            ["District",                     extract_parenthetical(meta.get("district")),
             "No of Farmers attended",       total],
            ["Name of the Coordinator",      extract_parenthetical(meta.get("coordinator_name")),
             "Female Farmers",               female],
            ["Name of the Reporting Manager",extract_parenthetical(meta.get("reporting_manager_name")),
             "Male Farmers",                 male],
            ["Event Start Time",             normalize_time(safe(meta.get("event_start_time", ""))),
             "Event End Time",               normalize_time(safe(meta.get("event_end_time", "")))],
        ]

        table_data = []
        for r in meta_rows:
            table_data.append([
                _p(r[0], key_style), _p(r[1], val_style),
                _p(r[2], key_style), _p(r[3], val_style),
            ])

        usable_width = A4[0] - doc.leftMargin - doc.rightMargin
        # value columns narrower than key columns
        weights = [1.4, 1.2, 1.4, 1.2]
        col_widths = [usable_width * (w / sum(weights)) for w in weights]

        meta_tbl = LongTable(
            table_data,
            colWidths=col_widths,
            splitByRow=1,
        )
        meta_tbl.setStyle(TableStyle([
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.black),
            ("BOX",           (0, 0), (-1, -1), 0.8, colors.black),
            ("BACKGROUND",    (0, 0), (0, -1), colors.whitesmoke),
            ("BACKGROUND",    (2, 0), (2, -1), colors.whitesmoke),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(meta_tbl)
        story.append(Spacer(1, 0.2 * inch))


        # ── Detailed Narration ───────────────────────────────────────────────

        story.append(self._boxed_header("Detailed Narration of the Interaction:"))
        
        narration_content = []
        
        if narration_text:
            narration_content.append(Paragraph("<b>Translation / Dictation:</b>", styles["Heading4"]))
        
            # IMPORTANT: build many Paragraphs (many rows) so the box can split
            narr_clean = strip_markdown(narration_text)
            narration_content.extend(paragraphize_long_text(narr_clean, body_style, max_chunk_chars=900))
        else:
            narration_content.append(Paragraph("No narration available.", body_style))
        
        if summary_text:
            narration_content.append(Spacer(1, 8))
            narration_content.append(Paragraph("<b>Summary:</b>", styles["Heading4"]))
            narration_content.extend(paragraphize_long_text(strip_markdown(summary_text), body_style, max_chunk_chars=900))
        
        story.append(self._boxed_body_splittable(narration_content))
        story.append(Spacer(1, 0.2 * inch))
        # ── Key Challenges ───────────────────────────────────────────────────

        # story.append(self._boxed_header("Key Challenges Shared by the farmers:"))
        # story.append(
        #     self._boxed_body_splittable(self._make_bullets(key_challenges, body_style))
        # )
        # story.append(Spacer(1, 0.2 * inch))
        # ── Key Challenges ───────────────────────────────────────────────────
        
        story.append(self._boxed_header("Key Challenges Shared by the farmers:"))
        
        # Render as a 2-column table: Category | Challenges (wrapped, page-fit width)
        groups = [g for g in key_challenges_groups if isinstance(g, dict)]
        if groups:
            cat_style = ParagraphStyle(
                "ChCat",
                parent=styles["BodyText"],
                fontName=self.header_font,
                fontSize=9,
                leading=11,
                alignment=TA_LEFT,
                spaceBefore=0,
                spaceAfter=0,
            )
            chal_style = ParagraphStyle(
                "ChTxt",
                parent=styles["BodyText"],
                fontName=self.font,
                fontSize=9,
                leading=11,
                alignment=TA_LEFT,
                spaceBefore=0,
                spaceAfter=0,
            )
        
            def _p(s: Any, style: ParagraphStyle) -> Paragraph:
                return Paragraph(_escape_for_para(safe(s)), style)
        
            def _bullets(items: Any) -> str:
                items = coerce_list(items)
                cleaned = [safe(x) for x in items if safe(x)]
                if not cleaned:
                    return ""
                # HTML bullets so they wrap inside a Paragraph
                return "<br/>".join([f"&bull; { _escape_for_para(c) }" for c in cleaned])
        
            table_data = [[
                Paragraph(_escape_for_para("Category"), styles["BodyText"]),
                Paragraph(_escape_for_para("Challenges"), styles["BodyText"]),
            ]]
        
            for g in groups:
                category = safe(g.get("category", ""))
                challenges_html = _bullets(g.get("challenges", []))
                table_data.append([
                    _p(category, cat_style),
                    Paragraph(challenges_html or _escape_for_para(""), chal_style),
                ])
        
            usable_width = A4[0] - doc.leftMargin - doc.rightMargin
            weights = [1.0, 3.0]
            col_widths = [usable_width * (w / sum(weights)) for w in weights]
        
            chal_tbl = LongTable(
                table_data,
                colWidths=col_widths,
                repeatRows=1,
                splitByRow=1,
            )
        
            # Alternate background by category row (row 1..n; row 0 is header)
            bg_x = colors.HexColor("#F7F7F7")
            bg_y = colors.HexColor("#EFEFEF")
            bg_cmds = []
            for i in range(1, len(table_data)):
                bg = bg_x if (i % 2 == 1) else bg_y
                bg_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
        
            chal_tbl.setStyle(TableStyle([
                ("GRID",          (0, 0), (-1, -1), 0.5, colors.black),
                ("BOX",           (0, 0), (-1, -1), 0.8, colors.black),
                ("BACKGROUND",    (0, 0), (-1, 0), colors.lightgrey),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 4),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ] + bg_cmds))
        
            story.append(chal_tbl)
        else:
            # Fallback: if input is already a list of strings
            story.append(self._boxed_body_splittable(self._make_bullets(key_challenges_groups, body_style)))
        
        story.append(Spacer(1, 0.2 * inch))

        # ── Farmer Questions ─────────────────────────────────────────────────
        usable_width = A4[0] - doc.leftMargin - doc.rightMargin
        story.append(self._boxed_header("Questions asked by farmers:", width=usable_width))
        structured_q = [q for q in questions if isinstance(q, dict)]

        if structured_q:
            common_cols = ["question", "asked_by", "crop", "timestamp", "notes"]
            cols = [
                c for c in common_cols
                if any(c in r and r.get(c) not in (None, "", "None") for r in structured_q)
            ] or sorted({k for r in structured_q for k in r.keys()})[:6]

            story.append(self._table_from_rows(
                structured_q, cols, doc, styles,
                header_title_map={"question": "Question", "asked_by": "Asked By",
                                  "timestamp": "Time"},
            ))
        else:
            story.append(
                self._boxed_body_splittable(self._make_bullets(questions, body_style), width=usable_width)
            )
        story.append(Spacer(1, 0.2 * inch))

        # story.append(self._boxed_header("Questions asked by farmers:"))
        # structured_q = [q for q in questions if isinstance(q, dict)]

        # if structured_q:
        #     common_cols = ["question", "asked_by", "crop", "timestamp", "notes"]
        #     cols = [
        #         c for c in common_cols
        #         if any(c in r and r.get(c) not in (None, "", "None") for r in structured_q)
        #     ] or sorted({k for r in structured_q for k in r.keys()})[:6]

        #     story.append(self._table_from_rows(
        #         structured_q, cols, doc, styles,
        #         header_title_map={"question": "Question", "asked_by": "Asked By",
        #                           "timestamp": "Time"},
        #     ))
        # else:
        #     story.append(
        #         self._boxed_body_splittable(self._make_bullets(questions, body_style))
        #     )
        # story.append(Spacer(1, 0.2 * inch))

        # ── Terminology Mapping ──────────────────────────────────────────────
        # Our schema keys: Crop, Local Name, Standard Name, Scientific Name, Language

        story.append(self._boxed_header(
            "Crop-wise Disease Terminology Mapping (Dialect to Scientific):"
        ))
        valid_rows = [r for r in terminology_rows if isinstance(r, dict)]

        if valid_rows:
            preferred_cols = ["Crop", "Local Name", "Standard Name", "Scientific Name", "Language"]
            cols = [
                c for c in preferred_cols
                if any(c in r and r.get(c) not in (None, "", "None") for r in valid_rows)
            ] or sorted({k for r in valid_rows for k in r.keys()})[:6]

            story.append(self._table_from_rows(
                valid_rows, cols, doc, styles,
                header_title_map={
                    "Local Name":      "Local / Dialect Name",
                    "Standard Name":   "Standard Name",
                    "Scientific Name": "Scientific Name",
                },
            ))
        else:
            story.append(self._boxed_body_splittable(
                [Paragraph("No terminology mapping data available.", body_style)]
            ))
        story.append(Spacer(1, 0.2 * inch))

        # ── Participants Table ───────────────────────────────────────────────

        story.append(Paragraph("Participants Details (Farmers)", styles["Heading2"]))
        story.append(Spacer(1, 6))

        wanted_cols  = ["name", "phone_number", "total_land_acres", "qualification", "animals", "main_crops", "notes"]
        present_cols = [c for c in wanted_cols if c in participants_df.columns]
        
        if participants_df.empty or not present_cols:
            story.append(Paragraph("No farmer participant details available.", body_style))
        else:
            present_cols = [c for c in wanted_cols if c in participants_df.columns]
        
            cell_style = ParagraphStyle(
                "CellP",
                parent=styles["BodyText"],
                fontName=self.font,
                fontSize=10,
                leading=12,
                alignment=TA_LEFT,
                spaceBefore=0,
                spaceAfter=0,
            )
            hdr_style = ParagraphStyle(
                "HeaderCellP",
                parent=cell_style,
                fontName=self.header_font,
            )
        
            def _fmt_cell(v: Any) -> str:
                if v is None:
                    return ""
                try:
                    if pd.isna(v):
                        return ""
                except Exception:
                    pass
                s = str(v).replace("\n", " ").strip()
                if s.lower() in ("none", "null", "nan"):
                    return ""
                return s
        
            p_data = [[Paragraph(_escape_for_para(c), hdr_style) for c in present_cols]]
        
            # Sort by ordinal if possible
            df_show = participants_df.copy()
            if "ordinal" in df_show.columns:
                try:
                    df_show["_ord_int"] = df_show["ordinal"].apply(
                        lambda x: int(re.search(r"[0-9]+", str(x)).group(0)) if re.search(r"[0-9]+", str(x)) else 10**9
                    )
                    df_show = df_show.sort_values("_ord_int")
                except Exception:
                    pass
        
            for _, row in df_show[present_cols].iterrows():
                p_data.append([
                    Paragraph(_escape_for_para(_fmt_cell(row.get(c))), cell_style)
                    for c in present_cols
                ])
        
            col_widths_map = {
                "name":             110,
                "phone_number":      75,
                "total_land_acres":  55,
                "qualification":     55,
                "animals":           75,
                "main_crops":       110,
                "notes":            120,
            }
        
            usable_width = A4[0] - doc.leftMargin - doc.rightMargin
            widths       = [col_widths_map.get(c, 80) for c in present_cols]
            total_w      = sum(widths)
            if total_w > usable_width:
                scale  = usable_width / total_w
                widths = [w * scale for w in widths]

            p_tbl = Table(p_data, colWidths=widths, repeatRows=1, splitByRow=1)
            p_tbl.setStyle(TableStyle([
                ("GRID",          (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND",    (0, 0), (-1,  0), colors.lightgrey),
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 4),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(p_tbl)

        story.append(Spacer(1, 0.2 * inch))

        # ── Conclusion ───────────────────────────────────────────────────────

        conclusion = safe(data.get("conclusion", "")) or summary_text
        if conclusion:
            story.append(self._boxed_header("Conclusion"))
            # story.append(self._boxed_body_splittable([
            #     Paragraph(_escape_for_para(strip_markdown(conclusion)), body_style)
            # ]))
            body_paras = paragraphize_long_text(strip_markdown(conclusion), body_style, max_chunk_chars=900) or \
                         [Paragraph("No conclusion available.", body_style)]
            story.append(self._boxed_body_splittable(body_paras))

        doc.build(story)
        return output_path