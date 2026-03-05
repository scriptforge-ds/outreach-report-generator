"""
NarrationGenerator  — concise summary + detailed narrative from the transcript.
SummaryGenerator    — formal 3-paragraph prose report from structured extractor outputs.

Both inherit BaseLLM. NarrationGenerator can optionally share an already-loaded
model to avoid double-loading GPU weights.
"""

import logging
from typing import Dict, List, Optional

from pipeline.extraction.base_llm import BaseLLM

log = logging.getLogger(__name__)


# =============================================================================
# NARRATION GENERATOR
# =============================================================================

class NarrationGenerator(BaseLLM):

    def __init__(self, base: Optional[BaseLLM] = None, device: Optional[str] = None):
        if base is not None:
            # Share already-loaded weights — skip __init__ model load
            self.model     = base.model
            self.tokenizer = base.tokenizer
            self.device    = base.device
        else:
            super().__init__(device=device)

    def generate(self, entries: List[Dict], max_chars: int = 20000) -> Dict:
        transcript = self._get_initial_transcript(entries, max_chars)
        log.info(f"NarrationGenerator: {len(transcript)} chars of transcript.")
        return {
            "summary":   self._generate_summary(transcript),
            "narration": self._generate_narration(transcript),
        }

    def _get_initial_transcript(self, entries: List[Dict], max_chars: int) -> str:
        lines, total = [], 0
        for e in entries:
            text = e.get("original_text", "").strip()
            if not text:
                continue
            if total + len(text) > max_chars:
                break
            lines.append(text)
            total += len(text)
        return "\n".join(lines)

    def _generate_summary(self, transcript: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert agricultural extension officer summarizing "
                    "farmer meeting transcripts."
                ),
            },
            {
                "role": "user",
                "content": f"""
Read the following agricultural meeting transcript and write a concise summary
in 2-3 sentences covering the main topic, key crops/diseases, and any decisions made.
Return ONLY the summary, no headings.

Transcript:
{transcript}
""",
            },
        ]
        return self._run_inference(messages, max_new_tokens=300).strip()

    def _generate_narration(self, transcript: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert agricultural extension officer writing "
                    "detailed meeting narrations from farmer transcripts."
                ),
            },
            {
                "role": "user",
                "content": f"""
Write a detailed, professional 3-5 paragraph narration (no bullet points) covering
all major topics, farmer problems, advice given, specific crops/diseases, and any
recommendations. Return ONLY the narration, no headings.

Transcript:
{transcript}
""",
            },
        ]
        return self._run_inference(messages, max_new_tokens=1200).strip()


# =============================================================================
# SUMMARY GENERATOR
# =============================================================================

class SummaryGenerator(BaseLLM):

    async def generate(
        self,
        participants: Dict,
        challenges:   List[Dict],
        questions:    List[str],
        narration:    str,
    ) -> str:
        context = self._build_context(participants, challenges, questions, narration)
        return self._generate_summary(context)

    def _build_context(
        self,
        participants: Dict,
        challenges:   List[Dict],
        questions:    List[str],
        narration:    str,
    ) -> str:
        total        = participants.get("total_count", 0)
        roles        = participants.get("participants_by_role", {})
        farmers      = len(roles.get("farmers", []))
        coordinators = len(roles.get("coordinators", []))

        challenge_lines = [
            f"{g.get('category', '')}: {', '.join(g.get('challenges', [])[:3])}"
            for g in challenges
        ]

        return f"""
MEETING SUMMARY CONTEXT:

Participants: {total} total ({farmers} farmers, {coordinators} coordinators)

Top Challenges:
{chr(10).join(challenge_lines)}

Farmer Questions (sample):
{chr(10).join(questions[:10])}

Meeting Narration:
{narration[:3000]}
"""

    def _generate_summary(self, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You write professional meeting summary reports for agricultural "
                    "outreach programs in rural India."
                ),
            },
            {
                "role": "user",
                "content": f"""
Based on the following meeting data, write a 2-3 paragraph summary.

RULES:
- Paragraph 1: Who attended and the overall purpose of the meeting.
- Paragraph 2: Key challenges and problems raised by farmers.
- Paragraph 3: Overall tone, farmer sentiment, and any notable outcomes or requests.
- Write in formal English.
- Do NOT use bullet points.
- Do NOT invent facts not present in the context.

{context}
""",
            },
        ]
        return self._run_inference(messages, max_new_tokens=900)