"""
ConclusionGenerator — generates a formal meeting conclusion (not a summary).

Uses BaseLLM (Ministral) to write a concise concluding section for the report,
focused on outcomes, takeaways, and next steps strictly grounded in the context.
"""

from __future__ import annotations

from typing import Dict, List
import logging

from pipeline.extraction.base_llm import BaseLLM

log = logging.getLogger(__name__)


class ConclusionGenerator(BaseLLM):

    async def generate(
        self,
        participants: Dict,
        challenges:   List[Dict],
        questions:    List[str],
        narration:    str,
    ) -> str:
        context = self._build_context(participants, challenges, questions, narration)
        return self._generate_conclusion(context)

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
MEETING CONCLUSION CONTEXT:

Participants: {total} total ({farmers} farmers, {coordinators} coordinators)

Key Challenges Raised:
{chr(10).join(challenge_lines)}

Farmer Questions (sample):
{chr(10).join(questions[:10])}

Meeting Narration (excerpt):
{narration[:3000]}
""".strip()

    def _generate_conclusion(self, context: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You write professional concluding sections for agricultural outreach "
                    "meeting reports in rural India. Your writing is factual, concise, and "
                    "strictly based on the provided context."
                ),
            },
            {
                "role": "user",
                "content": f"""
Write the MEETING CONCLUSION (not a summary) based on the context below.

RULES:
- Output 1–2 concise paragraphs (no bullet points).
- Focus on: overall closure of the meeting, key takeaways, and any stated or implied next steps.
- If decisions, commitments, timelines, or follow-ups are NOT explicitly present in the context, do NOT invent them.
- Do not repeat the full participant breakdown; mention attendance only briefly if relevant for closure.
- Use formal, report-ready English.

{context}
""".strip(),
            },
        ]

        return self._run_inference(messages, max_new_tokens=450)