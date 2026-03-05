"""
pipeline/asr/sarvam_asr.py

ASR + diarization + translation in one API call using Sarvam Saaras v3.

Replaces ALL of:
  - pipeline/asr/indic_conformer.py        (ASR)
  - pipeline/diarization/pyannote_diarizer.py (diarization)
  - pipeline/translation/indictrans2.py    (translation)

Workflow:
  1. Upload full combined.wav to Sarvam batch API
  2. Saaras v3 transcribes + diarizes + translates in one job
  3. Parse diarized_transcript.entries into shared transcript schema

Output schema is identical to the rest of the pipeline:
  [{"speaker_id": "SPEAKER_0", "start": 0.0, "end": 2.5,
    "original_text": "...", "translated_text": "..."}, ...]

Requires:
    pip install sarvamai
    SARVAM_API_KEY in .env
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)

# FLORES-200 → Sarvam language_code
FLORES_TO_SARVAM = {
    "pan_Guru": "pa-IN",
    "hin_Deva": "hi-IN",
    "tam_Taml": "ta-IN",
    "tel_Telu": "te-IN",
    "mar_Deva": "mr-IN",
    "kan_Knda": "kn-IN",
    "guj_Gujr": "gu-IN",
    "ben_Beng": "bn-IN",
    "mal_Mlym": "ml-IN",
    "ory_Orya": "od-IN",
}


def transcribe_and_diarize(
    audio_path:    str,
    flores_lang:   str,
    mode:          str = "translate",
    num_speakers:  int = None,
    min_duration:  float = 0.5,
) -> List[dict]:
    """
    Upload full audio to Sarvam, get back ASR + diarization + translation
    in a single API call.

    Args:
        audio_path:   Path to combined 16kHz mono WAV.
        flores_lang:  FLORES-200 language code (e.g. "pan_Guru").
        mode:         Sarvam mode — "translate" (recommended, returns English)
                      or "transcribe" (returns native language only).
        num_speakers: Expected number of speakers. None = auto-detect.
        min_duration: Skip segments shorter than this (seconds).

    Returns:
        List of transcript entry dicts with speaker_id, start, end,
        original_text, translated_text — ready for extraction stage.
    """
    from sarvamai import SarvamAI

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "SARVAM_API_KEY not found. Add it to your .env file:\n"
            "  SARVAM_API_KEY=your_key_here\n"
            "Get your key at: https://dashboard.sarvam.ai"
        )

    language_code = FLORES_TO_SARVAM.get(flores_lang, "pa-IN")
    log.info(f"Sarvam ASR: mode={mode}, language={language_code}, audio={audio_path}")

    client = SarvamAI(api_subscription_key=api_key)

    # ── Create job ───────────────────────────────────────────────────────────
    job_kwargs = dict(
        model             = "saaras:v3",
        mode              = mode,
        language_code     = language_code,
        with_diarization  = True,
    )
    if num_speakers is not None:
        job_kwargs["num_speakers"] = num_speakers

    job = client.speech_to_text_job.create_job(**job_kwargs)

    # ── Upload, start, wait ──────────────────────────────────────────────────
    log.info("  Uploading audio...")
    job.upload_files(file_paths=[audio_path])
    job.start()

    log.info("  Waiting for Sarvam job to complete...")
    job.wait_until_complete()

    # ── Check results ────────────────────────────────────────────────────────
    file_results = job.get_file_results()

    if not file_results["successful"]:
        error = file_results["failed"][0].get("error_message", "unknown error")
        raise RuntimeError(f"Sarvam job failed: {error}")

    log.info(f"  Job complete. Downloading output...")

    # ── Download and parse output ────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as out_dir:
        job.download_outputs(output_dir=out_dir)
        result_files = list(Path(out_dir).glob("*.json"))

        if not result_files:
            raise RuntimeError("Sarvam returned no output JSON.")

        with open(result_files[0], encoding="utf-8") as f:
            data = json.load(f)

    return _parse_output(data, mode, min_duration)


def _parse_output(
    data:         dict,
    mode:         str,
    min_duration: float,
) -> List[dict]:
    """
    Parse Sarvam response JSON into our shared transcript schema.

    Sarvam diarized_transcript.entries shape:
      {"transcript": str, "start_time_seconds": float,
       "end_time_seconds": float, "speaker_id": str}

    Our schema:
      {"speaker_id": str, "start": float, "end": float,
       "original_text": str, "translated_text": str}
    """
    diarized = data.get("diarized_transcript", {})
    raw_entries = diarized.get("entries", [])

    if not raw_entries:
        # Fallback: no diarization in response — wrap full transcript as one entry
        log.warning("No diarized entries in response — wrapping full transcript as single entry.")
        transcript = data.get("transcript", "").strip()
        return [{
            "speaker_id":      "SPEAKER_0",
            "start":           0.0,
            "end":             0.0,
            "original_text":   transcript,
            "translated_text": transcript if mode == "translate" else "",
        }]

    entries = []
    for raw in raw_entries:
        start    = float(raw.get("start_time_seconds", 0.0))
        end      = float(raw.get("end_time_seconds",   0.0))
        duration = end - start

        if duration < min_duration:
            log.debug(f"  Skipping short segment ({duration:.2f}s)")
            continue

        transcript = raw.get("transcript", "").strip()
        if not transcript:
            continue

        # Sarvam speaker_id is "0", "1" etc — normalise to "SPEAKER_0"
        speaker_id = f"SPEAKER_{raw.get('speaker_id', '0')}"

        entries.append({
            "speaker_id":      speaker_id,
            "start":           round(start, 3),
            "end":             round(end,   3),
            "original_text":   transcript,
            # mode="translate" → Sarvam returns English directly
            # mode="transcribe" → translated_text left empty for a separate step
            "translated_text": transcript if mode == "translate" else "",
        })

    log.info(f"  Parsed {len(entries)} segments from Sarvam response.")
    return entries