"""
main.py

Single entry point for the full outreach report pipeline:

  audio files (directory)
      → combine & normalize    [pipeline/ingestion/audio_utils.py]
      → ASR + diarization      [pipeline/asr/sarvam_asr.py]
      → structured transcript  [pipeline/transcript/builder.py]
      → translation            [pipeline/translation/indictrans2.py]
      → extraction & insights  [pipeline/extraction/*]
      → assemble & save        [pipeline/report/assembler.py]

Usage:
    python main.py --input_dir ./audio --language pa
    python main.py --input_dir ./audio --language hi --output_dir ./outputs/meeting_001
    python main.py --input_dir ./audio --language pa --skip_asr
    python main.py --input_dir ./audio --language pa --skip_asr --skip_translation

Language codes:
    pa  → Punjabi    hi  → Hindi      ta  → Tamil
    te  → Telugu     mr  → Marathi    kn  → Kannada
    gu  → Gujarati   bn  → Bengali    or  → Odia
    ml  → Malayalam
"""

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

# Load .env before any pipeline module is imported so all os.getenv() calls
# across the codebase see the variables without needing their own load_dotenv()
load_dotenv()

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("pipeline")

# FLORES-200 short-code → (IndicTrans2 lang tag, ASR lang tag)
LANGUAGE_MAP = {
    "pa": ("pan_Guru", "pa"),
    "hi": ("hin_Deva", "hi"),
    "ta": ("tam_Taml", "ta"),
    "te": ("tel_Telu", "te"),
    "mr": ("mar_Deva", "mr"),
    "kn": ("kan_Knda", "kn"),
    "gu": ("guj_Gujr", "gu"),
    "bn": ("ben_Beng", "bn"),
    "or": ("ory_Orya", "or"),
    "ml": ("mal_Mlym", "ml"),
}


# =============================================================================
# STAGE RUNNERS
# =============================================================================

def run_ingestion(input_dir: str, combined_wav: str) -> str:
    from pipeline.ingestion.audio_utils import combine_audio
    log.info("── STAGE 1: Audio ingestion & combining ──")
    return combine_audio(input_dir, combined_wav)


def run_asr_diarization(
    combined_wav: str,
    language_asr: str,
    flores_lang:  str,
    device:       str,
) -> list[dict]:
    from pipeline.asr.sarvam_asr import transcribe_and_diarize

    log.info("── STAGES 2+3: ASR + diarization + translation (Sarvam Saaras v3) ──")
    return transcribe_and_diarize(
        audio_path  = combined_wav,
        flores_lang = flores_lang,
        mode        = "translate",  # transcribe + translate to English in one shot
    )


def run_translation(
    entries:     list[dict],
    flores_lang: str,
    device:      str,
) -> list[dict]:
    # Translation is done by Sarvam inside run_asr_diarization() with mode="translate".
    # This function is kept so --skip_asr resumption still works correctly.
    log.info("── STAGE 4: Translation already done by Sarvam — skipping ──")
    return entries


async def run_extraction(entries: list[dict], flores_lang: str) -> dict:
    from pipeline.extraction.base_llm import BaseLLM
    from pipeline.extraction.insights import FarmerInsightExtractor
    # from pipeline.extraction.narration import NarrationGenerator, SummaryGenerator
    from pipeline.extraction.narration import NarrationGenerator
    from pipeline.extraction.conclusion import ConclusionGenerator
    from pipeline.extraction.metadata import MetadataExtractor
    from pipeline.extraction.participants import ParticipantExtractor
    from pipeline.extraction.terminology import TerminologyExtractor

    log.info("── STAGE 5: Extraction ──")

    def _share_model(source: BaseLLM, target: BaseLLM) -> None:
        """Point target at source's already-loaded model weights."""
        target.model     = source.model
        target.tokenizer = source.tokenizer
        target.device    = source.device

    # Load model once; share weights across all extractors
    terminology_extractor = TerminologyExtractor()
    narration_gen         = NarrationGenerator(base=terminology_extractor)

    insight_extractor     = FarmerInsightExtractor.__new__(FarmerInsightExtractor)
    participant_extractor = ParticipantExtractor.__new__(ParticipantExtractor)
    # summary_gen           = SummaryGenerator.__new__(SummaryGenerator)
    conclusion_gen        = ConclusionGenerator.__new__(ConclusionGenerator)
    metadata_extractor    = MetadataExtractor.__new__(MetadataExtractor)
    
    for ext in (insight_extractor, participant_extractor, conclusion_gen, metadata_extractor):
        _share_model(terminology_extractor, ext)
    
    # for ext in (insight_extractor, participant_extractor, summary_gen):
    #     _share_model(terminology_extractor, ext)

    terminology_task = asyncio.ensure_future(terminology_extractor.extract(entries, flores_lang=flores_lang))
    narration        = narration_gen.generate(entries, max_chars=20000)
    insights         = await insight_extractor.extract(entries)
    participants     = await participant_extractor.extract(entries)
    terminology      = await terminology_task
    # metadata from the narration text (English)
    metadata = metadata_extractor.extract(narration["narration"], use_llm=True)

    # final_summary = await summary_gen.generate(
    #     participants = participants,
    #     challenges   = insights.get("challenges", []),
    #     questions    = insights.get("farmer_questions", []),
    #     narration    = narration["narration"],
    # )
    final_conclusion = await conclusion_gen.generate(
    participants = participants,
    challenges   = insights.get("challenges", []),
    questions    = insights.get("farmer_questions", []),
    narration    = narration["narration"],
    )

    return {
        # "summary":      final_summary,
        "conclusion":   final_conclusion,
        "metadata":     metadata,
        "narration":    narration,
        "terminology":  terminology,
        "insights":     insights,
        "participants": participants,
    }


# =============================================================================
# PIPELINE
# =============================================================================

async def pipeline(args):
    from pipeline.report.assembler import assemble, save
    from pipeline.transcript.builder import load_transcript, save_transcript

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    if device == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.language not in LANGUAGE_MAP:
        raise ValueError(
            f"Unsupported language '{args.language}'. "
            f"Supported: {list(LANGUAGE_MAP.keys())}"
        )
    flores_lang, asr_lang = LANGUAGE_MAP[args.language]
    log.info(f"Language: {args.language} → ASR={asr_lang}, IndicTrans2={flores_lang}")

    # Intermediate file paths
    combined_wav          = os.path.join(args.output_dir, "combined.wav")
    raw_transcript_path   = os.path.join(args.output_dir, "transcript_raw.json")
    trans_transcript_path = os.path.join(args.output_dir, "transcript_translated.json")

    # ── Stage 1: Ingestion ───────────────────────────────────────────────────
    if args.skip_combine:
        if not os.path.exists(combined_wav):
            raise FileNotFoundError(
                f"--skip_combine set but {combined_wav} not found."
            )
        log.info(f"Skipping combine — using existing {combined_wav}")
    else:
        run_ingestion(args.input_dir, combined_wav)

    # ── Stage 2 + 3: Diarization + ASR ──────────────────────────────────────
    if args.skip_asr and os.path.exists(raw_transcript_path):
        log.info(f"Skipping ASR — loading {raw_transcript_path}")
        entries = load_transcript(raw_transcript_path)
    else:
        entries = run_asr_diarization(combined_wav, asr_lang, device)
        save_transcript(entries, raw_transcript_path)
        log.info(f"Raw transcript: {raw_transcript_path} ({len(entries)} segments)")

    # ── Stage 4: Translation ─────────────────────────────────────────────────
    if args.skip_translation and os.path.exists(trans_transcript_path):
        log.info(f"Skipping translation — loading {trans_transcript_path}")
        entries = load_transcript(trans_transcript_path)
    else:
        entries = run_translation(entries, flores_lang, device)
        save_transcript(entries, trans_transcript_path)
        log.info(f"Translated transcript: {trans_transcript_path}")

    # ── Stage 5: Extraction ──────────────────────────────────────────────────
    extracted = await run_extraction(entries, flores_lang=flores_lang)

    # ── Stage 6: Assemble + save report ─────────────────────────────────────
    log.info("── STAGE 6: Assembling report ──")
    report = assemble(
        # summary      = extracted["summary"],
        conclusion   = extracted["conclusion"],
        metadata     = extracted["metadata"],
        narration    = extracted["narration"],
        terminology  = extracted["terminology"],
        insights     = extracted["insights"],
        participants = extracted["participants"],
    )
    export_pdf = not args.no_pdf
    save(report, args.output_dir, export_pdf=export_pdf)

    log.info("══ Pipeline complete ══")
    log.info(f"  Segments processed : {len(entries)}")
    log.info(f"  Terminology terms  : {len(extracted['terminology'])}")
    log.info(f"  Farmer questions   : {len(extracted['insights'].get('farmer_questions', []))}")
    log.info(f"  Participants       : {extracted['participants'].get('total_count', 0)}")
    log.info(f"  Metadata fields    : {sum(1 for v in extracted['metadata'].values() if v is not None)} / {len(extracted['metadata'])}")
    log.info(f"  Conclusion chars   : {len(extracted['conclusion']) if extracted.get('conclusion') else 0}")
    log.info(f"  Outputs saved to   : {args.output_dir}")
    if export_pdf:
        log.info(f"  PDF report         : {args.output_dir}/outreach_report.pdf")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Outreach Report Pipeline — audio → report"
    )
    parser.add_argument(
        "--input_dir",  required=True,
        help="Directory containing audio files (wav / mp3 / m4a / flac)"
    )
    parser.add_argument(
        "--language",   required=True,
        help=f"Source language code. Supported: {list(LANGUAGE_MAP.keys())}"
    )
    parser.add_argument(
        "--output_dir", default="./outputs",
        help="Directory for all outputs (default: ./outputs)"
    )
    parser.add_argument(
        "--skip_combine",     action="store_true",
        help="Skip audio combining — reuse existing combined.wav"
    )
    parser.add_argument(
        "--skip_asr",         action="store_true",
        help="Skip ASR + diarization — reuse existing transcript_raw.json"
    )
    parser.add_argument(
        "--skip_translation", action="store_true",
        help="Skip translation — reuse existing transcript_translated.json"
    )
    parser.add_argument(
        "--no_pdf", action="store_true",
        help="Skip PDF generation — save JSON outputs only"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(pipeline(args))