"""Voice input for the same clinical-note -> dialogue pipeline `try_model.py` scores.

Transcribes speech to text locally with faster-whisper — no audio ever leaves
this machine, consistent with `docs/PROJECT_SPEC.md` §1.1 — then hands the
transcript straight to the same `generate()`/`report()` code path
`try_model.py` uses, so a voice note is judged by the identical checks as a
typed one.

Examples
--------
    # Transcribe an existing recording, then generate + score
    python scripts/voice_to_note.py --file recordings/note1.wav

    # Record from the microphone (Enter to stop), then generate + score
    python scripts/voice_to_note.py --mic

    # Compare base vs. fine-tuned on a recorded note
    python scripts/voice_to_note.py --mic --compare
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from scipy.io.wavfile import write as write_wav  # noqa: E402

from scripts.try_model import DEFAULT_ADAPTER, DEFAULT_BASE, RULE, build_arms, report  # noqa: E402
from src.inference.local_hf import generate, load_model  # noqa: E402

WHISPER_SAMPLE_RATE = 16_000


def record_from_mic(sample_rate: int = WHISPER_SAMPLE_RATE) -> np.ndarray:
    """Record mono float32 audio from the default input device until Enter is pressed."""
    import sounddevice as sd

    chunks: list[np.ndarray] = []

    def callback(indata, frames, time_info, status):
        chunks.append(indata.copy())

    print("Recording... press Enter to stop.", file=sys.stderr)
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
        input()

    if not chunks:
        raise SystemExit("No audio captured.")
    return np.concatenate(chunks, axis=0).reshape(-1)


def transcribe(audio, model_size: str, device: str, compute_type: str, language: str | None) -> str:
    """Run faster-whisper on a file path (str) or an in-memory mono waveform (np.ndarray)."""
    from faster_whisper import WhisperModel

    print(f"Loading Whisper ({model_size}, {device}/{compute_type})...", file=sys.stderr)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(audio, language=language)
    text = " ".join(segment.text.strip() for segment in segments)
    print(f"Detected language: {info.language} (p={info.language_probability:.2f})", file=sys.stderr)
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a clinical note by voice, then generate + score a dialogue from it.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="Path to an audio file (wav/mp3/m4a/flac/...).")
    source.add_argument("--mic", action="store_true", help="Record from the microphone (Enter to stop).")
    parser.add_argument("--save-audio", help="Also write the recorded audio to this .wav path (only with --mic).")

    parser.add_argument(
        "--whisper-model", default="small", help="faster-whisper model size (tiny/base/small/medium/large-v3)."
    )
    parser.add_argument("--whisper-device", default="cpu", choices=["cpu", "cuda"], help="Device for the STT model.")
    parser.add_argument(
        "--whisper-compute-type", default="int8", help="ctranslate2 compute type (int8 for CPU, float16 for GPU)."
    )
    parser.add_argument("--language", default=None, help="Force a transcription language (default: auto-detect).")

    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="LoRA adapter dir (default: the committed one).")
    parser.add_argument("--base", action="store_true", help="Use the frozen base model instead of the adapter.")
    parser.add_argument("--compare", action="store_true", help="Run base AND fine-tuned on the same note.")
    parser.add_argument("--model-name", default=DEFAULT_BASE, help="Base checkpoint id.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--sample", action="store_true", help="Stochastic decoding (default greedy, as in eval).")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.file:
        audio_source = args.file
    else:
        waveform = record_from_mic()
        audio_source = waveform
        if args.save_audio:
            write_wav(args.save_audio, WHISPER_SAMPLE_RATE, waveform)
            print(f"Saved recording to {args.save_audio}", file=sys.stderr)

    note = transcribe(audio_source, args.whisper_model, args.whisper_device, args.whisper_compute_type, args.language)
    if not note:
        raise SystemExit("Transcription produced no text — nothing to generate from.")

    print(f"\n{RULE}\nTRANSCRIPT ({len(note)} chars)\n{RULE}")
    print(note)

    for label, adapter in build_arms(args):
        if adapter is not None and not Path(adapter).exists():
            raise SystemExit(f"Adapter not found at {adapter} — train first, or pass --base.")

        print(f"\nLoading {label} ...", file=sys.stderr)
        model, tokenizer = load_model(
            model_name=args.model_name,
            max_seq_len=args.max_seq_len,
            load_in_4bit=True,
            seed=args.seed,
            adapter_path=adapter,
        )
        gen = generate(
            model,
            tokenizer,
            {"clinical_note": note},
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.sample,
        )
        report(label, note, gen, None)


if __name__ == "__main__":
    main()
