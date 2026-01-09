# File: precompute_voice.py
# Pre-compute voice conditionals for faster TTS inference.
#
# Usage:
#   python precompute_voice.py voices/MickeyMouse.mp3
#   python precompute_voice.py --all  # Pre-compute all voices in voices/ folder
#
# This extracts speaker embeddings and other conditionals from reference audio,
# saving them as .conds files that can be loaded at startup for faster inference.

import argparse
import logging
from pathlib import Path
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Determine device
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def precompute_conditionals(audio_path: Path, output_path: Path = None, device: str = None):
    """
    Pre-compute voice conditionals from a reference audio file.

    Args:
        audio_path: Path to the reference audio file (.wav or .mp3)
        output_path: Path to save the .conds file (defaults to same name with .conds extension)
        device: Device to use ('cuda', 'mps', or 'cpu')

    Returns:
        Path to the saved .conds file
    """
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    if device is None:
        device = get_device()

    if output_path is None:
        output_path = audio_path.with_suffix('.conds')

    logger.info(f"Loading ChatterboxTurboTTS model on {device}...")
    model = ChatterboxTurboTTS.from_pretrained(device=device)

    logger.info(f"Extracting conditionals from: {audio_path}")
    model.prepare_conditionals(str(audio_path), exaggeration=0.5)

    logger.info(f"Saving conditionals to: {output_path}")
    model.conds.save(output_path)

    # Verify the saved file
    file_size = output_path.stat().st_size
    logger.info(f"Saved {file_size:,} bytes to {output_path}")

    # Verify it can be loaded
    from chatterbox.tts_turbo import Conditionals
    loaded_conds = Conditionals.load(output_path, map_location=device)
    logger.info("Verified: conditionals can be loaded successfully")

    return output_path


def precompute_all_voices(voices_dir: Path, device: str = None):
    """
    Pre-compute conditionals for all voice files in a directory.

    Args:
        voices_dir: Directory containing voice files
        device: Device to use
    """
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    if device is None:
        device = get_device()

    # Find all audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
    audio_files = [
        f for f in voices_dir.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]

    if not audio_files:
        logger.warning(f"No audio files found in {voices_dir}")
        return

    logger.info(f"Found {len(audio_files)} audio files in {voices_dir}")

    # Load model once
    logger.info(f"Loading ChatterboxTurboTTS model on {device}...")
    model = ChatterboxTurboTTS.from_pretrained(device=device)

    # Process each file
    for audio_path in audio_files:
        output_path = audio_path.with_suffix('.conds')

        # Skip if already exists and is newer than source
        if output_path.exists():
            if output_path.stat().st_mtime >= audio_path.stat().st_mtime:
                logger.info(f"Skipping {audio_path.name} (up-to-date)")
                continue

        try:
            logger.info(f"Processing: {audio_path.name}")
            model.prepare_conditionals(str(audio_path), exaggeration=0.5)
            model.conds.save(output_path)
            logger.info(f"  Saved: {output_path.name} ({output_path.stat().st_size:,} bytes)")
        except Exception as e:
            logger.error(f"  Failed: {e}")

    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute voice conditionals for faster TTS inference"
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        help="Path to audio file to process"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all voice files in the voices/ directory"
    )
    parser.add_argument(
        "--voices-dir",
        type=Path,
        default=Path("voices"),
        help="Directory containing voice files (default: voices/)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for .conds file (default: same as input with .conds extension)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect)"
    )

    args = parser.parse_args()

    if args.all:
        precompute_all_voices(args.voices_dir, device=args.device)
    elif args.audio_path:
        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            logger.error(f"File not found: {audio_path}")
            return 1
        precompute_conditionals(audio_path, args.output, args.device)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
