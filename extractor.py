"""
HuBERT Embedding Extractor
==========================
Extracts encoder hidden-state embeddings from a HuBERT model and saves them
to disk as .npy files, along with a metadata JSON that maps each embedding
file to its transcription text.

Usage:
  python extractor.py --dataset_name <HF_DATASET_OR_LOCAL_PATH> \
                      --output_dir ./embeddings \
                      --split train

  python extractor.py --dataset_name mozilla-foundation/common_voice_11_0 \
                      --dataset_config vi --split train \
                      --audio_column audio --text_column sentence \
                      --max_samples 1000
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoFeatureExtractor, HubertModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract HuBERT embeddings from an audio dataset."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/hubert-base-ls960",
        help="HuBERT model name or local path (default: facebook/hubert-base-ls960)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path (e.g. 'mozilla-foundation/common_voice_11_0' or './my_data')",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset config/language (e.g. 'vi' for Vietnamese Common Voice)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to extract from (default: 'train')",
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default="audio",
        help="Name of the audio column in the dataset (default: 'audio')",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text/transcription column (default: 'text')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embeddings",
        help="Directory to save extracted embeddings (default: ./embeddings)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to extract (default: all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for extraction. Keep at 1 if audio lengths vary a lot (default: 1)",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=False,
        help="Use GPU if available (default: CPU only for local machines)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for gated datasets",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Trust remote code when loading dataset",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Device setup ──────────────────────────────────────────────────
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # ── Load model & feature extractor ────────────────────────────────
    print(f"Loading HuBERT model: {args.model_name}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = HubertModel.from_pretrained(args.model_name)
    model = model.eval().to(device)
    print(f"Model loaded. Hidden size: {model.config.hidden_size}")

    # ── Load dataset ──────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset_name} (config={args.dataset_config}, split={args.split})")
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Using {len(dataset)} samples (limited by --max_samples)")
    else:
        print(f"Total samples: {len(dataset)}")

    # Validate columns
    if args.audio_column not in dataset.column_names:
        print(f"ERROR: Audio column '{args.audio_column}' not found. Available: {dataset.column_names}")
        sys.exit(1)
    if args.text_column not in dataset.column_names:
        print(f"ERROR: Text column '{args.text_column}' not found. Available: {dataset.column_names}")
        sys.exit(1)

    # ── Prepare output directory ──────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    embeddings_dir = os.path.join(args.output_dir, "features")
    os.makedirs(embeddings_dir, exist_ok=True)

    # ── Extract embeddings ────────────────────────────────────────────
    metadata = []
    errors = []

    print("\nExtracting embeddings...")
    for i, sample in enumerate(tqdm(dataset, desc="Extracting")):
        try:
            audio = sample[args.audio_column]
            text = sample[args.text_column]

            # Handle different audio formats from datasets library
            if isinstance(audio, dict):
                audio_array = audio["array"]
                sampling_rate = audio["sampling_rate"]
            else:
                print(f"  WARNING: Sample {i} has unexpected audio format, skipping.")
                errors.append({"index": i, "error": "unexpected audio format"})
                continue

            # Resample if needed
            if sampling_rate != feature_extractor.sampling_rate:
                import librosa
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sampling_rate,
                    target_sr=feature_extractor.sampling_rate,
                )
                sampling_rate = feature_extractor.sampling_rate

            # Extract features through the feature extractor (preprocessing)
            inputs = feature_extractor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            )
            input_values = inputs.input_values.to(device)

            # Forward through HuBERT encoder → get hidden states
            with torch.no_grad():
                outputs = model(input_values)
                hidden_states = outputs.last_hidden_state  # (1, T, hidden_size)

            # Save embedding as .npy
            embedding = hidden_states.squeeze(0).cpu().numpy()  # (T, hidden_size)
            npy_filename = f"{i:06d}.npy"
            npy_path = os.path.join(embeddings_dir, npy_filename)
            np.save(npy_path, embedding)

            # Record metadata
            metadata.append({
                "id": i,
                "embedding_file": npy_filename,
                "text": text,
                "embedding_shape": list(embedding.shape),
                "audio_duration_sec": round(len(audio_array) / sampling_rate, 3),
            })

        except Exception as e:
            print(f"  ERROR on sample {i}: {e}")
            errors.append({"index": i, "error": str(e)})
            continue

    # ── Save metadata ─────────────────────────────────────────────────
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Save extraction config for reproducibility
    config_path = os.path.join(args.output_dir, "extraction_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": args.model_name,
            "hidden_size": model.config.hidden_size,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "audio_column": args.audio_column,
            "text_column": args.text_column,
            "total_extracted": len(metadata),
            "total_errors": len(errors),
            "feature_extractor_sampling_rate": feature_extractor.sampling_rate,
        }, f, indent=2)

    if errors:
        errors_path = os.path.join(args.output_dir, "errors.json")
        with open(errors_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Embeddings saved to : {os.path.abspath(embeddings_dir)}")
    print(f"  Metadata saved to   : {os.path.abspath(metadata_path)}")
    print(f"  Total extracted      : {len(metadata)}")
    print(f"  Total errors         : {len(errors)}")
    if metadata:
        print(f"  Embedding shape      : ({metadata[0]['embedding_shape'][0]}, {metadata[0]['embedding_shape'][1]})")
        print(f"  Hidden size          : {model.config.hidden_size}")
    print("=" * 60)


if __name__ == "__main__":
    main()
