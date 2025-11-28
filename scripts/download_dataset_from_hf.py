#!/usr/bin/env python
"""
Download dataset from HuggingFace Datasets Hub

This script downloads a dataset from HuggingFace Datasets Hub and saves it
as JSONL files (train.jsonl, valid.jsonl, test.jsonl) for training.

Example usage:
    # Download Java dataset
    python scripts/download_dataset_from_hf.py \
        --dataset-id reiprasetya-study/java-class-names \
        --output datasets/java

    # Download Python dataset
    python scripts/download_dataset_from_hf.py \
        --dataset-id reiprasetya-study/python-class-names \
        --output datasets/python

    # Download specific splits only
    python scripts/download_dataset_from_hf.py \
        --dataset-id reiprasetya-study/java-class-names \
        --output datasets/java \
        --splits train,validation
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from hf_utils import get_hf_token


def download_dataset(
    dataset_id: str,
    output_dir: str,
    splits: Optional[List[str]] = None,
    token: Optional[str] = None
):
    """Download dataset from HuggingFace Datasets Hub"""

    # Get token (optional for public datasets)
    if token is None:
        token = get_hf_token()
        # Don't fail if token not found - public datasets don't need auth

    print(f"\n{'='*60}")
    print(f"📥 Downloading Dataset from HuggingFace Datasets Hub")
    print(f"{'='*60}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Output Directory: {output_dir}")
    if splits:
        print(f"Splits: {', '.join(splits)}")
    else:
        print(f"Splits: all (train, validation, test)")
    print(f"{'='*60}\n")

    # Import datasets library
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: 'datasets' library not installed")
        print("   Install with: pip install datasets")
        return False

    # Step 1: Download dataset
    print("📥 Step 1/3: Downloading dataset from HuggingFace Hub...")
    try:
        if splits:
            # Download specific splits
            dataset_dict = {}
            for split in splits:
                print(f"  Downloading split: {split}...")
                dataset_dict[split] = load_dataset(
                    dataset_id,
                    split=split,
                    token=token
                )
        else:
            # Download all splits
            print("  Downloading all splits...")
            dataset_dict = load_dataset(dataset_id, token=token)

        print("  ✅ Download complete")

    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        print("\n💡 Troubleshooting:")
        print("   - Check dataset ID is correct")
        print("   - For private datasets, ensure HF_TOKEN is set")
        print("   - Check internet connection")
        return False

    # Step 2: Create output directory
    print(f"\n📁 Step 2/3: Creating output directory...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"  ✅ Directory ready: {output_dir}")

    # Step 3: Save as JSONL files
    print(f"\n💾 Step 3/3: Saving as JSONL files...")

    # Map HuggingFace split names to our naming convention
    split_mapping = {
        'train': 'train',
        'validation': 'valid',
        'test': 'test',
    }

    saved_files = []
    total_samples = 0

    for hf_split, local_name in split_mapping.items():
        # Skip if split not in dataset
        if hf_split not in dataset_dict:
            continue

        # Skip if user specified splits and this one is not included
        if splits and hf_split not in splits:
            continue

        split_data = dataset_dict[hf_split]
        output_file = output_path / f"{local_name}.jsonl"

        # Write JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in split_data:
                # Convert to dict if needed
                if hasattr(sample, 'to_dict'):
                    sample = sample.to_dict()
                json.dump(sample, f, ensure_ascii=False)
                f.write('\n')

        sample_count = len(split_data)
        total_samples += sample_count
        saved_files.append(output_file.name)

        print(f"  ✅ {local_name}.jsonl: {sample_count:,} samples")

    # Success!
    print(f"\n{'='*60}")
    print(f"✅ Download Complete!")
    print(f"{'='*60}")
    print(f"📊 Total samples: {total_samples:,}")
    print(f"📁 Files saved to: {output_dir}")
    for filename in saved_files:
        print(f"   - {filename}")
    print(f"{'='*60}")
    print(f"\n🚀 Ready to train:")
    print(f"   python scripts/train_codegen.py \\")
    print(f"       --data {output_dir} \\")
    print(f"       --output model/checkpoints/run1 \\")
    print(f"       --batch-size 12 --grad-accum 3 --epochs 5 \\")
    print(f"       --gradient-checkpointing")
    print()

    return True


def list_dataset_info(dataset_id: str, token: Optional[str] = None):
    """List dataset information without downloading"""

    # Get token (optional)
    if token is None:
        token = get_hf_token()

    print(f"\n{'='*60}")
    print(f"ℹ️  Dataset Information")
    print(f"{'='*60}")
    print(f"Dataset ID: {dataset_id}")
    print(f"{'='*60}\n")

    try:
        from datasets import load_dataset_builder

        # Get dataset builder (lightweight, doesn't download data)
        builder = load_dataset_builder(dataset_id, token=token)

        print(f"📋 Description:")
        if builder.info.description:
            print(f"   {builder.info.description[:200]}...")
        else:
            print(f"   No description available")

        print(f"\n📊 Splits:")
        if builder.info.splits:
            for split_name, split_info in builder.info.splits.items():
                print(f"   - {split_name}: {split_info.num_examples:,} samples")
        else:
            print(f"   No split information available")

        print(f"\n🔗 URL: https://huggingface.co/datasets/{dataset_id}")
        print()

        return True

    except ImportError:
        print("❌ Error: 'datasets' library not installed")
        print("   Install with: pip install datasets")
        return False
    except Exception as e:
        print(f"❌ Failed to get dataset info: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download dataset from HuggingFace Datasets Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Java dataset
  python scripts/download_dataset_from_hf.py \\
      --dataset-id reiprasetya-study/java-class-names \\
      --output datasets/java

  # Download Python dataset
  python scripts/download_dataset_from_hf.py \\
      --dataset-id reiprasetya-study/python-class-names \\
      --output datasets/python

  # Download only training data
  python scripts/download_dataset_from_hf.py \\
      --dataset-id reiprasetya-study/java-class-names \\
      --output datasets/java \\
      --splits train

  # Get dataset info without downloading
  python scripts/download_dataset_from_hf.py \\
      --dataset-id reiprasetya-study/java-class-names \\
      --info
        """
    )

    parser.add_argument(
        '--dataset-id',
        required=True,
        help='HuggingFace dataset ID (e.g., username/dataset-name)'
    )

    parser.add_argument(
        '--output',
        help='Output directory to save JSONL files (required unless using --info)'
    )

    parser.add_argument(
        '--splits',
        help='Comma-separated list of splits to download (e.g., train,validation,test). Default: all splits'
    )

    parser.add_argument(
        '--info',
        action='store_true',
        help='Show dataset information without downloading'
    )

    args = parser.parse_args()

    # Info mode
    if args.info:
        success = list_dataset_info(args.dataset_id)
        sys.exit(0 if success else 1)

    # Download mode - require output directory
    if not args.output:
        parser.error("--output is required (unless using --info)")

    # Parse splits
    splits = None
    if args.splits:
        splits = [s.strip() for s in args.splits.split(',')]

    success = download_dataset(
        dataset_id=args.dataset_id,
        output_dir=args.output,
        splits=splits
    )

    sys.exit(0 if success else 1)
