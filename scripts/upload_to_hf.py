#!/usr/bin/env python
"""
Upload trained model to HuggingFace Hub

This script uploads a trained model checkpoint along with training logs,
metrics, and auto-generated README to HuggingFace Hub.

Example usage:
    python scripts/upload_to_hf.py \
        --ckpt model/checkpoints/run1-java-codegen \
        --hub-model-id reiprasetyastudy/codegen-java-run1 \
        --metrics model/metrics/run1-java-codegen/metrics.json \
        --private
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from hf_utils import (
    get_hf_token,
    create_repo_if_not_exists,
    upload_file_safe,
    upload_folder_safe,
    get_repo_url
)
from generate_readme import generate_readme


def upload_model_to_hub(
    checkpoint_path: str,
    hub_model_id: str,
    metrics_file: Optional[str] = None,
    model_name: str = "CodeGen",
    language: str = "java",
    private: bool = False,
    token: Optional[str] = None
):
    """
    Upload model checkpoint and artifacts to HuggingFace Hub

    Args:
        checkpoint_path: Path to model checkpoint directory
        hub_model_id: HuggingFace model ID (e.g., username/model-name)
        metrics_file: Path to metrics.json (optional)
        model_name: Model name for README (default: CodeGen)
        language: Programming language (default: java)
        private: Make repository private (default: False)
        token: HuggingFace token (optional, will use env if not provided)
    """

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint path does not exist: {checkpoint_path}")
        return False

    # Get token
    if token is None:
        token = get_hf_token()
        if token is None:
            print("❌ HuggingFace token not found!")
            print("   Set HF_TOKEN environment variable or create .env file")
            print("   Get token from: https://huggingface.co/settings/tokens")
            return False

    print(f"\n{'='*60}")
    print(f"🚀 Uploading to HuggingFace Hub")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Hub Model ID: {hub_model_id}")
    print(f"Private: {private}")
    print(f"{'='*60}\n")

    # Step 1: Create repository
    print("📦 Step 1/5: Creating repository...")
    if not create_repo_if_not_exists(hub_model_id, private=private, token=token):
        return False

    # Step 2: Generate README
    print("\n📝 Step 2/5: Generating README...")
    readme_path = ckpt_path / "README.md"

    # Auto-detect metrics file if not provided
    if metrics_file is None:
        # Try to find metrics in standard locations
        possible_metrics = [
            ckpt_path / "metrics.json",
            ckpt_path / "metrics" / "metrics.json",
        ]
        for possible in possible_metrics:
            if possible.exists():
                metrics_file = str(possible)
                print(f"  ℹ️  Auto-detected metrics: {metrics_file}")
                break

    if metrics_file is None:
        print("  ⚠️  No metrics file found, generating README without metrics")
        metrics_file = ""

    generate_readme(
        hub_model_id=hub_model_id,
        metrics_file=metrics_file,
        model_name=model_name,
        language=language,
        output_file=str(readme_path)
    )

    # Step 3: Upload model files
    print("\n📤 Step 3/5: Uploading model files...")

    # Core model files to upload
    model_files = [
        "pytorch_model.bin",
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "generation_config.json",
    ]

    uploaded_count = 0
    for filename in model_files:
        file_path = ckpt_path / filename
        if file_path.exists():
            if upload_file_safe(
                file_path=str(file_path),
                repo_id=hub_model_id,
                path_in_repo=filename,
                token=token
            ):
                uploaded_count += 1

    print(f"  ✅ Uploaded {uploaded_count} model files")

    # Step 4: Upload README
    print("\n📄 Step 4/5: Uploading README...")
    upload_file_safe(
        file_path=str(readme_path),
        repo_id=hub_model_id,
        path_in_repo="README.md",
        token=token
    )

    # Step 5: Upload logs and metrics
    print("\n📊 Step 5/5: Uploading logs and metrics...")

    # Upload training log
    training_log = ckpt_path / "training_log.csv"
    if training_log.exists():
        upload_file_safe(
            file_path=str(training_log),
            repo_id=hub_model_id,
            path_in_repo="logs/training_log.csv",
            token=token
        )

    # Upload training curve
    training_curve = ckpt_path / "training_curve.png"
    if training_curve.exists():
        upload_file_safe(
            file_path=str(training_curve),
            repo_id=hub_model_id,
            path_in_repo="logs/training_curve.png",
            token=token
        )

    # Upload checkpoints list
    checkpoints_file = ckpt_path / "checkpoints.txt"
    if checkpoints_file.exists():
        upload_file_safe(
            file_path=str(checkpoints_file),
            repo_id=hub_model_id,
            path_in_repo="logs/checkpoints.txt",
            token=token
        )

    # Upload metrics
    if metrics_file and Path(metrics_file).exists():
        upload_file_safe(
            file_path=metrics_file,
            repo_id=hub_model_id,
            path_in_repo="metrics/metrics.json",
            token=token
        )

    # Upload detailed results if exists
    metrics_path = Path(metrics_file).parent if metrics_file else None
    if metrics_path:
        detailed_results = metrics_path / "detailed_results.jsonl"
        if detailed_results.exists():
            upload_file_safe(
                file_path=str(detailed_results),
                repo_id=hub_model_id,
                path_in_repo="metrics/detailed_results.jsonl",
                token=token
            )

    # Success!
    repo_url = get_repo_url(hub_model_id)
    print(f"\n{'='*60}")
    print(f"✅ Upload Complete!")
    print(f"{'='*60}")
    print(f"🔗 View your model: {repo_url}")
    print(f"{'='*60}\n")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload trained model to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload CodeGen model
  python scripts/upload_to_hf.py \\
      --ckpt model/checkpoints/run1-java-codegen \\
      --hub-model-id reiprasetyastudy/codegen-java-run1 \\
      --metrics model/metrics/run1-java-codegen/metrics.json

  # Upload as private repository
  python scripts/upload_to_hf.py \\
      --ckpt model/checkpoints/run1-java-codegen \\
      --hub-model-id reiprasetyastudy/codegen-java-run1 \\
      --private

  # Upload CodeT5+ model
  python scripts/upload_to_hf.py \\
      --ckpt model/checkpoints/run1-python-codet5 \\
      --hub-model-id reiprasetyastudy/codet5-python-run1 \\
      --model-name CodeT5+ \\
      --language python
        """
    )

    parser.add_argument(
        '--ckpt',
        required=True,
        help='Path to model checkpoint directory'
    )

    parser.add_argument(
        '--hub-model-id',
        required=True,
        help='HuggingFace model ID (e.g., username/model-name)'
    )

    parser.add_argument(
        '--metrics',
        default=None,
        help='Path to metrics.json file (auto-detected if not provided)'
    )

    parser.add_argument(
        '--model-name',
        default='CodeGen',
        help='Model name for README (default: CodeGen)'
    )

    parser.add_argument(
        '--language',
        default='java',
        help='Programming language (default: java)'
    )

    parser.add_argument(
        '--private',
        action='store_true',
        help='Make repository private'
    )

    args = parser.parse_args()

    success = upload_model_to_hub(
        checkpoint_path=args.ckpt,
        hub_model_id=args.hub_model_id,
        metrics_file=args.metrics,
        model_name=args.model_name,
        language=args.language,
        private=args.private
    )

    sys.exit(0 if success else 1)
