#!/usr/bin/env python
"""
View training results from HuggingFace Hub without downloading the model

This script allows you to view training logs, metrics, and evaluation results
from HuggingFace Hub without downloading the full model weights.

Example usage:
    # View metrics summary
    python scripts/view_hf_results.py --hub-model-id reiprasetyastudy/codegen-java-run1

    # View detailed results
    python scripts/view_hf_results.py --hub-model-id reiprasetyastudy/codegen-java-run1 --detailed

    # Download logs for plotting
    python scripts/view_hf_results.py --hub-model-id reiprasetyastudy/codegen-java-run1 --download-logs
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from hf_utils import (
    get_hf_token,
    download_file_from_hub,
    list_repo_files,
    get_repo_url
)


def view_metrics(repo_id: str, token: Optional[str] = None):
    """Download and display metrics from HuggingFace Hub"""

    print(f"\n{'='*60}")
    print(f"📊 Viewing Results from HuggingFace Hub")
    print(f"{'='*60}")
    print(f"Model: {repo_id}")
    print(f"URL: {get_repo_url(repo_id)}")
    print(f"{'='*60}\n")

    # Create temp directory for downloads
    temp_dir = Path(".hf_temp") / repo_id.replace("/", "_")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Download metrics.json
    print("📥 Downloading metrics...")
    metrics_path = download_file_from_hub(
        repo_id=repo_id,
        filename="metrics/metrics.json",
        local_dir=str(temp_dir),
        token=token
    )

    if metrics_path is None:
        print("❌ Metrics not found in repository")
        return False

    # Load and display metrics
    with open(metrics_path) as f:
        metrics = json.load(f)

    print(f"\n{'='*60}")
    print("📈 EVALUATION METRICS")
    print(f"{'='*60}")

    # Format metrics table
    metrics_table = [
        ("Exact Match", f"{metrics.get('exact_match', 0):.2%}"),
        ("Case-Insensitive Match", f"{metrics.get('exact_match_case_insensitive', 0):.2%}"),
        (f"Top-{metrics.get('k', 5)} Accuracy", f"{metrics.get('topk_accuracy', 0):.2%}"),
        ("Avg Levenshtein Distance", f"{metrics.get('avg_levenshtein', 0):.2f}"),
        ("Samples Evaluated", f"{metrics.get('n', 0):,}"),
    ]

    if 'elapsed_time_seconds' in metrics:
        elapsed_min = metrics['elapsed_time_seconds'] / 60
        metrics_table.append(("Evaluation Time", f"{elapsed_min:.1f} minutes"))
        metrics_table.append(("Speed", f"{metrics.get('samples_per_second', 0):.2f} samples/sec"))

    # Print formatted table
    max_label_len = max(len(label) for label, _ in metrics_table)
    for label, value in metrics_table:
        print(f"  {label:<{max_label_len}} : {value}")

    print(f"{'='*60}\n")

    return True


def view_detailed_results(repo_id: str, token: Optional[str] = None, limit: int = 10):
    """Download and display sample predictions"""

    temp_dir = Path(".hf_temp") / repo_id.replace("/", "_")
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading detailed results...")
    detailed_path = download_file_from_hub(
        repo_id=repo_id,
        filename="metrics/detailed_results.jsonl",
        local_dir=str(temp_dir),
        token=token
    )

    if detailed_path is None:
        print("❌ Detailed results not found in repository")
        return False

    print(f"\n{'='*60}")
    print(f"🔍 SAMPLE PREDICTIONS (showing first {limit})")
    print(f"{'='*60}\n")

    # Read and display samples
    with open(detailed_path) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            sample = json.load(line)
            print(f"Sample {i+1}:")
            print(f"  True Name: {sample['true_name']}")
            print(f"  Predictions: {', '.join(sample['predictions'][:5])}")
            print(f"  Exact Match: {'✅' if sample['exact_match'] else '❌'}")
            print(f"  In Top-{len(sample['predictions'])}: {'✅' if sample['in_topk'] else '❌'}")
            print()

    return True


def download_training_logs(repo_id: str, output_dir: str, token: Optional[str] = None):
    """Download training logs for local plotting"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Downloading training logs to: {output_dir}")

    # Download training log CSV
    csv_path = download_file_from_hub(
        repo_id=repo_id,
        filename="logs/training_log.csv",
        local_dir=str(output_path),
        token=token
    )

    # Download training curve image
    png_path = download_file_from_hub(
        repo_id=repo_id,
        filename="logs/training_curve.png",
        local_dir=str(output_path),
        token=token
    )

    # Download checkpoints list
    ckpt_path = download_file_from_hub(
        repo_id=repo_id,
        filename="logs/checkpoints.txt",
        local_dir=str(output_path),
        token=token
    )

    if csv_path or png_path or ckpt_path:
        print(f"\n✅ Logs downloaded to: {output_dir}")
        if csv_path:
            print(f"   - training_log.csv")
        if png_path:
            print(f"   - training_curve.png")
        if ckpt_path:
            print(f"   - checkpoints.txt")
        return True
    else:
        print("❌ No training logs found in repository")
        return False


def list_available_files(repo_id: str, token: Optional[str] = None):
    """List all files in the repository"""

    print(f"\n📂 Listing files in repository...")
    files = list_repo_files(repo_id, token=token)

    if not files:
        return False

    print(f"\n{'='*60}")
    print(f"📁 FILES IN REPOSITORY")
    print(f"{'='*60}")

    # Organize files by directory
    from collections import defaultdict
    by_dir = defaultdict(list)

    for file in sorted(files):
        if "/" in file:
            dir_name = file.rsplit("/", 1)[0]
            by_dir[dir_name].append(file.rsplit("/", 1)[1])
        else:
            by_dir["root"].append(file)

    # Print organized list
    for dir_name in sorted(by_dir.keys()):
        if dir_name == "root":
            print(f"\n📄 Root:")
        else:
            print(f"\n📁 {dir_name}/:")

        for filename in sorted(by_dir[dir_name]):
            print(f"   - {filename}")

    print(f"\n{'='*60}\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="View training results from HuggingFace Hub without downloading the model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View metrics summary
  python scripts/view_hf_results.py --hub-model-id reiprasetyastudy/codegen-java-run1

  # View detailed predictions
  python scripts/view_hf_results.py --hub-model-id reiprasetyastudy/codegen-java-run1 --detailed

  # Download training logs
  python scripts/view_hf_results.py --hub-model-id reiprasetyastudy/codegen-java-run1 --download-logs

  # List all files in repository
  python scripts/view_hf_results.py --hub-model-id reiprasetyastudy/codegen-java-run1 --list-files
        """
    )

    parser.add_argument(
        '--hub-model-id',
        required=True,
        help='HuggingFace model ID (e.g., username/model-name)'
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed sample predictions'
    )

    parser.add_argument(
        '--detailed-limit',
        type=int,
        default=10,
        help='Number of samples to show in detailed view (default: 10)'
    )

    parser.add_argument(
        '--download-logs',
        action='store_true',
        help='Download training logs to local directory'
    )

    parser.add_argument(
        '--logs-dir',
        default='downloaded_logs',
        help='Directory to save downloaded logs (default: downloaded_logs)'
    )

    parser.add_argument(
        '--list-files',
        action='store_true',
        help='List all files in the repository'
    )

    args = parser.parse_args()

    # Get token
    token = get_hf_token()

    success = True

    # Always show metrics first
    if not view_metrics(args.hub_model_id, token=token):
        success = False

    # Show detailed results if requested
    if args.detailed:
        if not view_detailed_results(args.hub_model_id, token=token, limit=args.detailed_limit):
            success = False

    # Download logs if requested
    if args.download_logs:
        if not download_training_logs(args.hub_model_id, args.logs_dir, token=token):
            success = False

    # List files if requested
    if args.list_files:
        if not list_available_files(args.hub_model_id, token=token):
            success = False

    # Clean up temp directory
    temp_dir = Path(".hf_temp")
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)

    sys.exit(0 if success else 1)
