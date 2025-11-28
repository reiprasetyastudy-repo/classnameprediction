#!/usr/bin/env python
"""
Upload dataset to HuggingFace Datasets Hub

This script uploads a prepared dataset (train.jsonl, valid.jsonl, test.jsonl)
to HuggingFace Datasets Hub for easy reuse across cloud instances.

Example usage:
    # Upload Java dataset
    python scripts/upload_dataset_to_hf.py \
        --dataset-dir datasets/java \
        --dataset-id reiprasetyastudy/java-class-names \
        --language java \
        --private

    # Upload Python dataset
    python scripts/upload_dataset_to_hf.py \
        --dataset-dir datasets/python \
        --dataset-id reiprasetyastudy/python-class-names \
        --language python
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from hf_utils import get_hf_token, create_repo_if_not_exists, get_repo_url


def analyze_dataset(dataset_dir: Path):
    """Analyze dataset and return statistics"""
    stats = {
        'train': {'count': 0, 'repos': set(), 'languages': Counter()},
        'valid': {'count': 0, 'repos': set(), 'languages': Counter()},
        'test': {'count': 0, 'repos': set(), 'languages': Counter()},
    }

    for split in ['train', 'valid', 'test']:
        file_path = dataset_dir / f"{split}.jsonl"
        if not file_path.exists():
            continue

        with open(file_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    stats[split]['count'] += 1
                    if 'repo' in data:
                        stats[split]['repos'].add(data['repo'])
                    if 'language' in data:
                        stats[split]['languages'][data['language']] += 1
                except:
                    continue

    return stats


def generate_dataset_card(
    dataset_id: str,
    language: str,
    stats: dict,
    output_file: Path
):
    """Generate dataset card (README.md) for HuggingFace"""

    total_samples = stats['train']['count'] + stats['valid']['count'] + stats['test']['count']
    total_repos = len(stats['train']['repos'] | stats['valid']['repos'] | stats['test']['repos'])

    card = f"""---
language:
- code
- {language}
tags:
- code-generation
- class-naming
- dataset
- {language}
task_categories:
- text-generation
size_categories:
- 100K<n<1M
license: apache-2.0
---

# {language.title()} Class Name Prediction Dataset

This dataset contains {language.title()} code snippets with masked class names, designed for training models to predict class names from code context.

## Dataset Description

- **Language:** {language.title()}
- **Task:** Class name prediction from masked code
- **Format:** JSONL (JSON Lines)
- **Total Samples:** {total_samples:,}
- **Source Repositories:** {total_repos}

## Dataset Structure

Each sample contains:
- `language`: Programming language (e.g., "java", "python")
- `repo`: Source GitHub repository
- `path`: File path in repository
- `class_span`: Line numbers of the class definition
- `source`: Code snippet with masked class name (replaced with `____`)
- `target`: Original class name

### Data Splits

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | {stats['train']['count']:,} | {stats['train']['count']/total_samples*100:.1f}% |
| Validation | {stats['valid']['count']:,} | {stats['valid']['count']/total_samples*100:.1f}% |
| Test | {stats['test']['count']:,} | {stats['test']['count']/total_samples*100:.1f}% |
| **Total** | **{total_samples:,}** | **100%** |

### Example Sample

```json
{{
  "language": "{language}",
  "repo": "https://github.com/example/repo",
  "path": "src/Example.java",
  "class_span": [1, 10],
  "source": "public class ____ {{\\n    private String name;\\n    public String getName() {{ return name; }}\\n}}",
  "target": "Example"
}}
```

## Dataset Creation

### Source Data

- **Repositories:** Public GitHub repositories
- **Selection Criteria:** Popular and well-maintained projects
- **Minimum Lines:** Classes with at least 3 lines of code
- **Deduplication:** Whitespace-normalized to remove duplicates

### Processing Steps

1. Clone GitHub repositories
2. Parse {language.title()} files for class definitions
3. Extract class name and surrounding code
4. Mask class name with `____` placeholder
5. Remove duplicates based on normalized content
6. Split into train/validation/test (80/10/10)

## Usage

### Load Dataset

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("{dataset_id}")

# Load specific split
train_data = load_dataset("{dataset_id}", split="train")
valid_data = load_dataset("{dataset_id}", split="validation")
test_data = load_dataset("{dataset_id}", split="test")
```

### Access Samples

```python
# Get first training sample
sample = dataset['train'][0]
print(f"Source: {{sample['source']}}")
print(f"Target: {{sample['target']}}")
```

### Use in Training

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load dataset
dataset = load_dataset("{dataset_id}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

# Prepare data for training
def prepare_sample(example):
    prompt = f"{{example['source']}}\\nClass name:"
    full_text = prompt + f" {{example['target']}}"
    return tokenizer(full_text, truncation=True, max_length=1024)

tokenized_dataset = dataset.map(prepare_sample, remove_columns=dataset['train'].column_names)

# Train model
training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=5,
    per_device_train_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
)

trainer.train()
```

## Related Resources

- **Project Repository:** [classnameprediction](https://github.com/reiprasetyastudy-repo/classnameprediction)
- **Models Trained on This Dataset:**
  - CodeGen: `reiprasetyastudy/codegen-{language}-*`
  - CodeT5+: `reiprasetyastudy/codet5-{language}-*`

## Citation

```bibtex
@misc{{{dataset_id.split('/')[-1]}}},
  author = {{Rei Prasetya}},
  title = {{{language.title()} Class Name Prediction Dataset}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/datasets/{dataset_id}}}
}}
```

## License

Apache 2.0

---

*This dataset was created using the [classnameprediction](https://github.com/reiprasetyastudy-repo/classnameprediction) pipeline.*
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(card)

    print(f"✅ Dataset card generated: {output_file}")


def upload_dataset(
    dataset_dir: str,
    dataset_id: str,
    language: str = "java",
    private: bool = False,
    token: Optional[str] = None
):
    """Upload dataset to HuggingFace Datasets Hub"""

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False

    # Check required files
    required_files = ['train.jsonl', 'valid.jsonl', 'test.jsonl']
    missing_files = [f for f in required_files if not (dataset_path / f).exists()]
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
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
    print(f"🚀 Uploading Dataset to HuggingFace Datasets Hub")
    print(f"{'='*60}")
    print(f"Dataset Directory: {dataset_dir}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Language: {language}")
    print(f"Private: {private}")
    print(f"{'='*60}\n")

    # Step 1: Analyze dataset
    print("📊 Step 1/4: Analyzing dataset...")
    stats = analyze_dataset(dataset_path)
    print(f"  Train samples: {stats['train']['count']:,}")
    print(f"  Validation samples: {stats['valid']['count']:,}")
    print(f"  Test samples: {stats['test']['count']:,}")
    print(f"  Total: {stats['train']['count'] + stats['valid']['count'] + stats['test']['count']:,}")

    # Step 2: Generate dataset card
    print("\n📝 Step 2/4: Generating dataset card...")
    readme_path = dataset_path / "README.md"
    generate_dataset_card(dataset_id, language, stats, readme_path)

    # Step 3: Create repository
    print("\n📦 Step 3/4: Creating repository...")
    if not create_repo_if_not_exists(dataset_id, private=private, token=token):
        return False

    # Step 4: Upload using datasets library
    print("\n📤 Step 4/4: Uploading dataset files...")

    try:
        from datasets import load_dataset, DatasetDict

        # Load JSONL files
        print("  Loading JSONL files...")
        dataset_dict = load_dataset(
            'json',
            data_files={
                'train': str(dataset_path / 'train.jsonl'),
                'validation': str(dataset_path / 'valid.jsonl'),
                'test': str(dataset_path / 'test.jsonl'),
            }
        )

        # Upload to hub
        print("  Uploading to HuggingFace Hub...")
        dataset_dict.push_to_hub(
            dataset_id,
            private=private,
            token=token
        )

        print("  ✅ Dataset uploaded successfully")

    except ImportError:
        print("  ❌ Error: 'datasets' library not installed")
        print("     Install with: pip install datasets")
        return False
    except Exception as e:
        print(f"  ❌ Upload failed: {e}")
        return False

    # Success!
    repo_url = get_repo_url(dataset_id).replace("/models/", "/datasets/")
    print(f"\n{'='*60}")
    print(f"✅ Upload Complete!")
    print(f"{'='*60}")
    print(f"🔗 View your dataset: {repo_url}")
    print(f"{'='*60}")
    print(f"\n📥 To download on another machine:")
    print(f"   python scripts/download_dataset_from_hf.py \\")
    print(f"       --dataset-id {dataset_id} \\")
    print(f"       --output datasets/{language}")
    print(f"\n💡 Or load directly in Python:")
    print(f"   from datasets import load_dataset")
    print(f"   dataset = load_dataset('{dataset_id}')")
    print()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload dataset to HuggingFace Datasets Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload Java dataset
  python scripts/upload_dataset_to_hf.py \\
      --dataset-dir datasets/java \\
      --dataset-id reiprasetyastudy/java-class-names \\
      --language java \\
      --private

  # Upload Python dataset
  python scripts/upload_dataset_to_hf.py \\
      --dataset-dir datasets/python \\
      --dataset-id reiprasetyastudy/python-class-names \\
      --language python

  # Upload public dataset
  python scripts/upload_dataset_to_hf.py \\
      --dataset-dir datasets/java \\
      --dataset-id reiprasetyastudy/java-class-names \\
      --language java
        """
    )

    parser.add_argument(
        '--dataset-dir',
        required=True,
        help='Path to dataset directory containing train.jsonl, valid.jsonl, test.jsonl'
    )

    parser.add_argument(
        '--dataset-id',
        required=True,
        help='HuggingFace dataset ID (e.g., username/dataset-name)'
    )

    parser.add_argument(
        '--language',
        default='java',
        help='Programming language (e.g., java, python) (default: java)'
    )

    parser.add_argument(
        '--private',
        action='store_true',
        help='Make dataset repository private'
    )

    args = parser.parse_args()

    success = upload_dataset(
        dataset_dir=args.dataset_dir,
        dataset_id=args.dataset_id,
        language=args.language,
        private=args.private
    )

    sys.exit(0 if success else 1)
