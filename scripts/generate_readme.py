#!/usr/bin/env python
"""
Generate README for HuggingFace model repository
"""
import json
import argparse
from pathlib import Path
from datetime import datetime


def generate_readme(
    hub_model_id: str,
    metrics_file: str,
    model_name: str = "CodeGen",
    language: str = "java",
    output_file: str = "README.md"
):
    """Generate README with training results"""

    # Load metrics if exists
    metrics = {}
    if metrics_file and Path(metrics_file).exists():
        with open(metrics_file) as f:
            metrics = json.load(f)

    # Extract run name from hub_model_id
    run_name = hub_model_id.split('/')[-1]

    readme = f"""---
language: {language}
tags:
- code-generation
- class-naming
- {model_name.lower()}
license: apache-2.0
---

# {model_name} {language.title()} Class Name Prediction

Fine-tuned {model_name} model for {language.title()} class name prediction from code snippets.

## Model Details

- **Base Model:** Salesforce/codegen-350M-mono
- **Task:** Class name prediction from masked code
- **Language:** {language.title()}
- **Fine-tuning Dataset:** GitHub repositories
"""

    # Add metrics if available
    if metrics:
        readme += f"""
## Evaluation Results

| Metric | Value |
|--------|-------|
| **Exact Match** | {metrics.get('exact_match', 0):.2%} |
| **Case-Insensitive Match** | {metrics.get('exact_match_case_insensitive', 0):.2%} |
| **Top-{metrics.get('k', 5)} Accuracy** | {metrics.get('topk_accuracy', 0):.2%} |
| **Avg Levenshtein Distance** | {metrics.get('avg_levenshtein', 0):.2f} |
| **Samples Evaluated** | {metrics.get('n', 0):,} |
"""

        if 'elapsed_time_seconds' in metrics:
            elapsed_min = metrics['elapsed_time_seconds'] / 60
            readme += f"| **Evaluation Time** | {elapsed_min:.1f} minutes |\n"
            readme += f"| **Speed** | {metrics.get('samples_per_second', 0):.2f} samples/sec |\n"

    readme += """
## Training Configuration

- **Model:** Salesforce/codegen-350M-mono
- **Batch Size:** 12
- **Gradient Accumulation:** 3
- **Effective Batch Size:** 36
- **Learning Rate:** 5e-5
- **Epochs:** 5
- **Max Length:** 1024
- **Gradient Checkpointing:** Yes
- **FP16:** Yes

## Usage

### Load Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("__HUB_MODEL_ID__")
tokenizer = AutoTokenizer.from_pretrained("__HUB_MODEL_ID__")
```

### Predict Class Name

```python
# Example Java code with masked class name
code = \"\"\"
public class ____ {
    private String name;

    public void setName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }
}
Class name:\"\"\"

# Generate prediction
inputs = tokenizer(code, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=10,
    num_beams=5,
    num_return_sequences=5
)

# Decode predictions
predictions = [tokenizer.decode(output, skip_special_tokens=True)
               for output in outputs]
print("Top predictions:", predictions)
```

## Files in This Repository

- `pytorch_model.bin` - Model weights
- `config.json` - Model configuration
- `tokenizer_config.json` - Tokenizer configuration
- `logs/training.log` - Full training logs
- `metrics/metrics.json` - Evaluation metrics
- `metrics/detailed_results.jsonl` - Per-sample predictions

## Project Repository

Full source code: [https://github.com/reiprasetyastudy-repo/classnameprediction](https://github.com/reiprasetyastudy-repo/classnameprediction)

## Citation

```bibtex
@misc{__RUN_NAME__,
  author = {Rei Prasetya},
  title = {__MODEL_NAME__ for __LANGUAGE__ Class Name Prediction},
  year = {__YEAR__},
  publisher = {HuggingFace},
  url = {https://huggingface.co/__HUB_MODEL_ID__}
}
```

## License

Apache 2.0

---

*Generated on __DATE__*
"""

    # Replace placeholders
    readme = readme.replace('__HUB_MODEL_ID__', hub_model_id)
    readme = readme.replace('__RUN_NAME__', run_name)
    readme = readme.replace('__MODEL_NAME__', model_name)
    readme = readme.replace('__LANGUAGE__', language.title())
    readme = readme.replace('__YEAR__', str(datetime.now().year))
    readme = readme.replace('__DATE__', datetime.now().strftime('%Y-%m-%d'))

    # Write README
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(readme)

    print(f"✅ README generated: {output_file}")
    return output_file


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate README for HuggingFace model repository")
    ap.add_argument('--hub-model-id', required=True, help='HuggingFace model ID (e.g., username/model-name)')
    ap.add_argument('--metrics', required=True, help='Path to metrics.json file')
    ap.add_argument('--model-name', default='CodeGen', help='Model name (default: CodeGen)')
    ap.add_argument('--language', default='java', help='Programming language (default: java)')
    ap.add_argument('--output', default='README.md', help='Output README file (default: README.md)')

    args = ap.parse_args()

    generate_readme(
        hub_model_id=args.hub_model_id,
        metrics_file=args.metrics,
        model_name=args.model_name,
        language=args.language,
        output_file=args.output
    )
