# ESG Metrics Extractor

This tool extracts Environmental, Social, and Governance (ESG) metrics from sustainability reports using offline-capable vision-language AI models.

## Features

- Offline processing using local models
- Multimodal parsing of text and visual elements (tables, charts)
- Structured output of ESG metrics
- Performance evaluation against ground truth
- Resource usage tracking

## Installation

### On Linux/macOS:

```bash
chmod +x setup.sh
./setup.sh
source esg_env/bin/activate
```

### On Windows:

```powershell
.\setup.ps1
```

## Usage

### Basic Usage

```bash
python run_esg_extraction.py --pdf your_report.pdf --output results.json
```

### With Evaluation

```bash
python run_esg_extraction.py --pdf your_report.pdf --ground_truth sample_ground_truth.json --output results.json
```

### Options

- `--pdf`: Path to the PDF file (required)
- `--output`: Output JSON file path (default: esg_metrics.json)
- `--ground_truth`: Path to ground truth JSON for evaluation (optional)
- `--max_pages`: Maximum number of pages to process (optional)
- `--batch_size`: Batch size for processing pages (default: 4)

## Ground Truth Format

Create a JSON file with this structure:

```json
{
  "emissions": {
    "scope 1": [25000, "25,000 tCO2e"],
    "scope 2": [12500, "12,500 tCO2e"]
  },
  "energy": {
    "renewable": ["35%", "35", "35 percent"]
  }
}
```

## ESG Schema

The tool extracts metrics across these categories:

- Emissions: scope 1, scope 2, scope 3, carbon footprint, ghg
- Energy: renewable, consumption, efficiency, kwh, mwh
- Water: consumption, recycled, discharge, withdrawal
- Waste: recycled, landfill, hazardous, non-hazardous
- Diversity: gender, ethnicity, inclusion, equality

## Output Format

The tool produces a JSON file with:

1. Page-level results
2. Consolidated metrics
3. Performance statistics
4. Evaluation metrics (if ground truth provided)

## Requirements

- Python 3.8+
- PyTorch
- Transformers library
- PyMuPDF
- Hugging Face Hub

## Offline Usage

The setup script pre-downloads the model for offline use. To manually download:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="ibm-granite/granite-vision-3.3-2b")
```

## Fine-tuning (Future Development)

At present, the MVP model runs inference using a pre-trained vision-language backbone without domain-specific fine-tuning on ESG reports. In the next phase (P2), the model will be fine-tuned on a curated dataset of ESG documents to better capture sustainability-specific terminology, layout patterns, and metric schemas. The goal is to improve extraction accuracy for both textual and visual elements.

Once fine-tuning is complete, the enhanced model will be tested against the current baseline using standard evaluation metrics (precision, recall, F1-score) to quantify performance improvements and demonstrate the value of ESG-specific adaptation.
