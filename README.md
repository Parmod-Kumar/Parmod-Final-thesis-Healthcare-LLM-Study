# Healthcare LLM Study

Comparing Prompt Engineering and Fine-Tuning for Medical Question Answering

## Overview

This repository contains the experimental work for a healthcare-focused LLM study comparing prompt engineering strategies and parameter-efficient fine-tuning methods for domain-specific medical question answering.

The project evaluates how different adaptation strategies affect performance on clinical QA tasks, with a focus on trade-offs among accuracy, latency, and resource efficiency.

## Research Goals

- Compare prompting strategies such as zero-shot, few-shot, and chain-of-thought prompting
- Evaluate LoRA and QLoRA fine-tuning against baseline prompting approaches
- Assess model performance across healthcare sub-domains and failure modes
- Produce reusable benchmark tables and publication-ready figures

## Models Evaluated

The project includes the following model families and variants:

- Mistral-7B
- Gemma-7B
- Phi-3-Mini
- DistilGPT2

## Prompting and Fine-Tuning Approaches

The study evaluates:

- Zero-shot prompting
- Few-shot prompting
- Chain-of-thought prompting
- Role-based prompting
- Prompt ensemble strategies
- LoRA fine-tuning
- QLoRA fine-tuning

## Evaluation Metrics

The main evaluation pipeline reports:

- F1 score
- ROUGE-1
- ROUGE-L
- BERTScore
- Exact Match (EM)
- Inference latency
- GPU memory usage
- Error breakdown by failure mode

## Repository Structure

```text
healthcare_llm_study/
├── data/                 # train, validation, and test datasets
├── src/                  # experimental scripts for loading, prompting, training, evaluation, and visualization
├── results/              # generated result CSV files
├── visualisations/       # generated plots and publication-style figures
├── models/               # model adapters/checkpoints
├── requirements.txt      # Python dependencies
├── README.md             # project documentation
└── .gitignore
```

## Setup

Create a Python environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Generate Results and Figures

Run the project scripts from the repository root:

```bash
python src/10_generate_results_csv.py
python src/08_prompting_visualizations.py
python src/09_thesis_styled_visualizations.py
```

## Output Files

- `results/` contains the main study summaries and output tables
- `all_results_csv/` contains a consolidated copy of all CSV results
- `visualisations/` contains PNG charts and publication-style figures

## Notes

This repository includes both the experimental pipeline and the generated artifacts used for the analysis. The result files are intended to support benchmarking, comparison, and thesis-style visual reporting for the healthcare LLM study.