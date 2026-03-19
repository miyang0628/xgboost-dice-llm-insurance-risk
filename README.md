# xgboost-dice-llm-insurance-risk

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10.15-blue.svg)](https://www.python.org/)
[![Under Review](https://img.shields.io/badge/Status-Manuscript%20in%20Preparation-lightgrey.svg)]()

> **From Risk Score to Risk Narrative: Counterfactual XAI and LLM Guides for Proactive Insurance Risk Communication**  
> Munil Yang, Heuiju Chun — *manuscript in preparation*

---

## Overview

This repository contains the full analysis code for an end-to-end **XGBoost–DiCE–LLM pipeline** for diabetes risk stratification in insurance underwriting. Applied to sex-stratified data from the Korea National Health and Nutrition Examination Survey (KNHANES 2020–2024; *n* = 13,437), the pipeline generates individualised counterfactual scenarios and converts them automatically into natural-language consultation guides evaluated via a DISCERN-adapted LLM-as-a-Judge framework.

```
Raw health survey data (KNHANES)
  → XGBoost risk scoring
  → Lift Chart high-risk identification
  → DiCE counterfactual explanation
  → LLM consultation guide generation (GPT-4o-mini)
  → LLM-as-a-Judge quality evaluation
```

---

## Repository Structure

```
xgboost-dice-llm-insurance-risk/
│
├── data/                          # Data directory (see Data section below)
│   └── README_data.md             # KNHANES download instructions
│
├── 01_data_preprocessing.ipynb
├── 02_eda.ipynb
├── 03_model_training_male.ipynb
├── 04_model_training_female.ipynb
├── 05_model_evaluation.ipynb
├── 06_lift_chart_highrisk.ipynb
├── 07_dice_counterfactual_male.ipynb
├── 08_dice_counterfactual_female.ipynb
├── 09_robustness_analysis.ipynb
├── 10_llm_prompt_engineering.ipynb
├── 11_llm_as_judge.ipynb
├── 12_visualization_dashboard.ipynb
│
├── outputs/                       # Generated figures and result files
├── catboost_info/                 # CatBoost training logs (auto-generated)
├── requirements.txt
├── .env.example                   # API key template
└── README.md
```

---

## Data

This study uses the **Korea National Health and Nutrition Examination Survey (KNHANES) 2020–2024**, administered by the Korea Disease Control and Prevention Agency (KDCA).

- Download: [https://knhanes.kdca.go.kr](https://knhanes.kdca.go.kr)
- Access: Free registration required; data are de-identified and publicly available
- After downloading, place the raw files in the `data/` directory

> **Note:** Raw data files are not included in this repository due to the KDCA data use agreement.

---

## Requirements

```bash
# Create and activate a conda environment
conda create -n diceml python=3.10.15
conda activate diceml

# Install dependencies
pip install -r requirements.txt
```

Key dependencies:

| Package | Version |
|---|---|
| xgboost | 1.7.5 |
| catboost | 1.2.8 |
| lightgbm | 4.6.0 |
| scikit-learn | 1.5.2 |
| optuna | 4.5.0 |
| dice-ml | 0.11 |
| openai | 1.55.1 |
| shap | 0.46.0 |
| pandas | 2.3.3 |
| numpy | 1.26.4 |
| scipy | 1.14.1 |
| matplotlib | 3.9.2 |

---

## API Key Setup

The LLM generation and evaluation steps (notebooks 10–11) require an OpenAI API key.

```bash
cp .env.example .env
# Edit .env and add your key:
# OPENAI_API_KEY=sk-...
```

---

## Reproducibility

All stochastic components use `random_state=42`:

- XGBoost model training
- Optuna hyperparameter search
- DiCE counterfactual sampling

LLM API settings:
- Guide generation: `temperature=0.3`, `max_tokens=1200`
- LLM-as-a-Judge: `temperature=0.0`

**Computational environment:**

| Component | Specification |
|---|---|
| OS | Windows 10 (build 19045) |
| CPU | Intel Core i5-14500 (14 cores, 20 threads) |
| RAM | 128 GB |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| Python | 3.10.15 (Miniconda) |
| CUDA | 11.7 |

---

## Citation

If you use this code, please cite:

```bibtex
@article{yang2025xai,
  title   = {From Risk Score to Risk Narrative: Counterfactual {XAI}
             and {LLM} Guides for Proactive Insurance Risk Communication},
  author  = {Yang, Munil and Chun, Heuiju},
  journal = {manuscript in preparation},
  year    = {2025}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
