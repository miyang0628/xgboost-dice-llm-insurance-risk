# xgboost-dice-llm-insurance-risk

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10.15-blue.svg)](https://www.python.org/)
[![Under Review](https://img.shields.io/badge/Status-Under%20Review-lightgrey.svg)](https://github.com/miyang0628/xgboost-dice-llm-insurance-risk)

> **From Risk Score to Risk Narrative: Counterfactual XAI and LLM-Based Health Risk Communication for Insurance Underwriting**  
> *manuscript under review, Journal of Risk and Insurance*

---

## Overview

This repository contains the full analysis code for an end-to-end **XGBoost–DiCE–LLM pipeline** for diabetes risk stratification in insurance underwriting. Applied to sex-stratified data from the Korea National Health and Nutrition Examination Survey (KNHANES 2020–2024; *n* = 13,437), the pipeline generates individualised counterfactual scenarios and converts them automatically into natural-language consultation guides, evaluated via a DISCERN-adapted LLM-as-a-Judge framework employing three expert personas.

```
Raw health survey data (KNHANES 2020–2024)
  → XGBoost risk scoring  (AUC: Male 0.836 / Female 0.877)
  → Lift Chart high-risk identification  (top 40%; Lift ≥ 2.0)
  → DiCE counterfactual explanation  (non-diabetic high-risk: Male n=834 / Female n=1,655)
  → LLM consultation guide generation  (GPT-4o-mini; 40 guides, $0.015 USD)
  → LLM-as-a-Judge quality evaluation  (3 personas × 7 dimensions; mean score 4.00/5.00)
```

---

## Key Results

| Component | Male | Female |
|---|---|---|
| Final model | XGBoost | XGBoost |
| AUC | 0.836 | 0.877 |
| PR-AUC | 0.620 | 0.581 |
| Brier Score | 0.167 | 0.131 |
| 5-fold CV F1 | 0.669 ± 0.012 | 0.628 ± 0.015 |
| High-risk boundary | Top 40% (Lift = 2.052) | Top 40% (Lift = 2.423) |
| Non-diabetic high-risk *n* | 834 | 1,655 |

**LLM-as-a-Judge Evaluation** (mean ± SD across 40 cases, scale 1–5):

| Dimension | Medical Expert | Insurance Specialist | General Public |
|---|---|---|---|
| D1: Medical Accuracy | 4.60 ± 0.50 | 3.00 ± 0.00 | 3.05 ± 0.32 |
| D2: Assoc./Causation Dist. | 5.00 ± 0.00 | 4.92 ± 0.27 | 4.75 ± 0.44 |
| D3: Actionability | 4.00 ± 0.00 | 4.00 ± 0.00 | 4.00 ± 0.00 |
| D4: Readability & Clarity | 4.12 ± 0.33 | 4.08 ± 0.27 | 4.00 ± 0.00 |
| D5: Disclaimer Adequacy | 5.00 ± 0.00 | 5.00 ± 0.00 | 5.00 ± 0.00 |
| D6: Insurance Relevance | 3.00 ± 0.00 | 4.00 ± 0.00 | 3.00 ± 0.00 |
| D7: Overall Quality | 4.00 ± 0.00 | 4.00 ± 0.00 | 4.00 ± 0.00 |
| **Mean (D1–D7)** | **4.25** | **4.14** | **3.97** |

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
- Study population: Adults aged 19+; *n* = 13,437 after listwise deletion (Male 5,422 / Female 8,015)
- Outcome: Diabetes status (HbA1c level + physician diagnosis)
- Predictors: 26 variables — anthropometric, dietary, lifestyle, and socioeconomic indicators

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

| Package | Version | Reference |
|---|---|---|
| xgboost | 1.7.5 | Chen & Guestrin (2016) |
| catboost | 1.2.8 | Prokhorenkova et al. (2018) |
| lightgbm | 4.6.0 | Ke et al. (2017) |
| scikit-learn | 1.5.2 | |
| optuna | 4.5.0 | Akiba et al. (2019) |
| dice-ml | 0.11 | Mothilal et al. (2020) |
| openai | 1.55.1 | |
| shap | 0.46.0 | |
| pandas | 2.3.3 | |
| numpy | 1.26.4 | |
| scipy | 1.14.1 | |
| matplotlib | 3.9.2 | |

---

## API Key Setup

The LLM generation and evaluation steps (notebooks 10–11) require an OpenAI API key.

```bash
cp .env.example .env
# Edit .env and add your key:
# OPENAI_API_KEY=sk-...
```

**API cost summary:**
- Guide generation (40 cases): **$0.015 USD**
- LLM-as-a-Judge evaluation (120 calls, 152,172 tokens): **$0.033 USD**

---

## Reproducibility

All stochastic components use `random_state=42`:

- XGBoost model training
- Optuna hyperparameter search (100 trials, 5-fold stratified CV)
- DiCE counterfactual sampling

LLM API settings:
- Guide generation: `temperature=0.3`, `max_tokens=1200`
- LLM-as-a-Judge: `temperature=0.0` (maximise scoring determinism)

**Computational environment:**

| Component | Specification |
|---|---|
| OS | Windows 10 (build 19045) |
| CPU | Intel Core i5-14500 (14 cores, 20 threads, 2.60 GHz base) |
| RAM | 128 GB |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| Python | 3.10.15 (Miniconda) |
| CUDA | 11.7 |

> GPU was used for Neural Network training only; all other models ran on CPU.

---

## Candidate On-Premise Deployment (Future Work)

The current proof-of-concept uses the OpenAI cloud API. A GDPR-compliant production pathway based on **Meerkat-8B** ([Kim et al. 2025](https://doi.org/10.1038/s41746-025-01616-9)) — a medical LLM fine-tuned from Llama-3-8B-Instruct, surpassing GPT-3.5 on MedQA — is described in Appendix A of the manuscript. Migration requires only a single-line change in client initialisation; all prompt engineering and pipeline logic remain intact.

| Component | Specification |
|---|---|
| Base model | Llama-3-8B-Instruct |
| Fine-tuned model | Meerkat-8B v1.0 |
| Quantisation | 4-bit GPTQ |
| Inference engine | vLLM 0.8.4 |
| GPU | NVIDIA RTX 4060Ti 16 GB |
| Data boundary | Institutional server only (GDPR) |

---

## Citation

If you use this code, please cite:

```bibtex
@article{anon2025riskscore,
  title   = {From Risk Score to Risk Narrative: Counterfactual {XAI}
             and {LLM}-Based Health Risk Communication for Insurance Underwriting},
  author  = {Anonymous},
  journal = {manuscript under review},
  year    = {2025}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
