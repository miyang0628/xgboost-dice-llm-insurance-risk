# xgboost-dice-llm-insurance-risk

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10.15-blue.svg)](https://www.python.org/)
[![Under Review](https://img.shields.io/badge/Status-Under%20Review-lightgrey.svg)](https://github.com/anonymous/xgboost-dice-llm-insurance-risk)

> **An Integrated XGBoost–DiCE–LLM Framework for Explainable Health Risk Communication in Insurance Underwriting**
> *Anonymous authors — manuscript under review*

---

## Overview

This repository contains the full analysis code for an end-to-end **XGBoost–DiCE–LLM pipeline** for diabetes risk stratification in insurance underwriting. Applied to sex-stratified data from the Korea National Health and Nutrition Examination Survey (KNHANES 2020–2024; *n* = 13,437), the pipeline generates individualised counterfactual scenarios and converts them automatically into natural-language consultation guides, evaluated via a DISCERN-adapted LLM-as-a-Judge framework employing three expert personas.

```
Raw health survey data (KNHANES 2020–2024)
  → XGBoost risk scoring          (AUC: Male 0.836 / Female 0.877)
  → Lift Chart high-risk ID       (top 40%; Lift ≥ 2.0)
  → DiCE counterfactual           (non-diabetic high-risk: Male n=834 / Female n=1,655)
  → LLM consultation guide        (GPT-4o-mini; 40 guides, $0.015 USD)
  → LLM-as-a-Judge evaluation     (3 personas × 7 dimensions; mean 4.00/5.00)
  → Multi-model judge validation  (GPT-4o independent replication; D1 r=0.860)
  → SHAP vs DiCE comparison       (D3: DiCE 4.00 vs SHAP 3.27, p<.001)
```

---

## Key Results

### XGBoost Risk Classifier

| Component | Male | Female |
|---|---|---|
| Final model | XGBoost | XGBoost |
| AUC | 0.836 | 0.877 |
| PR-AUC | 0.620 | 0.581 |
| Brier Score | 0.167 | 0.131 |
| 5-fold CV F1 | 0.669 ± 0.012 | 0.628 ± 0.015 |
| High-risk boundary | Top 40% (Lift = 2.052) | Top 40% (Lift = 2.423) |
| Non-diabetic high-risk *n* | 834 | 1,655 |

### LLM-as-a-Judge Evaluation (mean ± SD, scale 1–5)

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

### Multi-Model Judge Validation (GPT-4o-mini vs GPT-4o)

| Dimension | GPT-4o-mini M±SD | GPT-4o M±SD | Pearson r | Agreement |
|---|---|---|---|---|
| D1: Medical Accuracy | 3.55 ± 0.82 | 3.29 ± 0.47 | **0.860***  | Strong |
| D2: Assoc./Causation | 4.89 ± 0.31 | 4.87 ± 0.34 | 0.179 | — |
| D3: Actionability | 4.00 ± 0.00 | 3.50 ± 0.57 | — | Constant (mini) |
| D4: Readability | 4.07 ± 0.25 | 4.14 ± 0.37 | 0.258 | — |
| D5: Disclaimer | 5.00 ± 0.00 | 5.00 ± 0.00 | — | Perfect Consensus |
| D6: Insurance Relevance | 3.33 ± 0.47 | 3.39 ± 0.69 | 0.626** | Moderate |
| D7: Overall Quality | 4.00 ± 0.00 | 3.77 ± 0.44 | — | Constant (mini) |

\* *p* < .001 &nbsp; \*\* *p* < .01 &nbsp; — = SD=0 in mini (not disagreement)

### SHAP vs DiCE Comparison (n = 40 cases)

| Criterion | SHAP | DiCE | Advantage |
|---|---|---|---|
| Provides target value | No (direction only) | Yes (value + delta) | DiCE |
| D3 Actionability (M±SD) | 3.27 ± 0.45 | 4.00 ± 0.00 | **DiCE*** |
| D4 Readability (M±SD) | 4.38 ± 0.49 | 4.04 ± 0.19 | SHAP*** |
| Feature rank overlap (Jaccard) | 0.000 ± 0.000 | Reference | — |
| Diverse scenarios | Single ranking | Multiple CFs (n=4) | DiCE |
| GDPR recourse support | Partial | Full | DiCE |

\* *p* < .001 (paired *t*-test)

---

## Repository Structure

```
xgboost-dice-llm-insurance-risk/
│
├── data/
│   └── README_data.md              # KNHANES download instructions
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
├── 11_llm_as_judge_evaluation.ipynb
├── 12_multi_llm_judge.ipynb        # NEW: GPT-4o independent judge replication
├── 13_shap_vs_dice_comparison.ipynb # NEW: SHAP vs DiCE comparative analysis
├── 14_visualization_dashboard.ipynb
│
├── outputs/
│   ├── models_male/
│   ├── models_female/
│   ├── dice_male/
│   ├── dice_female/
│   ├── judge_eval/
│   ├── multi_judge/                # NEW: cross-model validation results
│   ├── shap_vs_dice/               # NEW: SHAP vs DiCE comparison results
│   └── llm_outputs/
│
├── catboost_info/
├── requirements.txt
├── .env.example
└── README.md
```

---

## Data

This study uses the **Korea National Health and Nutrition Examination Survey (KNHANES) 2020–2024**, administered by the Korea Disease Control and Prevention Agency (KDCA).

- **Download:** [https://knhanes.kdca.go.kr](https://knhanes.kdca.go.kr)
- **Access:** Free registration required; data are de-identified and publicly available
- **Study population:** Adults aged 19+; *n* = 13,437 after listwise deletion (Male 5,422 / Female 8,015)
- **Outcome:** Diabetes status (HbA1c level + physician diagnosis)
- **Predictors:** 26 variables — anthropometric, dietary, lifestyle, and socioeconomic indicators

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
| xgboost | 3.2.0 |
| catboost | 1.2.10 |
| lightgbm | 4.6.0 |
| scikit-learn | 1.7.2 |
| optuna | 4.9.0 |
| dice-ml | 0.11 |
| openai | 1.55.1 |
| pandas | 2.3.3 |
| numpy | 1.26.4 |
| scipy | 1.14.1 |
| matplotlib | 3.9.2 |

---

## API Key Setup

Notebooks 10–13 require an OpenAI API key.

```bash
cp .env.example .env
# Edit .env and add your key:
# OPENAI_API_KEY=sk-...
```

**API cost summary:**

| Task | Model | Calls | Est. Cost |
|---|---|---|---|
| Guide generation | GPT-4o-mini | 40 | $0.015 |
| LLM-as-a-Judge (original) | GPT-4o-mini | 120 | $0.033 |
| Multi-model judge validation | GPT-4o | 120 | $0.549 |
| SHAP guide generation | GPT-4o-mini | 40 | $0.004 |
| SHAP judge evaluation | GPT-4o-mini | 80 | ~$0.010 |
| **Total** | | **400** | **~$0.611** |

---

## Reproducibility

All stochastic components use `random_state=42`:
- XGBoost model training
- Optuna hyperparameter search (100 trials, 5-fold stratified CV)
- DiCE counterfactual sampling

LLM API settings:
- Guide generation: `temperature=0.3`, `max_tokens=1200`
- LLM-as-a-Judge: `temperature=0.0` (deterministic scoring)
- Multi-model judge: `temperature=0.0` (identical to original)

**Computational environment:**

| Component | Specification |
|---|---|
| OS | Windows 10 (build 19045) |
| CPU | Intel Core i5-14500 (14 cores, 2.60 GHz) |
| RAM | 128 GB |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| Python | 3.10.15 (Miniconda) |
| CUDA | 11.7 |

> GPU was used for Neural Network training only; all other models ran on CPU.

---

## Notebook Execution Order

```
01 → 02 → 03 → 04 → 05 → 06        # Data & modeling
→ 07 → 08 → 09                       # DiCE & robustness
→ 10 → 11                            # LLM generation & evaluation
→ 12                                 # Multi-model judge validation (NEW)
→ 13                                 # SHAP vs DiCE comparison (NEW)
→ 14                                 # Visualization dashboard
```

**Note on notebooks 03/04:** These notebooks require xgboost 3.x compatibility.
The `use_label_encoder` and `eval_metric` constructor parameters have been removed.
CatBoost cross-validation uses a manual fold loop to avoid sklearn 1.7.x cloning issues.

---

## Candidate On-Premise Deployment (Future Work)

A GDPR-compliant production pathway based on **Meerkat-8B** — a medical LLM fine-tuned from Llama-3-8B-Instruct, surpassing GPT-3.5 on MedQA — is described in Appendix D of the manuscript. Migration requires only a single-line change in client initialisation; all prompt engineering and pipeline logic remain intact.

| Component | Specification | Role |
|---|---|---|
| Base model | Llama-3-8B-Instruct | Foundation LLM |
| Fine-tuned model | Meerkat-8B v1.0 | Medical reasoning |
| Quantisation | 4-bit GPTQ | Memory efficiency |
| Inference engine | vLLM 0.8.4 | API serving |
| GPU | NVIDIA RTX 4060Ti 16 GB | Inference |
| Data boundary | Institutional server only | GDPR compliance |

---

## Citation

> Anonymous authors. (under review). An Integrated XGBoost–DiCE–LLM Framework
> for Explainable Health Risk Communication in Insurance Underwriting.
> *Expert Systems with Applications.*

```bibtex
@article{anonymous2025xgboost,
  title   = {An Integrated {XGBoost--DiCE--LLM} Framework for Explainable
             Health Risk Communication in Insurance Underwriting},
  author  = {Anonymous},
  journal = {Expert Systems with Applications},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
