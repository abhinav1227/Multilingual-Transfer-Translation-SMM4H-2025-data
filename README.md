# Multilingual Medical NLP Benchmark Study

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow)
![Task](https://img.shields.io/badge/Task-Multilingual%20NLP-green)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

This project investigates **multilingual and cross-lingual transfer learning for medical social media classification** using the **SMM4H 2025 dataset**.  

The objective is to evaluate how well **Transformer-based models** generalize across languages, including **zero-shot transfer** and **machine translation–based evaluation**.

---

# Project Overview

The study focuses on:

- Multilingual transfer learning
- Cross-lingual generalization
- Machine translation for data augmentation
- Error analysis in multilingual medical NLP

## Research Questions

This project investigates the following research questions:

- **Q1:** How does multilingual training compare to monolingual training for medical social media classification?

- **Q2:** Does zero-shot cross-lingual transfer significantly degrade performance?

- **Q3:** Can machine translation improve cross-lingual generalization?

- **Q4:** How do Transformer-based models behave compared to translation-based approaches?

Experiments are structured into four main stages:

1. **Language Selection & Data Preparation**
2. **Multilingual Data Exploration**
3. **Multilingual and Cross-Lingual Modeling**
4. **Evaluation and Error Analysis**

---

## Repository Structure

```text
data/
├── raw/                         # Original SMM4H dataset
├── processed/                   # Train / validation / test splits
│   ├── train_en.csv
│   ├── val_en.csv
│   ├── test_en.csv
│   ├── train_ru.csv
│   ├── val_ru.csv
│   ├── test_ru.csv
│   └── train_de.csv
│
└── translated/                  # Machine translated data

models/                          # Saved trained models

notebooks/
└── main.ipynb                   # Exploratory data analysis

pipeline/
├── preprocessing.py             # Data cleaning and splitting
├── translate.py                 # Machine translation (Mistral via Ollama)
├── model_training.py            # Model setup and evaluation
├── train.py                     # Training loops
├── tasks.py                     # Experiment setups
├── utils.py                     # Utility functions
└── main.py                      # Main experiment runner

results/                         # Evaluation outputs
scripts/                         # Optional helper scripts
.env                             # Environment variables
README.md
---
```

# Task 1.1 — Language Selection & Data Preparation

Three languages from the dataset are selected:

- **Two languages used for training**
- **One language used only for evaluation (zero-shot)**

Data preparation steps:

- Redefine the original **validation set as test set**
- Create a **new validation set sampled from training data**
- Report **dataset sizes and language distributions**
- Apply preprocessing suitable for **Transformer-based models**

Preprocessing includes:

- text normalization  
- handling noisy social media tokens  
- preparing inputs for Transformer tokenization  

Additionally, a **subset of English training data** is translated into the third language using **LLM-based machine translation**.

---

# Task 1.2 — Multilingual Data Exploration

Exploratory analysis examines multilingual characteristics of the dataset.

Key analyses:

- Label distributions per language
- Dataset size differences
- Linguistic characteristics relevant to the task

Manual inspection is conducted to identify:

- translation artifacts  
- language-specific phenomena  
- annotation ambiguities  

Findings help guide model design and training strategies.

---

# Task 1.3 — Multilingual & Cross-Lingual Modeling

Transformer models are trained under multiple experimental setups.

## 1. Monolingual Models
- Train one model per training language
- Evaluate on the corresponding test language

## 2. Multilingual Model
- Train a single model on combined multilingual data
- Evaluate on:
  - seen languages
  - unseen zero-shot language

## 3. Machine Translation–Based Evaluation

Models are trained using:

- English-only data
- Multilingual data
- Machine-translated data

All models are evaluated on the **test set of the third language**.

The same **Transformer architecture** is used across setups where possible.

---

# Task 1.4 — Evaluation & Error Analysis

Evaluation includes standard classification metrics.

Comparisons are made across:

- Monolingual vs multilingual training
- Seen vs unseen (zero-shot) languages
- Native-language vs machine-translated evaluation

Cross-lingual error analysis investigates:

- language-specific failure cases
- translation-induced errors
- robustness differences across languages

The analysis also discusses:

- when multilingual training helps or hurts
- limitations of cross-lingual transfer in medical social media
- implications for multilingual medical NLP systems

---

# Setup

## 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

## 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## 3. Run Experiments
```bash
python pipeline/main.py
```

# Translation

Translation experiments were conducted using LLM-based machine translation via Mistral (Ollama).
Translated samples were stored for reproducibility and used for controlled experimental comparison.

Make sure Ollama is running locally before executing translation scripts.

# Experimental Results

## Quantitative Comparison
```text

| Setup                  | Accuracy | Precision | Recall | F1 Score | Loss  |
|------------------------|----------|----------|--------|--------|--------|
| EN Monolingual         | 0.952    | 0.949    | 0.952  | 0.950  | 0.215 |
| RU Monolingual         | 0.918    | 0.908    | 0.918  | 0.911  | 0.326 |
| Multilingual (EN)       | 0.948    | 0.945    | 0.948  | 0.947  | 0.228 |
| Multilingual (RU)       | 0.918    | 0.911    | 0.918  | 0.913  | 0.338 |
| Zero-Shot (DE)          | 0.841    | 0.902    | 0.841  | 0.869  | 0.555 |
| EN → DE (Zero-Shot)     | 0.940    | 0.892    | 0.940  | 0.916  | 0.386 |
| Multi → DE (Zero-Shot)  | 0.841    | 0.902    | 0.841  | 0.869  | 0.555 |
| DE Translated           | 0.658    | 0.893    | 0.658  | 0.751  | 0.849 |

---
```

## Discussion of Results

### Q1: Multilingual vs Monolingual Training

Multilingual training performs comparably to monolingual training for seen languages.

- Performance drop is small for EN and RU.
- In some cases, multilingual training slightly improves robustness.
- Conclusion: Multilingual training does not hurt performance significantly and provides better generalization capability.

---

### Q2: Zero-Shot Cross-Lingual Transfer

Zero-shot performance on German (DE) shows a noticeable drop compared to supervised training.

- Accuracy decreases compared to English/Russian training.
- Loss increases significantly in zero-shot setups.
- However, performance remains relatively strong, indicating effective cross-lingual transfer.

Conclusion: Zero-shot transfer works reasonably but introduces performance degradation.

---

### Q3: Effect of Machine Translation

Training on translated data (DE translated from EN) results in lower performance compared to native-language training.

- Translation introduces noise and artifacts.
- The translated dataset performs significantly worse than monolingual and multilingual setups.
- This suggests that translation-based data augmentation alone is insufficient.

---

### Q4: Transformer vs Translation-Based Approaches

Transformer-based multilingual training clearly outperforms the pure translation-based approach.

Key observation:

- Native multilingual modeling is more robust.
- Machine translation adds limited benefit and may introduce semantic distortion.

---

## Overall Conclusion

- Multilingual training is the most stable approach.
- Zero-shot transfer is feasible but not optimal.
- Machine translation improves data coverage but reduces quality.
- Direct multilingual modeling is preferable for medical social media NLP.
