# Multilingual Transfer for Medical Social Media NLP (SMM4H 2025)

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

Experiments are structured into four main stages:

1. **Language Selection & Data Preparation**
2. **Multilingual Data Exploration**
3. **Multilingual and Cross-Lingual Modeling**
4. **Evaluation and Error Analysis**

---

# Repository Structure
data/
├── raw/ # Original SMM4H dataset
├── processed/ # Train / validation / test splits
│ ├── train_en.csv
│ ├── val_en.csv
│ ├── test_en.csv
│ ├── train_ru.csv
│ ├── val_ru.csv
│ ├── test_ru.csv
│ └── train_de.csv
└── translated/ # Machine translated data

models/ # Saved trained models

notebooks/
└── main.ipynb # Exploratory data analysis

pipeline/
├── preprocessing.py # Data cleaning and splitting
├── translate.py # Machine translation (Mistral via Ollama)
├── model_training.py # Model setup and evaluation
├── train.py # Training loops
├── tasks.py # Experiment setups
├── utils.py # Utility functions
└── main.py # Main experiment runner

results/ # Evaluation outputs
scripts/ # Optional helper scripts
.env # Environment variables
README.md

---

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

## 2. Install Dependencies
```bash
pip install -r requirements.txt
