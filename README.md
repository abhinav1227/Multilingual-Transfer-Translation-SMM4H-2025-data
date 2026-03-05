# Advanced NLP Project

This repository contains the code for the Advanced NLP course project, consisting of two tasks:

- **Task 1**: Multilingual Transfer and Translation in Medical Social Media NLP (SMM4H 2025 data)

## Repository Structure
.
├── data/ # Task 1 datasets
│ ├── raw/ # Original SMM4H data (train.csv, test.csv)
│ ├── processed/ # Preprocessed train/val/test splits per language
│ │ ├── train_en.csv, val_en.csv, test_en.csv
│ │ ├── train_ru.csv, val_ru.csv, test_ru.csv
│ │ └── train_de.csv, val_de.csv, test_de.csv
│ └── translated/ # Machine-translated German data (translated_train_de.csv)
│
├── models/ # Saved fine-tuned models for Task 1
│
├── notebooks/ # Jupyter notebook for exploratory analysis (main.ipynb)
│
├── pipeline/ # Core Python modules for Task 1
│ ├── preprocessing.py # Data loading, cleaning, splitting(1.1)
│ ├── translate.py # Translation using Mistral via Ollama(1.1)
│ ├── model_training.py # Training and evaluation functions 
│ ├── train.py # Training loop and utilities
│ ├── tasks.py # Task-specific experiment setups (1.3.1, 1.3.2, 1.3.3)
│ ├── main.py # Entry point to run all experiments
│ └── utils.py # Helper functions
│
├── results/ # Output evaluation results (JSON files)
│
├── scripts/ # Additional helper scripts (optional)
│
├── .env # Environment variables (e.g., API keys)
└── README.md # This file



## Task 1: Multilingual Transfer (SMM4H Data)

All Task 1 code is in the `pipeline/` directory.  
- `main.py` runs all three experimental setups (monolingual, multilingual, translation-based).  
- `preprocessing.py` handles data preparation and splitting.  
- `translate.py` performs translation using Mistral (Ollama).  
- `model_training.py` and `train.py` manage model training and evaluation.  
- `notebooks/main.ipynb` contains exploratory data analysis and visualizations.


## How to Run(ensure Ollama is available for data translation)

1. # Make and Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

2. python pipeline/main.py