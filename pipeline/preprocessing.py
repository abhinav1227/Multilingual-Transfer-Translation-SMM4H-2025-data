"""
Data preprocessing module for SMM4H multilingual modeling
"""
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import re
from sklearn.model_selection import train_test_split
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, model_name="bert-base-multilingual-cased", target_languages=['en', 'ru', 'de']):
        """
        Initialize preprocessor for specific target languages
        
        Args:
            model_name: Transformer model name for tokenizer
            target_languages: List of language codes to filter
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = 128
        self.target_languages = target_languages
        
    def filter_by_language(self, df, language_column='language'):
        """Filter DataFrame to include only target languages"""
        if language_column not in df.columns:
            print(f"Warning: Language column '{language_column}' not found. Using all data.")
            return df
        
        original_size = len(df)
        filtered_df = df[df[language_column].isin(self.target_languages)].copy()
        
        print(f"Filtered from {original_size} to {len(filtered_df)} samples")
        print(f"Language distribution:")
        print(filtered_df[language_column].value_counts())
        
        return filtered_df
    
    def split_train_val(self, df, val_size=0.2, stratify_column='label', random_state=42):
        """Split data into train and validation sets"""
        if stratify_column in df.columns:
            train_df, val_df = train_test_split(
                df, 
                test_size=val_size, 
                stratify=df[stratify_column],
                random_state=random_state
            )
        else:
            train_df, val_df = train_test_split(
                df, 
                test_size=val_size, 
                random_state=random_state
            )
        
        print(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples")
        return train_df, val_df
    
    def should_process_data(self, output_dir):
        """Check if data needs to be processed by verifying if all language files exist."""
        for lang in self.target_languages:
            train_file = f"{output_dir}train_{lang}.csv"
            val_file = f"{output_dir}val_{lang}.csv"
            
            if not (os.path.exists(train_file) and os.path.exists(val_file)):
                return False
        return True

    def load_processed_data(self, output_dir):
        """Load previously processed data from disk."""
        print("Loading previously processed data...")
        datasets = {}
        for lang in self.target_languages:
            train_file = f"{output_dir}train_{lang}.csv"
            val_file = f"{output_dir}val_{lang}.csv"
            test_file = f"{output_dir}test_{lang}.csv"
            
            train_split = pd.read_csv(train_file) if os.path.exists(train_file) else pd.DataFrame()
            val_split = pd.read_csv(val_file) if os.path.exists(val_file) else pd.DataFrame()
            test_lang = pd.read_csv(test_file) if os.path.exists(test_file) else pd.DataFrame()
            
            datasets[lang] = {
                'train': train_split,
                'val': val_split,
                'test': test_lang
            }
        
        print(f"Loaded processed data for languages: {list(datasets.keys())}")
        return datasets


    def process_raw_data(self, train_path, test_path, output_dir='data/processed/'):
        """Process raw data: filter languages and split train/val"""
        os.makedirs(output_dir, exist_ok=True)

        # Check if data has already been processed
        if self.should_process_data(output_dir):
            return self.load_processed_data(output_dir)
        
        # Load raw data
        print("Loading raw data...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Original train shape: {train_df.shape}, test shape: {test_df.shape}")
        
        # Filter by target languages
        print("\nFiltering by target languages...")
        train_filtered = self.filter_by_language(train_df)
        test_filtered = self.filter_by_language(test_df)
        
        # Process each language
        datasets = {}
        for lang in self.target_languages:
            print(f"\nProcessing {lang}...")
            
            # Filter for current language
            train_lang = train_filtered[train_filtered['language'] == lang].copy()
            test_lang = test_filtered[test_filtered['language'] == lang].copy()
            
            if len(train_lang) > 0:
                # Split train into train and validation
                train_split, val_split = self.split_train_val(train_lang)
                
                # Save processed data
                train_split.to_csv(f"{output_dir}train_{lang}.csv", index=False)
                val_split.to_csv(f"{output_dir}val_{lang}.csv", index=False)
                if len(test_lang) > 0:
                    test_lang.to_csv(f"{output_dir}test_{lang}.csv", index=False)
                
                datasets[lang] = {
                    'train': train_split,
                    'val': val_split,
                    'test': test_lang if len(test_lang) > 0 else pd.DataFrame()
                }
            else:
                print(f"No training data for {lang}")
                datasets[lang] = {}
        
        return datasets
    
    
    def preprocess_dataset(self, df, lang='en'):
        """Preprocess entire dataset"""
        from utils import clean_text_for_training

        df = df.copy()
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(lambda x: clean_text_for_training(x, lang))
        
        # Ensure labels are integers
        if 'label' in df.columns:
            df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
        
        return df