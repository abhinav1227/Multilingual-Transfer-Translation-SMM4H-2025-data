"""
Shared utilities for the SMM4H pipeline
"""
import re
import pandas as pd
import numpy as np

def clean_text_for_training(text, lang='en'):
    """
    Clean text for model training - removes noise, standardizes text
    This is used for preprocessing training data
    """
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Common cleaning for all languages
    text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '[USER]', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Language-specific character preservation
    if lang == 'ru':
        # Cyrillic specific
        text = re.sub(r'[^\w\sа-яА-ЯёЁ.,!?\'\"\-\[\]\{\}()]', '', text)
    elif lang == 'de':
        # German specific
        text = re.sub(r'[^\w\säöüÄÖÜß.,!?\'\"\-\[\]\{\}()]', '', text)
    else:  # English
        text = re.sub(r'[^\w\s.,!?\'\"\-\[\]\{\}()]', '', text)
    
    return text

def prepare_text_for_translation(text):
    """
    Minimal cleaning for translation - just basic formatting
    Keep all content including hashtags, informal language, etc.
    """
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Only minimal cleaning:
    # 1. Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # 2. Trim
    text = text.strip()
    
    return text

def clean_translated_text(text, lang='de'):
    """
    Clean translated text for training - applied AFTER translation
    """
    if pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # German-specific cleaning (after translation)
    if lang == 'de':
        # Keep German special characters, clean others
        text = re.sub(r'[^\w\säöüÄÖÜß.,!?\'\"\-\[\]\{\}()]', '', text)
    elif lang == 'ru':
        text = re.sub(r'[^\w\sа-яА-ЯёЁ.,!?\'\"\-\[\]\{\}()]', '', text)
    else:
        text = re.sub(r'[^\w\s.,!?\'\"\-\[\]\{\}()]', '', text)
    
    return text.strip()

def get_optimal_subset_size(datasets):
    """
    Calculate optimal subset size for translation based on available data
    """
    if 'en' not in datasets:
        return 1000  # Default
    
    en_train_size = len(datasets['en'].get('train', pd.DataFrame()))
    de_train_size = len(datasets['de'].get('train', pd.DataFrame())) if 'de' in datasets else 0
    
    # Use smaller of: half of English data, or enough to match German data
    subset_size = min(en_train_size // 2, max(1000, de_train_size * 2))
    
    print(f"Calculated optimal subset size: {subset_size}")
    print(f"- English training samples: {en_train_size}")
    print(f"- German training samples: {de_train_size}")
    
    return subset_size