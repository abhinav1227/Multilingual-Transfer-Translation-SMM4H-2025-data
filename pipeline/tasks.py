"""
Task implementations for SMM4H multilingual modeling
"""
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import os

from train import MultilingualModelTrainer, TextDataset
from preprocessing import DataPreprocessor
from translate import TranslationPipeline

def create_dataloader(df, tokenizer, batch_size=16, shuffle=False, max_length=128):
    """Create DataLoader from DataFrame"""
    texts = df['cleaned_text'].tolist()
    labels = df['label'].values
    
    tokenized = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    dataset = TextDataset(tokenized, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def prepare_translated_data(datasets, raw_train_path, translated_output_path, 
                           n_samples=2000, api_key=None, force_retranslate=False):
    """
    Prepare translated German data for Task 1.3.3 (c)
    
    Args:
        datasets: Dictionary with language datasets
        raw_train_path: Path to raw training data
        translated_output_path: Path to save/load translated data
        n_samples: Number of samples to translate
        api_key: Google API key for translation
        force_retranslate: Whether to retranslate even if file exists
        
    Returns:
        DataFrame with translated German data
    """
    print("\n" + "="*60)
    print("PREPARING TRANSLATED GERMAN DATA")
    print("="*60)
    
    # Check if translated data already exists
    if os.path.exists(translated_output_path) and not force_retranslate:
        print(f"\nLoading existing translated data from: {translated_output_path}")
        translated_df = pd.read_csv(translated_output_path)
        print(f"Loaded {len(translated_df)} translated samples")

    else:

        print("\nNo translated data found or force_retranslate=True")
        print("Starting translation pipeline...")
        # Initialize translation pipeline
        try:
            translator = TranslationPipeline(api_key=api_key)
            
            # Translate dataset
            translated_df = translator.translate_dataset(
                input_path=raw_train_path,
                output_path=translated_output_path,
                n_samples=n_samples,
                batch_size=10,
                sleep_time=0.5,
                max_retries=5
            )
                        
        except Exception as e:
            print(f"\nError in translation: {e}")
            print("Falling back to English data for German training...")
            
            # Fallback: Use English data as pseudo-German
            if 'en' in datasets and 'train' in datasets['en']:
                english_train = datasets['en']['train'].copy()
                english_train['language'] = 'de'  # Mark as German
                english_train['is_translated'] = False
                print(f"Using {len(english_train)} English samples as fallback German data")
                return english_train
            else:
                raise ValueError("No English data available for fallback")
            
    translated_df.pop('text')  # Remove original text column
    translated_df = translated_df.rename(columns={'translated_text': 'text'})
    # Ensure language is set correctly
    translated_df['language'] = 'de'

    print(translated_df.head(3))
    print(translated_df.info())

    return translated_df

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

def task_1_monolingual_models(datasets, preprocessor, output_dir="models", epochs=3, batch_size=16):
    """
    Task 1.3.1: Train one model per training language
    
    Args:
        datasets: Dictionary with language keys containing train/val/test DataFrames
        preprocessor: DataPreprocessor instance
        output_dir: Directory to save models
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary with results for each language
    """
    print("\n" + "="*60)
    print("TASK 1.3.1: MONOLINGUAL MODELS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for lang in ['en', 'ru']:
        if lang not in datasets or 'train' not in datasets[lang]:
            print(f"Skipping {lang}: No data available")
            continue
            
        print(f"\nTraining {lang.upper()} monolingual model...")
        
        # Get data for this language
        train_df = datasets[lang]['train']
        val_df = datasets[lang]['val']
        test_df = datasets[lang]['test']
        
        # Preprocess
        train_processed = preprocessor.preprocess_dataset(train_df, lang)
        val_processed = preprocessor.preprocess_dataset(val_df, lang)
        test_processed = preprocessor.preprocess_dataset(test_df, lang)
        
        # Create DataLoaders
        train_loader = create_dataloader(train_processed, preprocessor.tokenizer, batch_size, True)
        val_loader = create_dataloader(val_processed, preprocessor.tokenizer, batch_size, False)
        test_loader = create_dataloader(test_processed, preprocessor.tokenizer, batch_size, False)
        
        # Train model
        trainer = MultilingualModelTrainer()

        flag = True
        if os.path.exists(f"{output_dir}/monolingual_{lang}"):
            print(f"Model already trained and saved at {output_dir}/monolingual_{lang}, therefore we will just load it instead of retraining.")
            trainer.load_model(f"{output_dir}/monolingual_{lang}")
            flag = False
        else:
            trainer.train(train_loader, val_loader, epochs=epochs)

        
        # Evaluate on test set
        test_results = trainer.evaluate(test_loader)
        results[f'{lang}_monolingual'] = test_results
        
        # Save model
        if flag:
            trainer.save_model(f"{output_dir}/monolingual_{lang}")
        
        print(f"{lang.upper()} model - Accuracy: {test_results['accuracy']:.4f}, F1: {test_results['f1']:.4f}")
    
    return results

def task_2_multilingual_model(datasets, preprocessor, output_dir="models", epochs=3, batch_size=16):
    """
    Task 1.3.2: Train a single multilingual model on combined training languages
    
    Args:
        datasets: Dictionary with language keys containing train/val/test DataFrames
        preprocessor: DataPreprocessor instance
        output_dir: Directory to save models
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary with results on seen and unseen languages
    """
    print("\n" + "="*60)
    print("TASK 1.3.2: MULTILINGUAL MODEL")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Combine English and Russian training data
    print("\nCombining English and Russian training data...")
    
    combined_train = []
    for lang in ['en', 'ru']:
        if lang in datasets and 'train' in datasets[lang]:
            train_df = datasets[lang]['train']
            processed = preprocessor.preprocess_dataset(train_df, lang)
            combined_train.append(processed)
    
    if len(combined_train) == 0:
        print("Error: No training data available for English or Russian")
        return results
    
    combined_train_df = pd.concat(combined_train, ignore_index=True)
    
    print(f"English samples: {len(datasets['en']['train']) if 'en' in datasets else 0}")
    print(f"Russian samples: {len(datasets['ru']['train']) if 'ru' in datasets else 0}")
    print(f"Total combined samples: {len(combined_train_df)}")
    
    # Use English validation for early stopping
    val_df = datasets['en']['val'] if 'en' in datasets else None
    if val_df is not None:
        val_processed = preprocessor.preprocess_dataset(val_df, 'en')
    
    # Create DataLoaders
    train_loader = create_dataloader(combined_train_df, preprocessor.tokenizer, batch_size, True)
    val_loader = create_dataloader(val_processed, preprocessor.tokenizer, batch_size, False) if val_df is not None else None
    
    # Train multilingual model
    
    trainer = MultilingualModelTrainer()

    if os.path.exists(f"{output_dir}/multilingual_en_ru"):
        print(f"Model already trained and saved at {output_dir}/multilingual_en_ru, therefore we will just load it instead of retraining.")
        trainer.load_model(f"{output_dir}/multilingual_en_ru")
    else:
        # Train multilingual model
        trainer.train(train_loader, val_loader, epochs=epochs)

        # Save model
        trainer.save_model(f"{output_dir}/multilingual_en_ru")
    
    # Evaluate on all languages
    print("\n" + "="*60)
    print("EVALUATING MULTILINGUAL MODEL")
    print("="*60)
    
    # Evaluate on seen languages
    print("\n1. Evaluation on SEEN languages:")
    
    for lang in ['en', 'ru']:
        if lang in datasets and 'test' in datasets[lang]:
            test_df = datasets[lang]['test']
            test_processed = preprocessor.preprocess_dataset(test_df, lang)
            test_loader = create_dataloader(test_processed, preprocessor.tokenizer, batch_size, False)
            
            test_results = trainer.evaluate(test_loader)
            results[f'multilingual_{lang}'] = test_results
            
            print(f"   {lang.upper()}: Accuracy: {test_results['accuracy']:.4f}, F1: {test_results['f1']:.4f}")
    
    # Evaluate on unseen language (German) - Zero-shot
    print("\n2. Evaluation on UNSEEN language (German) - Zero-shot:")
    if 'de' in datasets and 'test' in datasets['de']:
        de_test_df = datasets['de']['test']
        de_test_processed = preprocessor.preprocess_dataset(de_test_df, 'de')
        de_test_loader = create_dataloader(de_test_processed, preprocessor.tokenizer, batch_size, False)
        
        de_results = trainer.evaluate(de_test_loader)
        results['multilingual_de_zero_shot'] = de_results
        
        print(f"   German: Accuracy: {de_results['accuracy']:.4f}, F1: {de_results['f1']:.4f}")
    else:
        print("   Warning: German test data not available")
    
    return results

def task_3_mt_based_evaluation(datasets, preprocessor, 
                              raw_train_path="data/raw/train.csv",
                              translated_output_path="data/translated/translated_train_de.csv",
                              output_dir="models", 
                              epochs=3, 
                              batch_size=16,
                              translation_api_key=None,
                              translation_n_samples=None):
    """
    Task 1.3.3: Machine translation-based evaluation
    
    Now includes automatic translation of English data to German
    """
    print("\n" + "="*60)
    print("TASK 1.3.3: MT-BASED EVALUATION")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    # Check if we have German test data
    if 'de' not in datasets or 'test' not in datasets['de']:
        print("Error: German test data not available")
        return results
    
    # Prepare German test set
    de_test_df = datasets['de']['test']
    de_test_processed = preprocessor.preprocess_dataset(de_test_df, 'de')
    de_test_loader = create_dataloader(de_test_processed, preprocessor.tokenizer, batch_size, False)
    
    # a) English model (zero-shot on German)
    print("\n" + "-"*50)
    print("a) English model (zero-shot on German)")
    print("-"*50)
    
    en_results = evaluate_english_model(de_test_loader, output_dir)
    results['en_zero_shot_de'] = en_results
    
    # b) Multilingual model (zero-shot on German)
    print("\n" + "-"*50)
    print("b) Multilingual model (zero-shot on German)")
    print("-"*50)
    
    multi_results = evaluate_multilingual_model(de_test_loader, output_dir)
    results['multi_zero_shot_de'] = multi_results
    
    # c) Model trained on German translations
    print("\n" + "-"*50)
    print("c) Model trained on translated German data")
    print("-"*50)
    
    # Prepare translated data
    if translation_n_samples is None:
        translation_n_samples = get_optimal_subset_size(datasets)
    
    translated_df = prepare_translated_data(
        datasets=datasets,
        raw_train_path=raw_train_path,
        translated_output_path=translated_output_path,
        n_samples=translation_n_samples,
        api_key=translation_api_key,
        force_retranslate=False
    )
    
    # Train and evaluate model on translated data
    de_translated_results = train_and_evaluate_translated_model(
        translated_df=translated_df,
        de_test_loader=de_test_loader,
        preprocessor=preprocessor,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size
    )
    
    results['de_translated'] = de_translated_results
    
    # Compare results
    print_results_comparison(results)
    
    return results

def evaluate_english_model(de_test_loader, output_dir):
    """Helper: Evaluate English model on German test set"""
    try:
        en_model_path = f"{output_dir}/monolingual_en"
        if os.path.exists(en_model_path):
            trainer = MultilingualModelTrainer()
            trainer.load_model(en_model_path)
            results = trainer.evaluate(de_test_loader)
            print(f"   Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")
            return results
        else:
            print("   Warning: English monolingual model not found")
            return None
    except Exception as e:
        print(f"   Error: {e}")
        return None

def evaluate_multilingual_model(de_test_loader, output_dir):
    """Helper: Evaluate multilingual model on German test set"""
    try:
        multi_model_path = f"{output_dir}/multilingual_en_ru"
        if os.path.exists(multi_model_path):
            trainer = MultilingualModelTrainer()
            trainer.load_model(multi_model_path)
            results = trainer.evaluate(de_test_loader)
            print(f"   Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")
            return results
        else:
            print("   Warning: Multilingual model not found")
            return None
    except Exception as e:
        print(f"   Error: {e}")
        return None

def train_and_evaluate_translated_model(translated_df, de_test_loader, preprocessor, 
                                       output_dir, epochs=3, batch_size=16):
    """Helper: Train model on translated data and evaluate"""
    try:
        # Preprocess translated data
        translated_processed = preprocessor.preprocess_dataset(translated_df, 'de')
        
        # Split into train/val
        if len(translated_processed) > 100:
            train_size = int(0.8 * len(translated_processed))
            train_df = translated_processed.iloc[:train_size]
            val_df = translated_processed.iloc[train_size:]
        else:
            train_df = translated_processed
            val_df = None
        
        print(f"   Translated data: {len(train_df)} training samples")
        if val_df is not None:
            print(f"                  {len(val_df)} validation samples")
        
        # Create DataLoaders
        train_loader = create_dataloader(train_df, preprocessor.tokenizer, batch_size, True)
        val_loader = create_dataloader(val_df, preprocessor.tokenizer, batch_size, False) if val_df is not None else None
        
        # Train model
        trainer = MultilingualModelTrainer()
        trainer.train(train_loader, val_loader, epochs=epochs)
        
        # Evaluate on German test set
        results = trainer.evaluate(de_test_loader)
        print(f"   Accuracy: {results['accuracy']:.4f}, F1: {results['f1']:.4f}")
        
        # Save model
        trainer.save_model(f"{output_dir}/de_translated")
        
        return results
        
    except Exception as e:
        print(f"   Error training translated model: {e}")
        return None

def print_results_comparison(results):
    """Helper: Print comparison of results"""
    print("\n" + "="*60)
    print("MT-BASED EVALUATION RESULTS COMPARISON")
    print("="*60)
    
    print("\nPerformance on German test set:")
    print("-" * 80)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        if metrics:
            print(f"{model_name:<25} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} "
                  f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")
        else:
            print(f"{model_name:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    print("-" * 80)
    
    # Calculate improvements
    valid_results = {k: v for k, v in results.items() if v}
    
    if 'en_zero_shot_de' in valid_results and 'de_translated' in valid_results:
        en_f1 = valid_results['en_zero_shot_de']['f1']
        translated_f1 = valid_results['de_translated']['f1']
        improvement = ((translated_f1 - en_f1) / en_f1) * 100
        print(f"\nTranslation improves over English zero-shot by: {improvement:+.2f}%")
    
    if 'multi_zero_shot_de' in valid_results and 'de_translated' in valid_results:
        multi_f1 = valid_results['multi_zero_shot_de']['f1']
        translated_f1 = valid_results['de_translated']['f1']
        improvement = ((translated_f1 - multi_f1) / multi_f1) * 100
        print(f"Translation improves over multilingual zero-shot by: {improvement:+.2f}%")