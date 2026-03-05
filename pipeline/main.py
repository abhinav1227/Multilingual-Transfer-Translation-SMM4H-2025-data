# pipeline/main.py (updated)
"""
Main pipeline script for SMM4H multilingual modeling
"""
import argparse
import json
import os
import sys
from datetime import datetime
import pandas as pd
import torch

# Add pipeline directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing import DataPreprocessor
from tasks import task_1_monolingual_models, task_2_multilingual_model, task_3_mt_based_evaluation
from translate import TranslationPipeline

def setup_directories():
    """Create necessary directories"""
    dirs = [
        'data/raw',
        'data/processed',
        'data/translated',
        'models',
        'results',
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Directories created")

def load_processed_datasets(data_dir='data/processed'):
    """Load already processed datasets"""
    datasets = {}
    
    for lang in ['en', 'ru', 'de']:
        datasets[lang] = {}
        
        for split in ['train', 'val', 'test']:
            file_path = f"{data_dir}/{split}_{lang}.csv"
            if os.path.exists(file_path):
                datasets[lang][split] = pd.read_csv(file_path)
                print(f"Loaded {lang} {split}: {len(datasets[lang][split])} samples")
            else:
                datasets[lang][split] = pd.DataFrame()
                print(f"Warning: {file_path} not found")
    
    return datasets

def run_translation_only(args):
    """Run only the translation pipeline"""
    print("\n" + "="*70)
    print("TRANSLATION-ONLY MODE")
    print("="*70)
    
    # Check if raw data exists
    if not os.path.exists(args.raw_train_path):
        print(f"Error: Raw training data not found at {args.raw_train_path}")
        return
    
    # Initialize translation pipeline
    translator = TranslationPipeline(api_key=args.translation_api_key)
    
    # Run translation
    translator.translate_dataset(
        input_path=args.raw_train_path,
        output_path=args.translated_output_path,
        n_samples=args.translation_n_samples,
        batch_size=args.translation_batch_size,
        sleep_time=args.translation_sleep_time,
        max_retries=args.translation_max_retries
    )

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="SMM4H 2025 Multilingual Modeling Pipeline")
    
    # Main pipeline arguments
    parser.add_argument("--process_data", action="store_true", help="Process raw data before training")
    parser.add_argument("--task", type=str, default="all", 
                       choices=["1", "2", "3", "all", "translate"], 
                       help="Task to run (1.3.1, 1.3.2, 1.3.3, all, or translate only)")
    parser.add_argument("--model", type=str, default="bert-base-multilingual-cased",
                       help="Base transformer model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    
    # Data paths
    parser.add_argument("--raw_train_path", type=str, default="data/raw/train.csv",
                       help="Path to raw training data")
    parser.add_argument("--raw_test_path", type=str, default="data/raw/test.csv",
                       help="Path to raw test data")
    parser.add_argument("--translated_output_path", type=str, 
                       default="data/translated/translated_train_de.csv",
                       help="Path to save translated German data")
    
    # Translation-specific arguments
    parser.add_argument("--translation_api_key", type=str, default=None,
                       help="Google API key for translation (defaults to GOOGLE_API_KEY env var)")
    parser.add_argument("--translation_n_samples", type=int, default=2000,
                       help="Number of samples to translate")
    parser.add_argument("--translation_batch_size", type=int, default=10,
                       help="Batch size for translation API calls")
    parser.add_argument("--translation_sleep_time", type=float, default=0.5,
                       help="Sleep time between translation batches")
    parser.add_argument("--translation_max_retries", type=int, default=5,
                       help="Maximum retries for translation API calls")
    parser.add_argument("--force_retranslate", action="store_true",
                       help="Force re-translation even if file exists")
    
    args = parser.parse_args()
    
    print("="*70)
    print("SMM4H 2025 MULTILINGUAL MODELING PIPELINE")
    print("="*70)
    
    # Setup directories
    setup_directories()
    
    # If only translation is requested
    if args.task == "translate":
        run_translation_only(args)
        return
    
    # Initialize preprocessor
    print(f"\nInitializing preprocessor with model: {args.model}")
    preprocessor = DataPreprocessor(
        model_name=args.model,
        target_languages=['en', 'ru', 'de']
    )
    
    # Process or load data
    if args.process_data:
        print("\nProcessing raw data...")
        datasets = preprocessor.process_raw_data(
            train_path=args.raw_train_path,
            test_path=args.raw_test_path,
            output_dir='data/processed/'
        )
    else:
        print("\nLoading processed data...")
        datasets = load_processed_datasets('data/processed')
    
    # Check data availability
    print("\nData Summary:")
    for lang in ['en', 'ru', 'de']:
        if lang in datasets:
            train_size = len(datasets[lang].get('train', pd.DataFrame()))
            val_size = len(datasets[lang].get('val', pd.DataFrame()))
            test_size = len(datasets[lang].get('test', pd.DataFrame()))
            print(f"  {lang.upper()}: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Run selected tasks
    all_results = {}
    
    # Task 1.3.1: Monolingual models
    if args.task in ["1", "all"]:
        print("\n" + "="*70)
        print("TASK 1.3.1: MONOLINGUAL MODELS")
        print("="*70)
        
        try:
            results_1 = task_1_monolingual_models(
                datasets=datasets,
                preprocessor=preprocessor,
                output_dir="models",
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            all_results.update(results_1)
            print("\n✓ Task 1.3.1 completed successfully")
        except Exception as e:
            print(f"\n✗ Error in Task 1.3.1: {e}")
    
    # Task 1.3.2: Multilingual model
    if args.task in ["2", "all"]:
        print("\n" + "="*70)
        print("TASK 1.3.2: MULTILINGUAL MODEL")
        print("="*70)
        
        try:
            results_2 = task_2_multilingual_model(
                datasets=datasets,
                preprocessor=preprocessor,
                output_dir="models",
                epochs=args.epochs,
                batch_size=args.batch_size
            )
            all_results.update(results_2)
            print("\n✓ Task 1.3.2 completed successfully")
        except Exception as e:
            print(f"\n✗ Error in Task 1.3.2: {e}")
    
    # Task 1.3.3: MT-based evaluation (with translation)
    if args.task in ["3", "all"]:
        print("\n" + "="*70)
        print("TASK 1.3.3: MT-BASED EVALUATION")
        print("="*70)
        
        try:
            results_3 = task_3_mt_based_evaluation(
                datasets=datasets,
                preprocessor=preprocessor,
                raw_train_path=args.raw_train_path,
                translated_output_path=args.translated_output_path,
                output_dir="models",
                epochs=args.epochs,
                batch_size=args.batch_size,
                translation_api_key=args.translation_api_key,
                translation_n_samples=args.translation_n_samples
            )
            all_results.update(results_3)
            print("\n✓ Task 1.3.3 completed successfully")
        except Exception as e:
            print(f"\n✗ Error in Task 1.3.3: {e}")
    
    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/experiment_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ All results saved to: {results_file}")
    
    # Generate summary report
    print_summary_report(all_results)

def print_summary_report(all_results):
    """Print comprehensive summary report"""
    print("\n" + "="*70)
    print("FINAL SUMMARY REPORT")
    print("="*70)
    
    if all_results:
        print("\nPerformance Summary:")
        print("-" * 90)
        print(f"{'Model/Setup':<30} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 90)
        
        for model_name, metrics in all_results.items():
            if metrics:
                print(f"{model_name:<30} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} "
                      f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f}")
        
        print("-" * 90)
        
        # Key insights
        print("\nKey Insights:")
        
        # Compare monolingual vs multilingual
        if 'en_monolingual' in all_results and 'multilingual_en' in all_results:
            mono_acc = all_results['en_monolingual']['accuracy']
            multi_acc = all_results['multilingual_en']['accuracy']
            diff = (multi_acc - mono_acc) / mono_acc * 100
            print(f"1. English Monolingual vs Multilingual on English: {diff:+.2f}% difference")
        
        if 'ru_monolingual' in all_results and 'multilingual_ru' in all_results:
            mono_acc = all_results['ru_monolingual']['accuracy']
            multi_acc = all_results['multilingual_ru']['accuracy']
            diff = (multi_acc - mono_acc) / mono_acc * 100
            print(f"2. Russian Monolingual vs Multilingual on Russian: {diff:+.2f}% difference")
        
        # Zero-shot performance
        if 'multilingual_de_zero_shot' in all_results:
            print(f"3. Multilingual zero-shot on German: F1 = {all_results['multilingual_de_zero_shot']['f1']:.4f}")
        
        # MT-based evaluation comparison
        mt_models = ['en_zero_shot_de', 'multi_zero_shot_de', 'de_translated']
        mt_results = [(name, all_results[name]['f1']) for name in mt_models 
                     if name in all_results and all_results[name]]
        
        if mt_results:
            best_mt = max(mt_results, key=lambda x: x[1])
            print(f"4. Best MT-based approach: {best_mt[0]} with F1 = {best_mt[1]:.4f}")
            
            # Calculate improvement
            if 'en_zero_shot_de' in all_results and 'de_translated' in all_results:
                if all_results['en_zero_shot_de'] and all_results['de_translated']:
                    improvement = ((all_results['de_translated']['f1'] - 
                                   all_results['en_zero_shot_de']['f1']) / 
                                   all_results['en_zero_shot_de']['f1']) * 100
                    print(f"5. Translation improves over English zero-shot by: {improvement:+.2f}%")
    else:
        print("\nNo results generated. Check for errors in the tasks.")
    
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()