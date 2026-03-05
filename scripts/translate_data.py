#!/usr/bin/env python3
"""
Standalone script for translating English data to German
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipeline.translate import TranslationPipeline

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Translate English medical data to German")
    parser.add_argument("--input", type=str, default="data/raw/train.csv",
                       help="Path to input CSV file")
    parser.add_argument("--output", type=str, default="data/translated/translated_train_de.csv",
                       help="Path to save translated data")
    parser.add_argument("--n_samples", type=int, default=2000,
                       help="Number of samples to translate")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size for API calls")
    parser.add_argument("--sleep_time", type=float, default=0.5,
                       help="Sleep time between batches")
    parser.add_argument("--max_retries", type=int, default=5,
                       help="Maximum retry attempts per batch")
    # parser.add_argument("--api_key", type=str, default=None,
    #                    help="Google API key (defaults to GOOGLE_API_KEY env var)")
    parser.add_argument("--model", type=str, default="mistral:latest",
                       help="Ollama model to use")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Will translate {args.n_samples} samples")
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist!")
        print("Available files in data/raw/:")
        if os.path.exists("data/raw"):
            for f in os.listdir("data/raw"):
                print(f"  - {f}")
        return 1
    
    # Initialize translation pipeline
    translator = TranslationPipeline(model_name=args.model)
    
    # Run translation
    try:
        result = translator.translate_dataset(
            input_path=args.input,
            output_path=args.output,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            sleep_time=args.sleep_time,
            max_retries=args.max_retries
        )
        
        print(f"\n✓ Translation completed successfully!")
        print(f"  Output saved to: {args.output}")
        print(f"  Translated {len(result)} samples")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Translation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Check for partial results
        partial_path = args.output.replace('.csv', '_partial.csv')
        checkpoint_dir = os.path.join("data/translated", "translation_checkpoints")
        
        if os.path.exists(partial_path):
            print(f"\n⚠ Partial results exist at: {partial_path}")
            print(f"  Size: {os.path.getsize(partial_path)} bytes")
        elif os.path.exists(checkpoint_dir):
            print(f"\n⚠ Checkpoints exist in: {checkpoint_dir}")
            print("  You can resume by running the same command again.")
        
        return 1

if __name__ == "__main__":
    exit(main())        