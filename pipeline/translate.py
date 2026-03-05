# pipeline/translate.py
"""
Module for translating English data to German for zero-shot evaluation
with checkpointing, using local LLMs via Ollama.
"""
import os
import sys
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import warnings
import logging
import re

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import from utils.py (it's in the same pipeline directory)
try:
    from pipeline.utils import prepare_text_for_translation, clean_translated_text
except ImportError:
    try:
        from utils import prepare_text_for_translation, clean_translated_text
    except ImportError:
        print("Warning: Could not import from utils.py. Using fallback functions.")
        
        def prepare_text_for_translation(text):
            """Minimal text preparation for translation"""
            if pd.isna(text):
                return ""
            text = str(text).strip()
            text = ' '.join(text.split())
            return text[:4000]
        
        def clean_translated_text(text, lang='de'):
            """Minimal cleaning for translated text"""
            if pd.isna(text):
                return ""
            text = str(text).strip()
            if text[:2].isdigit() and text[2] in ['.', ')', ':']:
                text = text[3:].strip()
            return text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TranslationPipeline:
    def __init__(self, model_name="mistral:latest", ollama_url="http://localhost:11434"):
        """
        Initialize translation pipeline with Ollama local LLM.
        
        Args:
            model_name: Ollama model name (e.g., 'mistral', 'llama2:latest', 'gemma:7b')
            ollama_url: URL of Ollama API (default: http://localhost:11434)
        """
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip('/')
        
        # Check if Ollama is running and model is available
        self._check_ollama()
        
        # Initialize counters and state
        self.api_calls = 0
        self.start_time = time.time()
        self.total_tokens_used = 0
        self.last_successful_batch = None
        
    def _check_ollama(self):
        """Check if Ollama is running and the model is available."""
        try:
            # Test connection
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if self.model_name not in model_names:
                # Try to find by prefix (e.g., 'mistral' matches 'mistral:latest')
                available = [m for m in model_names if m.startswith(self.model_name)]
                if available:
                    self.model_name = available[0]
                    logger.info(f"Using model: {self.model_name} (matched from available)")
                else:
                    logger.warning(f"Model '{self.model_name}' not found in Ollama. Available: {model_names[:5]}")
                    logger.warning("Will attempt to use anyway, but ensure it's pulled.")
            else:
                logger.info(f"✓ Using Ollama model: {self.model_name}")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to Ollama at {self.ollama_url}. Is Ollama running?")
        except Exception as e:
            logger.warning(f"Ollama check failed: {e}. Proceeding anyway.")
    
    def _call_ollama(self, prompt, max_tokens=4096, temperature=0.1, top_p=0.95):
        """
        Call Ollama API to generate translation.
        
        Returns:
            Generated text string.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
            }
        }
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # Longer timeout for generation
            )
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except requests.exceptions.Timeout:
            logger.warning("Ollama request timed out")
            raise
        except Exception as e:
            logger.warning(f"Ollama API error: {e}")
            raise
    
    def _setup_checkpointing(self, output_path):
        """Setup checkpoint directory and paths"""
        checkpoint_dir = os.path.join(os.path.dirname(output_path), "translation_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_prefix = f"checkpoint_{timestamp}"
        
        return {
            'dir': checkpoint_dir,
            'state': os.path.join(checkpoint_dir, f"{checkpoint_prefix}_state.json"),
            'partial': os.path.join(checkpoint_dir, f"{checkpoint_prefix}_partial.csv"),
            'log': os.path.join(checkpoint_dir, f"{checkpoint_prefix}_log.txt"),
            'prefix': checkpoint_prefix
        }
    
    def _save_checkpoint_state(self, checkpoint_paths, state_data):
        """Save checkpoint state to JSON"""
        try:
            state_data['timestamp'] = datetime.now().isoformat()
            state_data['api_calls'] = self.api_calls
            state_data['total_tokens'] = self.total_tokens_used
            
            with open(checkpoint_paths['state'], 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info(f"Checkpoint state saved: {checkpoint_paths['state']}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint state: {e}")
            return False
    
    def _load_checkpoint_state(self, checkpoint_path):
        """Load checkpoint state from JSON"""
        try:
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r') as f:
                    state_data = json.load(f)
                
                logger.info(f"Loaded checkpoint state from {checkpoint_path}")
                logger.info(f"  Timestamp: {state_data.get('timestamp', 'unknown')}")
                logger.info(f"  API calls: {state_data.get('api_calls', 0)}")
                logger.info(f"  Progress: {state_data.get('completed', 0)}/{state_data.get('total', 0)}")
                
                return state_data
        except Exception as e:
            logger.warning(f"Failed to load checkpoint state: {e}")
        
        return None
    
    def _save_partial_results(self, df_partial, output_path, checkpoint_paths):
        """Save partial translation results"""
        try:
            # Save to main partial file
            if os.path.exists(checkpoint_paths['partial']):
                existing_df = pd.read_csv(checkpoint_paths['partial'])
                df_combined = pd.concat([existing_df, df_partial], ignore_index=True)
                df_combined.to_csv(checkpoint_paths['partial'], index=False)
            else:
                df_partial.to_csv(checkpoint_paths['partial'], index=False)
            
            # Also update the main output file incrementally
            self._update_main_output(df_partial, output_path)
            
            logger.info(f"Saved {len(df_partial)} partial translations")
            return True
        except Exception as e:
            logger.error(f"Failed to save partial results: {e}")
            return False
    
    def _update_main_output(self, df_new, output_path):
        """Update main output file with new translations"""
        try:
            # If output file exists, append; otherwise create new
            if os.path.exists(output_path):
                mode = 'a'
                header = False
            else:
                mode = 'w'
                header = True
            
            df_new.to_csv(output_path, mode=mode, header=header, index=False)
            return True
        except Exception as e:
            logger.error(f"Failed to update main output: {e}")
            return False
    
    def _log_translation_attempt(self, checkpoint_paths, batch_idx, success, error_msg=""):
        """Log translation attempt"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'batch': batch_idx,
                'success': success,
                'error': error_msg,
                'api_calls': self.api_calls,
                'tokens': self.total_tokens_used
            }
            
            with open(checkpoint_paths['log'], 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to log translation attempt: {e}")
    
    def get_translation_subset(self, df_train, n_samples=2000, language_col='language'):
        """
        Return a balanced subset of English data for translation
        """
        # If language column doesn't exist, assume all data is English
        if language_col not in df_train.columns:
            logger.warning(f"Language column '{language_col}' not found. Assuming all data is English.")
            df_en = df_train.copy()
        else:
            df_en = df_train[df_train[language_col] == 'en'].copy()
        
        if len(df_en) == 0:
            raise ValueError("No English data found in the dataset")
        
        logger.info(f"Found {len(df_en)} English samples")
        
        # Check if label column exists
        if 'label' not in df_en.columns:
            raise ValueError("Data must have a 'label' column for balanced sampling.")
        
        logger.info(f"Label distribution in English data: {df_en['label'].value_counts().to_dict()}")
                
        # Separate classes
        df_pos = df_en[df_en['label'] == 1].copy()
        df_neg = df_en[df_en['label'] == 0].copy()
        
        n_pos = len(df_pos)
        n_neg = len(df_neg)
        
        if n_pos == 0 or n_neg == 0:
            logger.warning(f"Unbalanced data: positive={n_pos}, negative={n_neg}")
            logger.warning("Using all available English data")
            df_subset = df_en.sample(n=min(n_samples, len(df_en)), random_state=42)
        else:
            # Create balanced subset
            samples_per_class = min(n_samples // 2, n_pos, n_neg)
            
            if samples_per_class > 0:
                df_pos_sample = df_pos.sample(n=samples_per_class, random_state=42)
                df_neg_sample = df_neg.sample(n=samples_per_class, random_state=42)
                df_subset = pd.concat([df_pos_sample, df_neg_sample])
            else:
                # Fallback to proportional sampling
                ratio_pos = n_pos / (n_pos + n_neg)
                n_pos_samples = int(n_samples * ratio_pos)
                n_neg_samples = n_samples - n_pos_samples
                
                df_pos_sample = df_pos.sample(n=min(n_pos_samples, n_pos), random_state=42)
                df_neg_sample = df_neg.sample(n=min(n_neg_samples, n_neg), random_state=42)
                df_subset = pd.concat([df_pos_sample, df_neg_sample])
        
        # Shuffle and reset index
        df_subset = df_subset.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Created subset: {len(df_subset)} total samples")
        logger.info(f"Label distribution: {df_subset['label'].value_counts().to_dict()}")
        
        return df_subset
    
    def build_batch_prompt(self, texts, target_lang="German"):
        """
        Build a prompt for multiple translations that preserves the original style.
        Optimized for local LLMs to follow numbered output.
        """
        num_texts = len(texts)
        
        prompt = f"""You are a professional medical translator. Translate the following {num_texts} English medical social media posts into {target_lang}.

**CRITICAL INSTRUCTIONS**:
1. Preserve the original tone, style, and meaning.
2. Keep hashtags (#example), mentions (@user), and URLs unchanged.
3. Use accurate medical terminology in {target_lang}.
4. Make the translation sound natural in {target_lang}.
5. **Output format**: Provide ONLY the translations, each on its own line, numbered from 1 to {num_texts}.
6. Do NOT add any extra text, explanations, or commentary.
7. Do NOT repeat the English texts.

English posts:
"""
        
        # Add each text with numbering
        for i, text in enumerate(texts, 1):
            # Clean and truncate if necessary
            clean_text = str(text).strip()[:500]
            prompt += f"\n{i}. {clean_text}"
        
        prompt += f"\n\n{target_lang} translations (numbered 1 to {num_texts}):\n"
        
        return prompt
    
    def translate_batch_with_backoff(self, texts, max_retries=5, target_lang="German"):
        """
        Translate a batch with exponential backoff retry.
        Uses Ollama local LLM.
        """
        prompt = self.build_batch_prompt(texts, target_lang)
        
        for attempt in range(max_retries):
            try:
                # Calculate wait time (exponential backoff)
                if attempt > 0:
                    wait_time = 2 ** (attempt + 1)  # 4, 8, 16, 32, 64 seconds
                    logger.info(f"  Attempt {attempt + 1}/{max_retries} - waiting {wait_time}s")
                    time.sleep(wait_time)
                
                logger.info(f"  Translating {len(texts)} texts...")
                
                # Estimate tokens (roughly)
                estimated_tokens = len(prompt) // 4
                
                # Call Ollama
                response_text = self._call_ollama(
                    prompt,
                    max_tokens=4096,
                    temperature=0.1,
                    top_p=0.95
                )
                
                self.api_calls += 1
                self.total_tokens_used += estimated_tokens
                
                if not response_text or not response_text.strip():
                    logger.warning("  Empty response from Ollama")
                    continue
                
                # Parse response
                translations = self._parse_translation_response(response_text, len(texts))
                
                if len(translations) == len(texts) and all(t.strip() for t in translations):
                    logger.info(f"  ✓ Successfully translated {len(translations)} texts")
                    self.last_successful_batch = datetime.now()
                    return translations
                else:
                    logger.warning(f"  Expected {len(texts)} translations, got {len(translations)}")
                    logger.debug(f"  Response preview: {response_text[:200]}...")
                    
                    # If we got some translations but wrong count, try to fix
                    if translations:
                        # Pad with empty strings or truncate
                        if len(translations) > len(texts):
                            translations = translations[:len(texts)]
                        else:
                            translations.extend([""] * (len(texts) - len(translations)))
                        return translations
                    
            except requests.exceptions.Timeout:
                logger.warning("  Ollama request timed out, retrying...")
            except requests.exceptions.ConnectionError:
                logger.warning("  Cannot connect to Ollama, is it running? Retrying...")
                time.sleep(10)  # Extra wait if connection issue
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"  Attempt {attempt + 1}/{max_retries} failed: {error_msg[:100]}")
                
                # Check for specific errors
                if 'context length exceeded' in error_msg.lower():
                    logger.error("  ⚠ Context length exceeded. Try reducing batch size or text length.")
                    return [""] * len(texts)
        
        # All retries failed
        logger.error(f"  ✗ All {max_retries} retries failed for batch")
        return [""] * len(texts)
    
    def _parse_translation_response(self, response_text, expected_count):
        """
        Parse the Ollama response to extract numbered translations.
        Handles various formats.
        """
        lines = [line.strip() for line in response_text.strip().split("\n") if line.strip()]
        translations = []
        
        for line in lines:
            # Skip lines that are clearly not translations (headers, etc.)
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in [
                'english', 'german', 'translation', 'note:', 'sure', 'here are', 
                'translations:', 'output:', '---', '===']
            ):
                continue
            
            # Remove numbering (1., 2), 3:, 1-, etc.)
            line_clean = line.strip()
            
            # Try different numbering patterns
            for pattern in [r'^\d+\.\s*', r'^\d+\)\s*', r'^\d+:\s*', r'^\d+\-\s*', r'^\d+\s+']:
                if re.match(pattern, line_clean):
                    line_clean = re.sub(pattern, '', line_clean)
                    break
            
            if line_clean and len(line_clean) > 3:  # At least 3 characters
                translations.append(line_clean)
        
        # If parsing failed (e.g., model didn't use numbers), try a simpler approach:
        # Take first `expected_count` non-empty lines that are not the prompt.
        if len(translations) != expected_count and len(translations) > 0:
            logger.warning(f"Parsing with numbering failed, using raw line split.")
            # Reset and try raw split on newlines
            raw_lines = [l.strip() for l in response_text.strip().split('\n') if l.strip()]
            # Remove first few lines if they look like the English posts (start with number and English text)
            cleaned_raw = []
            for line in raw_lines:
                # If line starts with a number and contains English words, skip? This is tricky.
                # Instead, take last `expected_count` lines if they seem like translations.
                if len(cleaned_raw) < expected_count and len(line) > 10:
                    cleaned_raw.append(line)
            
            # If we have too many, take the last expected_count
            if len(cleaned_raw) > expected_count:
                cleaned_raw = cleaned_raw[-expected_count:]
            translations = cleaned_raw
        
        # Ensure we have the right number
        if len(translations) > expected_count:
            translations = translations[:expected_count]
        elif len(translations) < expected_count:
            translations.extend([""] * (expected_count - len(translations)))
        
        return translations
    
    def translate_dataset(self, input_path, output_path, n_samples=2000, 
                         batch_size=10, sleep_time=0.5, max_retries=5):
        """
        Main translation pipeline with checkpointing and resume capability.
        """
        print(f"\n{'='*60}")
        print("TRANSLATION PIPELINE - ENGLISH TO GERMAN (Local LLM via Ollama)")
        print(f"{'='*60}")
        
        # Setup checkpointing
        checkpoint_paths = self._setup_checkpointing(output_path)
        
        # Load data
        print(f"\nLoading data from {input_path}...")
        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            # Try different encodings
            try:
                df = pd.read_csv(input_path, encoding='latin1')
            except:
                try:
                    df = pd.read_csv(input_path, encoding='utf-8-sig')
                except:
                    raise ValueError(f"Cannot read file {input_path} with any encoding")
        
        print(f"Total samples: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check if we have necessary columns
        required_cols = ['text', 'label', 'language']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}. Trying to infer...")
            # Try to find alternative column names
            if 'text' not in df.columns:
                if 'post' in df.columns:
                    df = df.rename(columns={'post': 'text'})
                elif 'content' in df.columns:
                    df = df.rename(columns={'content': 'text'})
                else:
                    raise ValueError("No text column found in data")
        
        # Create subset for translation
        print(f"\nCreating balanced subset of {n_samples} samples...")
        df_subset = self.get_translation_subset(df, n_samples=n_samples)
        
        # Save subset for reference
        subset_path = output_path.replace('.csv', '_subset.csv')
        df_subset.to_csv(subset_path, index=False)
        print(f"Subset saved to: {subset_path}")
        
        # Prepare texts
        texts = df_subset['text'].tolist()
        texts_prepared = [prepare_text_for_translation(t) for t in texts]
        
        # Filter out very short texts
        valid_indices = [i for i, t in enumerate(texts_prepared) if len(str(t).strip()) > 10]
        if len(valid_indices) < len(texts_prepared):
            print(f"Filtered out {len(texts_prepared) - len(valid_indices)} texts that were too short")
            texts_prepared = [texts_prepared[i] for i in valid_indices]
            df_subset = df_subset.iloc[valid_indices].reset_index(drop=True)
            texts = [texts[i] for i in valid_indices]
        
        total_texts = len(texts_prepared)
        print(f"\nWill translate {total_texts} texts in batches of {batch_size}")
        print(f"Estimated API calls: {total_texts // batch_size + 1}")
        
        # Check for existing checkpoint
        checkpoint_state = self._load_checkpoint_state(checkpoint_paths['state'])
        
        if checkpoint_state and 'translations' in checkpoint_state:
            print(f"\nFound existing checkpoint with {len(checkpoint_state['translations'])} translations")
            
            translations = checkpoint_state['translations']
            completed_count = len([t for t in translations if t and str(t).strip()])
            
            if len(translations) == total_texts:
                print(f"Checkpoint appears complete: {completed_count}/{total_texts} translations")
                response = input("Continue with existing translations? (y/n): ")
                if response.lower() == 'y':
                    # Create final output from checkpoint
                    return self._create_final_output(df_subset, translations, output_path)
            
            # Resume from checkpoint
            start_idx = len(translations)
            print(f"Resuming from position {start_idx}/{total_texts}")
        else:
            translations = [None] * total_texts
            start_idx = 0
        
        # Main translation loop
        batch_count = 0
        successful_batches = 0
        failed_batches = 0
        
        print(f"\nStarting translation...")
        
        for batch_start in tqdm(range(start_idx, total_texts, batch_size), 
                               desc="Translating", initial=start_idx//batch_size,
                               total=(total_texts + batch_size - 1)//batch_size):
            batch_end = min(batch_start + batch_size, total_texts)
            batch_texts = texts_prepared[batch_start:batch_end]
            batch_idx = batch_start // batch_size
            
            print(f"\nBatch {batch_idx + 1}: Texts {batch_start}-{batch_end-1}")
            
            # Skip if already translated in checkpoint
            if translations[batch_start] is not None:
                print(f"  Already translated in checkpoint, skipping")
                continue
            
            # Translate batch
            batch_translations = self.translate_batch_with_backoff(
                batch_texts, max_retries=max_retries, target_lang="German"
            )
            
            # Update translations
            for i, trans in enumerate(batch_translations):
                translations[batch_start + i] = trans
            
            # Count success/failure
            successful = sum(1 for t in batch_translations if t and str(t).strip())
            if successful == len(batch_translations):
                successful_batches += 1
            else:
                failed_batches += 1
            
            print(f"  Result: {successful}/{len(batch_translations)} successful translations")
            
            # Save partial results every few batches
            if (batch_idx + 1) % 5 == 0 or batch_end >= total_texts:
                self._save_partial_progress(df_subset, translations, batch_end, 
                                          output_path, checkpoint_paths)
            
            # Sleep between batches to avoid overloading local LLM (optional)
            if batch_end < total_texts:
                time.sleep(sleep_time)
            
            batch_count += 1
        
        # Create final output
        print(f"\nTranslation complete!")
        print(f"  Total batches: {batch_count}")
        print(f"  Successful batches: {successful_batches}")
        print(f"  Failed batches: {failed_batches}")
        print(f"  API calls: {self.api_calls}")
        print(f"  Estimated tokens: {self.total_tokens_used}")
        
        return self._create_final_output(df_subset, translations, output_path)
    
    def _save_partial_progress(self, df_subset, translations, up_to_idx, 
                             output_path, checkpoint_paths):
        """Save partial progress"""
        try:
            # Create DataFrame with completed translations
            completed_data = []
            for i in range(up_to_idx):
                if translations[i] is not None and str(translations[i]).strip():
                    row = df_subset.iloc[i].copy()
                    row['original_text'] = row['text']
                    row['translated_text'] = translations[i]
                    row['text'] = clean_translated_text(translations[i], 'de')
                    row['language'] = 'de'
                    row['is_translated'] = True
                    completed_data.append(row)
            
            if completed_data:
                df_completed = pd.DataFrame(completed_data)
                
                # Save partial CSV
                self._save_partial_results(df_completed, output_path, checkpoint_paths)
                
                # Save checkpoint state
                checkpoint_state = {
                    'total': len(translations),
                    'completed': len(completed_data),
                    'translations': translations,
                    'batch_size': 10,
                    'model': self.model_name
                }
                self._save_checkpoint_state(checkpoint_paths, checkpoint_state)
                
                print(f"  Checkpoint saved: {len(completed_data)}/{len(translations)} completed")
        
        except Exception as e:
            print(f"  Warning: Failed to save checkpoint: {e}")
    
    def _create_final_output(self, df_subset, translations, output_path):
        """Create final output DataFrame"""
        print(f"\nCreating final output...")
        
        # Filter out failed translations
        output_data = []
        failed_indices = []
        
        for i, (_, row) in enumerate(df_subset.iterrows()):
            if i < len(translations) and translations[i] and str(translations[i]).strip():
                new_row = row.copy()
                new_row['original_text'] = row['text']
                new_row['translated_text'] = translations[i]
                
                # Clean translation for training
                cleaned = clean_translated_text(translations[i], 'de')
                if len(cleaned) > 20:  # Reasonable minimum length
                    new_row['text'] = cleaned
                    new_row['language'] = 'de'
                    new_row['is_translated'] = True
                    output_data.append(new_row)
                else:
                    failed_indices.append(i)
            else:
                failed_indices.append(i)
        
        df_output = pd.DataFrame(output_data)
        
        # Statistics
        total = len(df_subset)
        success = len(df_output)
        success_rate = (success / total) * 100 if total > 0 else 0
        
        print(f"\nTranslation Statistics:")
        print(f"  Total attempted: {total}")
        print(f"  Successfully translated: {success}")
        print(f"  Failed: {total - success}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Final dataset size: {len(df_output)}")
        print(f"  Label distribution: {df_output['label'].value_counts().to_dict()}")
        
        # Save final output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_output.to_csv(output_path, index=False)
        print(f"\n✓ Final translations saved to: {output_path}")
        
        # Save summary
        self._save_translation_summary(output_path, df_output, total, success)
        
        # Save failed samples for debugging
        if failed_indices:
            failed_df = df_subset.iloc[failed_indices].copy()
            failed_path = output_path.replace('.csv', '_failed.csv')
            failed_df.to_csv(failed_path, index=False)
            print(f"  Failed samples saved to: {failed_path}")
        
        return df_output
    
    def _save_translation_summary(self, output_path, df_output, total_attempted, success_count):
        """Save detailed summary report"""
        summary_path = output_path.replace('.csv', '_summary.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("TRANSLATION PIPELINE SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_name} (Ollama)\n")
            f.write(f"Ollama URL: {self.ollama_url}\n")
            f.write(f"API Calls: {self.api_calls}\n")
            f.write(f"Total Tokens (est): {self.total_tokens_used}\n")
            f.write(f"Total Attempted: {total_attempted}\n")
            f.write(f"Successfully Translated: {success_count}\n")
            f.write(f"Success Rate: {(success_count/total_attempted*100):.1f}%\n")
            f.write(f"Final Dataset Size: {len(df_output)}\n\n")
            
            f.write("Label Distribution:\n")
            f.write(str(df_output['label'].value_counts().to_dict()) + "\n\n")
            
            f.write("Sample Translations:\n")
            f.write("="*60 + "\n")
            for i in range(min(10, len(df_output))):
                f.write(f"\n[{i+1}] Label: {df_output.iloc[i]['label']}\n")
                f.write(f"Original (EN): {df_output.iloc[i]['original_text'][:200]}...\n")
                f.write(f"Translated (DE): {df_output.iloc[i]['translated_text'][:200]}...\n")
                f.write(f"Cleaned (DE): {df_output.iloc[i]['text'][:200]}...\n")
                f.write("-"*40 + "\n")
        
        print(f"✓ Summary saved to: {summary_path}")


def run_translation_from_cli():
    """Function to run translation from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Translate English medical text to German using local Ollama models")
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
    parser.add_argument("--model", type=str, default="mistral",
                       help="Ollama model name (e.g., 'mistral', 'llama2:latest', 'gemma:7b')")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434",
                       help="Ollama API URL")
    
    args = parser.parse_args()
    
    try:
        print(f"Initializing translation pipeline with Ollama model: {args.model}")
        translator = TranslationPipeline(
            model_name=args.model,
            ollama_url=args.ollama_url
        )
        
        print("Starting translation...")
        result = translator.translate_dataset(
            input_path=args.input,
            output_path=args.output,
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            sleep_time=args.sleep_time,
            max_retries=args.max_retries
        )
        
        print(f"\n{'='*60}")
        print("TRANSLATION COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"✓ Output saved to: {args.output}")
        print(f"✓ Translated {len(result)} samples")
        print(f"✓ Success rate: {(len(result)/args.n_samples*100):.1f}%")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Translation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(run_translation_from_cli())