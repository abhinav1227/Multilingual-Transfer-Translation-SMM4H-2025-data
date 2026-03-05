"""
Model training module for SMM4H multilingual modeling
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import json
import os

class TextDataset(Dataset):
    """Custom Dataset for transformer models"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    
    def __len__(self):
        return len(self.labels)

class MultilingualModelTrainer:
    def __init__(self, model_name="bert-base-multilingual-cased", num_labels=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        ).to(self.device)
        
    def create_dataloader(self, texts, labels, tokenizer, batch_size=16, shuffle=False):
        """Create DataLoader from texts and labels"""
        tokenized = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        dataset = TextDataset(tokenized, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train(self, train_loader, val_loader=None, epochs=3, learning_rate=2e-5):
        """Train model on given data"""
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        training_stats = []
        best_f1 = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            self.model.train()
            total_train_loss = 0
            train_preds, train_labels = [], []
            
            for batch in tqdm(train_loader, desc="Training"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Collect predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(batch['labels'].cpu().numpy())
            
            # Calculate training metrics
            avg_train_loss = total_train_loss / len(train_loader)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='weighted')
            
            # Validation
            val_metrics = None
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                
                # Save best model
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    self.save_model("models/best_model")
            
            # Store stats
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_acc,
                'train_f1': train_f1,
            }
            if val_metrics:
                epoch_stats.update(val_metrics)
            
            training_stats.append(epoch_stats)
            
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
            if val_metrics:
                print(f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        return training_stats
    
    def evaluate(self, test_loader):
        """Evaluate model on test data"""
        self.model.eval()
        all_preds, all_labels = [], []
        total_eval_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                total_eval_loss += loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        avg_loss = total_eval_loss / len(test_loader)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': avg_loss
        }
    
    def save_model(self, path):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model from disk"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        print(f"Model loaded from {path}")