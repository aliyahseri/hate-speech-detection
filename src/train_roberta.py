import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import multiprocessing

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_model(model, train_loader, val_loader, device, num_epochs=3, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc='Training'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            model.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        print(f'Training loss: {avg_train_loss:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                
                predictions.extend(np.argmax(logits, axis=1))
                true_labels.extend(label_ids)
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(true_labels, predictions)
        val_f1 = f1_score(true_labels, predictions, average='weighted')
        
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        print(f'Validation F1 Score: {val_f1:.4f}')
        print('\n' + classification_report(true_labels, predictions))
        
        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'models/roberta/best_model.pt')
            print(f'Best model saved with accuracy: {best_accuracy:.4f}')
    
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            
            predictions.extend(np.argmax(logits, axis=1))
            true_labels.extend(label_ids)
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test F1 Score: {f1:.4f}')
    print('\n' + classification_report(true_labels, predictions))
    
    return accuracy, f1

def main():
    # Check for available device (MPS for M1/M2, CUDA for NVIDIA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f'Using MPS (Apple Silicon GPU): {device}')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using CUDA GPU: {device}')
    else:
        device = torch.device('cpu')
        print(f'Using CPU: {device}')
        # Set number of threads for PyTorch to use all CPU cores
        torch.set_num_threads(multiprocessing.cpu_count())
        print(f'Using {multiprocessing.cpu_count()} CPU cores')
    
    # Load and preprocess data
    print('Loading and preprocessing data...')
    
    # Load Davidson dataset
    davidson_df = pd.read_csv('data/processed/davidson_cleaned.csv')
    print(f'Loaded Davidson dataset: {len(davidson_df)} samples')
    
    # Load Berkeley dataset
    berkeley_df = pd.read_csv('data/processed/berkeley_cleaned.csv')
    print(f'Loaded Berkeley dataset: {len(berkeley_df)} samples')
    
    # Combine datasets
    davidson_df = davidson_df[['processed_text', 'label']].rename(columns={'processed_text': 'text'})
    berkeley_df = berkeley_df[['processed_text', 'label']].rename(columns={'processed_text': 'text'})
    
    # Remove any rows with missing values
    davidson_df = davidson_df.dropna()
    berkeley_df = berkeley_df.dropna()
    
    # Combine datasets
    combined_df = pd.concat([davidson_df, berkeley_df], ignore_index=True)
    print(f'Combined dataset: {len(combined_df)} samples')
    print(f'Label distribution: {combined_df["label"].value_counts().to_dict()}')
    
    # Split into train, validation, and test sets
    train_df, temp_df = train_test_split(
        combined_df, test_size=0.3, random_state=SEED, stratify=combined_df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['label']
    )
    
    print(f'Train set: {len(train_df)} samples')
    print(f'Validation set: {len(val_df)} samples')
    print(f'Test set: {len(test_df)} samples')
    
    # Initialize tokenizer and model
    print('Initializing RoBERTa model...')
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    ).to(device)
    
    # Set batch size based on device
    if device.type == 'cpu':
        batch_size = 32  # Larger batch size for CPU
        num_workers = multiprocessing.cpu_count()  # Use all CPU cores
    else:
        batch_size = 16  # Smaller batch size for GPU
        num_workers = 4  # Fewer workers for GPU
    
    print(f'Using batch size: {batch_size}')
    print(f'Using {num_workers} workers for data loading')
    
    # Create datasets and dataloaders
    train_dataset = HateSpeechDataset(train_df['text'].values, train_df['label'].values, tokenizer)
    val_dataset = HateSpeechDataset(val_df['text'].values, val_df['label'].values, tokenizer)
    test_dataset = HateSpeechDataset(test_df['text'].values, test_df['label'].values, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Train the model
    print('Starting training...')
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=3,
        learning_rate=2e-5
    )
    
    # Evaluate the model
    print('Evaluating model on test set...')
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()
