import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def check_cuda_availability():
    """Check and print CUDA availability and details"""
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
        print("CUDA Device Count:", torch.cuda.device_count())
        print("Current CUDA Device:", torch.cuda.current_device())
        print("CUDA Capability:", torch.cuda.get_device_capability(0))

class ComplexWordDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        # Prepare input for tokenization with target word context
        context_sentences = []
        for idx, row in dataframe.iterrows():
            # Safely handle sentence tokenization
            try:
                modified_sentence = (
                    row['Sentence'][:int(row['Start_Offset'])] + 
                    f"[{row['Target_Word']}]" + 
                    row['Sentence'][int(row['End_Offset']):]
                )
                context_sentences.append(modified_sentence)
            except Exception as e:
                print(f"Error processing sentence at index {idx}: {e}")
                context_sentences.append(row['Sentence'])
        
        # Tokenize with emphasis on target word
        self.encodings = tokenizer(
            context_sentences,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Calculate annotator agreement
        try:
            total_annotators = dataframe['Native_Annotators'] + dataframe['Non_Native_Annotators']
            total_marked = dataframe['Native_Marked'] + dataframe['Non_Native_Marked']
            
            # Convert to numpy array to avoid Series issue
            self.annotator_agreement = torch.tensor(
                (total_marked / total_annotators).values.astype(np.float32)
            )
        except Exception as e:
            print(f"Error calculating annotator agreement: {e}")
            # Fallback to a default tensor if calculation fails
            self.annotator_agreement = torch.ones(len(dataframe), dtype=torch.float32)
        
        # Prepare labels - use int64
        self.labels = torch.tensor(
            dataframe['Binary_Label'].values.astype(np.int64)
        )
        
        # Store additional metadata
        self.target_words = dataframe['Target_Word'].tolist()

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'annotator_agreement': self.annotator_agreement[idx],
            'labels': self.labels[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)

def preprocess_dataframe(df):
    """
    Preprocess the input dataframe for complex word identification
    """
    # Validate column names
    required_cols = [
        "HIT_ID", "Sentence", "Start_Offset", "End_Offset", 
        "Target_Word", "Native_Annotators", "Non_Native_Annotators", 
        "Native_Marked", "Non_Native_Marked", "Binary_Label", "Prob_Label"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Convert to numeric
    numeric_cols = [
        "Start_Offset", "End_Offset", 
        "Native_Annotators", "Non_Native_Annotators", 
        "Native_Marked", "Non_Native_Marked", 
        "Binary_Label"
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Basic text cleaning
    df['Sentence'] = df['Sentence'].str.strip()
    
    # Drop rows with invalid data
    df.dropna(subset=numeric_cols, inplace=True)
    
    return df

def compute_metrics(pred):
    """Compute evaluation metrics"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # Print CUDA availability
    check_cuda_availability()

    # Load data from multiple datasets
    try:
        import os
        frames = []
        for folder in os.listdir(f'{os.getcwd()}/cwishareddataset/traindevset/'):
            if os.path.isdir(f'{os.getcwd()}/cwishareddataset/traindevset/{folder}'):
                for file in os.listdir(f'{os.getcwd()}/cwishareddataset/traindevset/{folder}'):
                    frames.append(pd.read_csv(f'{os.getcwd()}/cwishareddataset/traindevset/{folder}/{file}', delimiter='\t', header=None))

        df = pd.concat(frames)
        df.columns = [
            "HIT_ID", "Sentence", "Start_Offset", "End_Offset", 
            "Target_Word", "Native_Annotators", "Non_Native_Annotators", 
            "Native_Marked", "Non_Native_Marked", "Binary_Label", "Prob_Label"
        ]
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Preprocess data
    df = preprocess_dataframe(df)
    
    # Split data
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['Binary_Label'],  # Stratified split
        random_state=42
    )
    
    # Initialize tokenizer and model
    model_name = 'bert-base-multilingual-cased'  # Good for multiple languages
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,  # Binary classification
    )
    
    # Prepare datasets
    train_dataset = ComplexWordDataset(train_df, tokenizer)
    val_dataset = ComplexWordDataset(val_df, tokenizer)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Training arguments with GPU-friendly settings
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=16,  # Reduce if out of memory
        per_device_eval_batch_size=64,
        warmup_steps=500,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        save_total_limit=1
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train model with error handling
    try:
        # Reduce batch size if out of memory
        trainer.train()
    except RuntimeError as e:
        print(f"Training error: {e}")
        print("Suggestions:")
        print("1. Reduce batch size in TrainingArguments")
        print("2. Check GPU memory")
        print("3. Ensure CUDA is properly installed")
        return
    
    # Evaluate model
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)
    
    # Save model
    trainer.save_model('./complex_word_model')
    
    # Optional: Probabilistic prediction
    prob_predictions = trainer.predict(val_dataset)
    print("Probabilistic Predictions Shape:", prob_predictions.predictions.shape)

if __name__ == '__main__':
    main()