import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def plot_confusion_matrix(y_true, y_pred, output_path='confusion_matrix.png'):
    """
    Create and save a confusion matrix plot
    
    Args:
    y_true (array-like): True labels
    y_pred (array-like): Predicted labels
    output_path (str): Path to save the confusion matrix plot
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a figure and set its size
    plt.figure(figsize=(10, 8))
    
    # Use seaborn to create a heatmap of the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Complex', 'Complex'], 
                yticklabels=['Not Complex', 'Complex'])
    
    # Set title and labels
    plt.title('Confusion Matrix for Complex Word Identification', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Load training data
    try:
        train_frames = []
        for folder in os.listdir(f'{os.getcwd()}/cwishareddataset/traindevset/'):
            if os.path.isdir(f'{os.getcwd()}/cwishareddataset/traindevset/{folder}'):
                for file in os.listdir(f'{os.getcwd()}/cwishareddataset/traindevset/{folder}'):
                    train_frames.append(pd.read_csv(f'{os.getcwd()}/cwishareddataset/traindevset/{folder}/{file}', delimiter='\t', header=None))

        df = pd.concat(train_frames)
        df.columns = [
            "HIT_ID", "Sentence", "Start_Offset", "End_Offset", 
            "Target_Word", "Native_Annotators", "Non_Native_Annotators", 
            "Native_Marked", "Non_Native_Marked", "Binary_Label", "Prob_Label"
        ]
    except Exception as e:
        print(f"Error loading training datasets: {e}")
        return
    
    # Preprocess training data
    def preprocess_dataframe(df, is_test=False):
        """
        Preprocess the input dataframe for complex word identification
        """
        # Validate column names
        required_cols = [
            "HIT_ID", "Sentence", "Start_Offset", "End_Offset", 
            "Target_Word", "Native_Annotators", "Non_Native_Annotators"
        ]
        
        # For training set, add additional required columns
        if not is_test:
            required_cols.extend(["Native_Marked", "Non_Native_Marked", "Binary_Label", "Prob_Label"])
        
        # Ensure all required columns are present
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Convert to numeric
        numeric_cols = [
            "Start_Offset", "End_Offset", 
            "Native_Annotators", "Non_Native_Annotators"
        ]
        
        # Add numeric columns for training set
        if not is_test:
            numeric_cols.extend(["Native_Marked", "Non_Native_Marked", "Binary_Label"])
        
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Basic text cleaning
        df['Sentence'] = df['Sentence'].str.strip()
        
        # Drop rows with invalid data
        df.dropna(subset=numeric_cols, inplace=True)
        
        return df
    
    df = preprocess_dataframe(df)
    
    # Split data
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['Binary_Label'],  # Stratified split
        random_state=42
    )
    
    # Load test data 
    try:
        test_frames = []
        # Modify this path to your test dataset location
        for folder in os.listdir(f'{os.getcwd()}/cwishareddataset/testset/'):
            if os.path.isdir(f'{os.getcwd()}/cwishareddataset/testset/{folder}'):
                for file in os.listdir(f'{os.getcwd()}/cwishareddataset/testset/{folder}'):
                    test_frames.append(pd.read_csv(f'{os.getcwd()}/cwishareddataset/testset/{folder}/{file}', delimiter='\t', header=None))

        test_df = pd.concat(test_frames)
        test_df.columns = [
            "HIT_ID", "Sentence", "Start_Offset", "End_Offset", 
            "Target_Word", "Native_Annotators", "Non_Native_Annotators", 
            "Native_Marked", "Non_Native_Marked", "Binary_Label", "Prob_Label"
        ]
    except Exception as e:
        print(f"Error loading test datasets: {e}")
        return
    
    # Save the original Binary_Label for evaluation
    test_labels = test_df['Binary_Label']
    
    # Preprocess test data by removing the last 4 columns
    test_df = test_df.drop(columns=["Native_Marked", "Non_Native_Marked", "Binary_Label", "Prob_Label"])
    
    class ComplexWordDataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_length=128, is_test=False):
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
            
            # Handle annotator agreement and labels differently for test vs. train datasets
            if not is_test:
                # Calculate annotator agreement
                try:
                    total_annotators = dataframe['Native_Annotators'] + dataframe['Non_Native_Annotators']
                    
                    # Convert to numpy array to avoid Series issue
                    self.annotator_agreement = torch.tensor(
                        (total_annotators).values.astype(np.float32)
                    )
                except Exception as e:
                    print(f"Error calculating annotator agreement: {e}")
                    # Fallback to a default tensor if calculation fails
                    self.annotator_agreement = torch.ones(len(dataframe), dtype=torch.float32)
                
                # Prepare labels - use int64
                self.labels = torch.tensor(
                    dataframe['Binary_Label'].values.astype(np.int64)
                )
            else:
                # For test dataset, create dummy agreement
                self.annotator_agreement = torch.ones(len(dataframe), dtype=torch.float32)
                
                # If Binary_Label is present, convert to labels, otherwise set to None
                if 'Binary_Label' in dataframe.columns:
                    self.labels = torch.tensor(
                        dataframe['Binary_Label'].values.astype(np.int64)
                    )
                else:
                    self.labels = None
            
            # Store additional metadata
            self.target_words = dataframe['Target_Word'].tolist()

        def __getitem__(self, idx):
            item = {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'annotator_agreement': self.annotator_agreement[idx],
            }
            
            # Only add labels if they exist
            if self.labels is not None:
                item['labels'] = self.labels[idx]
            
            return item

        def __len__(self):
            return len(self.target_words)

    # Initialize tokenizer and model
    model_name = 'bert-base-multilingual-cased'  # Good for multiple languages
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,  # Binary classification
    )
    
    # Add original labels to the test dataset for evaluation
    test_df['Binary_Label'] = test_labels
    test_dataset = ComplexWordDataset(test_df, tokenizer, is_test=True)
    
    # Load the trained model
    model = AutoModelForSequenceClassification.from_pretrained('./complex_word_model')
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Initialize Trainer for prediction
    trainer = Trainer(model=model)
    
    # Predict on test set
    test_predictions = trainer.predict(test_dataset)
    
    # Convert predictions to labels
    predicted_labels = np.argmax(test_predictions.predictions, axis=1)
    
    # Compute metrics on test set
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predicted_labels, average='binary'
    )
    accuracy = accuracy_score(test_labels, predicted_labels)
    
    print("\nTest Set Evaluation:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    
    # Create and save confusion matrix
    plot_confusion_matrix(test_labels, predicted_labels, 'confusion_matrix.png')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")

if __name__ == '__main__':
    main()