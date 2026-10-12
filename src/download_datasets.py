import os
import pandas as pd
from datasets import load_dataset
import requests
from pathlib import Path

def download_file(url, local_path):
    """Download a file from a URL to a local path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def download_davidson_dataset():
    """Download the Davidson et al. hate speech dataset."""
    print("Downloading Davidson et al. dataset...")
    url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
    os.makedirs("../data/github", exist_ok=True)
    output_path = "../data/github/labeled_data.csv"
    
    download_file(url, output_path)
    print(f"Downloaded to {output_path}")
    
    # Load and preprocess the dataset
    df = pd.read_csv(output_path)
    
    # Create binary labels: 1 for hate speech, 0 otherwise
    df['label'] = (df['class'] == 0).astype(int)  # class 0 is hate speech in this dataset
    
    # Save processed data
    processed_path = "../data/processed/davidson_processed.csv"
    df[['tweet', 'label']].to_csv(processed_path, index=False)
    print(f"Processed data saved to {processed_path}")
    return df

def download_berkeley_dataset():
    """Download and preprocess the Berkeley hate speech dataset."""
    print("Downloading Berkeley hate speech dataset...")
    
    try:
        # Load the dataset from Hugging Face with the default config
        dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
        
        # Convert to pandas DataFrame for easier manipulation
        df = dataset['train'].to_pandas()
        
        # Save raw data
        raw_path = "../data/berkeley_raw.csv"
        df.to_csv(raw_path, index=False)
        print(f"Raw data saved to {raw_path}")
        
        # Check if 'hatespeech' column exists, if not, use 'hate_speech_score'
        hate_score_col = 'hatespeech' if 'hatespeech' in df.columns else 'hate_speech_score'
        
        # For binary classification, we'll use the hate score column
        # which is a float between 0 and 1. We'll use 0.5 as threshold
        df['label'] = (df[hate_score_col] > 0.5).astype(int)
        
        # Make sure text column exists
        text_col = 'text' if 'text' in df.columns else 'comment_text'
        
        # Save processed data
        processed_path = "../data/processed/berkeley_processed.csv"
        df[[text_col, 'label']].rename(columns={text_col: 'text'}).to_csv(processed_path, index=False)
        print(f"Processed data saved to {processed_path}")
        return df
        
    except Exception as e:
        print(f"Error downloading Berkeley dataset: {str(e)}")
        print("Please check the dataset availability at: https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech")
        return None

def main():
    # Create necessary directories
    os.makedirs("../data/processed", exist_ok=True)
    
    # Download and process datasets
    print("="*50)
    davidson_df = download_davidson_dataset()
    print("\n" + "="*50)
    berkeley_df = download_berkeley_dataset()
    
    print("\n" + "="*50)
    print("Dataset download and preprocessing complete!")
    print(f"Davidson dataset size: {len(davidson_df)}")
    print(f"Berkeley dataset size: {len(berkeley_df)}")

if __name__ == "__main__":
    main()
