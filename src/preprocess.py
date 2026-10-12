import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

try:
    import emoji  # type: ignore
except ModuleNotFoundError:
    emoji = None

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    """
    Preprocess the text data by:
    1. Converting to lowercase
    2. Removing URLs
    3. Removing mentions (@username)
    4. Removing special characters and numbers
    5. Tokenizing
    6. Removing stopwords
    7. Lemmatizing
    8. Converting emojis to text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert emojis to text
    if emoji is not None:
        text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back to string
    return ' '.join(tokens)

def preprocess_dataset(input_path, output_path, text_column='text'):
    """
    Preprocess a dataset and save the processed version.
    
    Args:
        input_path (str): Path to the input CSV file
        output_path (str): Path to save the processed CSV file
        text_column (str): Name of the column containing text data
    """
    print(f"Preprocessing {input_path}...")
    
    # Read the dataset
    df = pd.read_csv(input_path)
    
    # Preprocess text
    df['processed_text'] = df[text_column].apply(preprocess_text)
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess text data for hate speech detection')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to save processed CSV file')
    parser.add_argument('--text-column', type=str, default='text', 
                       help='Name of the column containing text data')
    
    args = parser.parse_args()
    
    preprocess_dataset(args.input, args.output, args.text_column)
