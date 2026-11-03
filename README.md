# Hate Speech Detection

## Introduction
This project detects hate speech in English text by fine-tuning transformer models. Tweets and comments are cleaned with an NLP preprocessing pipeline and used to train a binary classifier (hate speech vs. not). A command-line tool classifies new text.

## Models Used
* BERT (main model)
* RoBERTa
* DistilBERT
* BERTweet

## Training
* Download the datasets (run inside `src/`, then return to the project root): `cd src && python download_datasets.py && cd ..`
* Clean the text: `python src/preprocess.py --input data/processed/davidson_processed.csv --output data/processed/davidson_cleaned.csv --text-column tweet` (repeat for `berkeley_processed.csv` with `--text-column text`)
* Train BERT: `python src/train.py`
* Other models: `src/train_roberta.py`, `src/train_distilbert.py`, `src/train_bertweet.py`

## Usage
* `python src/inference.py --model bert --text "your sentence here"`

## Datasets
* [Davidson et al. hate speech dataset](https://github.com/t-davidson/hate-speech-and-offensive-language) (tweets)
* [UC Berkeley Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) (comments)

## Requirements
* python 3.10+
* pytorch
* transformers
* pandas
* scikit-learn
* nltk

## Contributors
* Aliyah Seri
* Aly Adham (collaborator): [GitHub](https://github.com/AlyAdham)
