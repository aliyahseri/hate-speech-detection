# Hate Speech Detection with Fine-Tuned Transformers

A binary hate-speech classifier built by fine-tuning BERT (with RoBERTa, DistilBERT and BERTweet variants) on two public English datasets: tweets from Davidson et al. and comments from UC Berkeley's Measuring Hate Speech. It includes a CLI for interactive predictions.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face Transformers](https://img.shields.io/badge/Transformers-4.39-FFD21E?logo=huggingface&logoColor=black)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

Content moderation has to tell actual hate speech apart from text that is merely offensive or profane. This project fine-tunes pretrained transformers to flag text as **hate speech** or **not hate speech**.

What's in the code:

- **Data pipeline** (`src/download_datasets.py`): downloads the Davidson et al. CSV from GitHub and the Berkeley *Measuring Hate Speech* dataset from Hugging Face, then maps both to binary labels.
- **Text cleaning** (`src/preprocess.py`): converts emojis to text, lowercases, strips URLs, @mentions, digits and punctuation, tokenizes, removes stopwords and lemmatizes with NLTK.
- **Fine-tuning** (`src/train.py`): `bert-base-uncased` with a classification head. Max length 128, AdamW (lr 2e-5), a linear LR schedule, gradient clipping and 3 epochs. The two datasets are combined and split 70/15/15 (stratified, seed 42). The model with the best validation accuracy is saved, and the test set is evaluated with accuracy, weighted F1 and a classification report. CUDA, Apple MPS or CPU is picked automatically.
- **Model variants**: `train_roberta.py` (`roberta-base`), `train_distilbert.py` (`distilbert-base-uncased`) and `train_bertweet.py` (`vinai/bertweet-base`) use the same pipeline with a different backbone.
- **Inference CLI** (`src/inference.py`): loads any of the four checkpoints and classifies one sentence (`--text`) or runs an interactive prompt. It shows the cleaned text and both class probabilities.
- **Terminal dashboard** (`dashboard/terminal_dashboard.py`): a formatted comparison table. Its numbers are **hardcoded placeholders, not results** (see [Results](#results)).

## Tech stack

| Area | Tools | Notes |
|---|---|---|
| Models | Hugging Face Transformers (BERT, RoBERTa, DistilBERT, BERTweet) | BERTweet was pretrained on English tweets, which suits the Davidson data |
| Training | PyTorch, `torch.optim.AdamW`, `get_linear_schedule_with_warmup` | Standard BERT fine-tuning recipe (lr 2e-5, 3 epochs, clip 1.0) |
| Data | pandas, Hugging Face `datasets`, `requests` | Downloads both sources reproducibly |
| Text cleaning | NLTK (tokenizer, stopwords, WordNet lemmatizer), `emoji` | Normalises noisy social-media text |
| Evaluation | scikit-learn (`train_test_split`, accuracy, F1, `classification_report`) | Stratified splits keep the class ratio |
| Dashboard | `tabulate`, `colorama` | Terminal table output |

## Demo / Screenshots

<!-- TODO: add a screenshot at docs/screenshot.png -->
![Screenshot](docs/screenshot.png)

*To do for the owner: after training, run `python src/inference.py --model bert` and capture a terminal session with a few example sentences showing the cleaned text, prediction and probabilities. Save it as `docs/screenshot.png`.*

## Dataset

**The data is not included in this repository** (everything in `data/` except `data/README.md` is git-ignored). The download script fetches it. More details are in [`data/README.md`](data/README.md).

| Dataset | Content | Size | Binary label used here | License |
|---|---|---|---|---|
| [Davidson et al. 2017](https://github.com/t-davidson/hate-speech-and-offensive-language): *Automated Hate Speech Detection and the Problem of Offensive Language* | Tweets labelled hate speech / offensive / neither | ~24.8k tweets | hate (class 0) → 1; offensive and neither → 0 | See source repo [TODO: confirm license] |
| [Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) (UC Berkeley D-Lab; Kennedy et al.) | Comments with per-annotator ratings, including `hatespeech` (0/1/2) and a continuous `hate_speech_score` | ~136k annotation rows for ~40k comments [TODO: confirm counts] | `hatespeech > 0.5` → 1 | CC BY 4.0 |

The combined data is imbalanced (most examples are "not hate"), which is why F1 and the per-class report matter more than accuracy.

Expected layout after preparation:

```
data/
├── github/labeled_data.csv           # original Davidson file
├── berkeley_raw.csv                  # raw Hugging Face dump
└── processed/
    ├── davidson_processed.csv        # tweet, label
    ├── berkeley_processed.csv        # text, label
    ├── davidson_cleaned.csv          # + processed_text   <- read by training
    └── berkeley_cleaned.csv          # + processed_text   <- read by training
```

**Model checkpoints** are not included either. Training writes them to `models/<model>/best_model.pt` (see [`models/README.md`](models/README.md)).

## Installation

Python 3.10+ is required (`src/inference.py` uses `str | None` type hints).

```bash
git clone https://github.com/aliyahseri/hate-speech-detection.git
cd hate-speech-detection

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The first run needs internet access to download the datasets, the NLTK resources (`punkt`, `stopwords`, `wordnet`, fetched automatically by `preprocess.py`) and the pretrained transformer weights.

## Usage

**1. Download and binarize the datasets.** The script uses `../data/...` paths, so run it from `src/`:

```bash
cd src
python download_datasets.py
cd ..
```

**2. Clean the text.** Run from the repository root. This adds the `processed_text` column that training reads:

```bash
python src/preprocess.py --input data/processed/davidson_processed.csv --output data/processed/davidson_cleaned.csv --text-column tweet
python src/preprocess.py --input data/processed/berkeley_processed.csv --output data/processed/berkeley_cleaned.csv --text-column text
```

**3. Train.** Create the checkpoint folder first, since the script doesn't. The training scripts take **no command-line arguments**; hyperparameters are set inside each script.

```bash
# macOS / Linux
mkdir -p models/bert
# Windows (PowerShell)
mkdir models/bert

python src/train.py                 # BERT -> models/bert/best_model.pt
# Variants (create models/roberta, models/distilbert, models/bertweet first):
python src/train_roberta.py
python src/train_distilbert.py
python src/train_bertweet.py
```

Training prints per-epoch validation loss, accuracy, weighted F1 and a classification report, then the test-set metrics.

**4. Predict.**

```bash
python src/inference.py --model bert --text "your sentence here"   # single prediction
python src/inference.py --model bert                               # interactive; empty line or 'exit' quits
```

The options are `--model {bert,roberta,distilbert,bertweet}` and `--device {cpu,cuda,mps,...}`.

**Dashboard (display only):** `python dashboard/terminal_dashboard.py`

## Results

**No verified results are recorded in this repository.** [TODO: confirm results]

The terminal dashboard (`dashboard/terminal_dashboard.py`) prints accuracy/F1 figures for six models, but **those values are hardcoded** in the script and don't come from any run in this repo. Three of the listed models (FastText, BERTweet-large, DistilRoBERTa) have no training code here, and the dashboard contradicts itself about which model is best. They should not be read as results.

To report results, run the training scripts and copy the **test-set** output into a table like this:

| Model | Test accuracy | Test F1 (weighted) | Hate-class F1 | Notes |
|---|---|---|---|---|
| BERT (`bert-base-uncased`) | [TODO] | [TODO] | [TODO] | 3 epochs, lr 2e-5 |
| RoBERTa (`roberta-base`) | [TODO] | [TODO] | [TODO] | |
| DistilBERT | [TODO] | [TODO] | [TODO] | |
| BERTweet | [TODO] | [TODO] | [TODO] | |

Before trusting the numbers, fix the train/test leakage in the Berkeley data and evaluate the saved best checkpoint (see Known issues).

## Project structure

```
hate-speech-detection/
├── src/
│   ├── download_datasets.py     # fetch + binarize Davidson and Berkeley data
│   ├── preprocess.py            # NLTK text-cleaning CLI
│   ├── train.py                 # BERT fine-tuning (main script)
│   ├── train_roberta.py         # RoBERTa variant
│   ├── train_distilbert.py      # DistilBERT variant
│   ├── train_bertweet.py        # BERTweet variant
│   ├── inference.py             # single-text / interactive prediction CLI
│   └── models.py                # unused helpers (custom DistilBERT head, loaders)
├── dashboard/
│   ├── terminal_dashboard.py    # hardcoded metrics table (display only)
│   └── README.md
├── data/README.md               # dataset instructions (data itself is git-ignored)
├── models/README.md             # checkpoint instructions (weights are git-ignored)
├── requirements.txt
├── setup.py                     # non-functional packaging stub (see Known issues)
└── LICENSE
```

## Challenges & what I learned

- **Offensive vs. hateful**: Davidson et al. separate offensive language from hate speech. Mapping only true hate speech to the positive class makes the task harder, and much more imbalanced, than a plain "toxic vs. clean" split.
- **Combining heterogeneous datasets**: tweets and longer comments differ in length, style and labelling scheme. Both had to be normalised to a single `text` / `label` format and a shared cleaning pipeline.
- **Annotator-level data**: the Berkeley dataset stores one row per annotator. Treating those rows as independent examples is a subtle source of leakage; aggregating them per comment (e.g. majority vote) is the fix. [TODO: confirm]
- **Comparing transformer backbones**: the same training loop is reused across BERT, RoBERTa, DistilBERT and the tweet-specific BERTweet. [TODO: confirm]
- **Device portability**: the training script picks CUDA, Apple MPS or a multi-threaded CPU automatically.

## Known issues & limitations

- **README-era CLI options don't exist**: the training scripts take no arguments (no argparse in `src/train.py`). Hyperparameters are hardcoded.
- **`setup.py` is non-functional**: `find_packages(where="src")` finds nothing because `src/` has no `__init__.py`. The entry points reference `src.train:main`, `src.inference:main` and a non-existent `src/evaluate.py`. `python_requires=">=3.8"` is also wrong (3.10+ syntax in `inference.py`). Install with `requirements.txt` instead. Consider deleting `setup.py` or rewriting it as a proper package.
- **No `src/evaluate.py`**: evaluation happens only at the end of the training scripts.
- **Checkpoint folder not created**: `train*.py` call `torch.save(..., 'models/<name>/best_model.pt')` without `os.makedirs`, so they crash at the first save unless the folder exists.
- **Model selection by accuracy**: on imbalanced data, the "best" checkpoint is chosen by validation accuracy rather than F1. The final test evaluation uses the **last-epoch** weights in memory, not the saved best checkpoint.
- **Berkeley labelling and leakage** (`src/download_datasets.py`): `hatespeech > 0.5` maps both "unclear" (1) and "yes" (2) to hate. Per-annotator rows aren't de-duplicated, so the same comment can land in train, validation and test.
- **Crash when the Berkeley download fails**: `download_berkeley_dataset()` returns `None`, and `main()` then calls `len(None)`.
- **Relative paths differ between scripts**: `download_datasets.py` must be run from `src/` (`../data`), while training must be run from the repo root (`data/`).
- **Heavy cleaning before transformers**: stopword removal (including negations such as "not") and lemmatization remove context that BERT-style models could use.
- **Unused module**: `src/models.py` isn't imported anywhere. `models.py` uses `OneHotEncoder(sparse=False)`, which fails on the pinned scikit-learn 1.4.2 (the argument is now `sparse_output`).
- **Near-duplicate training scripts**: four copies of the same script that differ only in model class and save path.
- **Dashboard metrics are hardcoded** and contradictory: the summary says DistilRoBERTa is best, but the highlight picks BERTweet_large by F1.
- **Leftover comment in `src/inference.py`**: `EXPLICIT_HATE_PHRASES` is an empty set with a note that the phrases were removed, so the keyword override never fires.
- **NLTK version sensitivity**: newer NLTK releases need the `punkt_tab` resource for `word_tokenize`. The pinned `nltk==3.8.1` avoids this.

## Future improvements

- Merge the four training scripts into one with argparse (`--model`, `--epochs`, `--batch-size`, `--lr`) and create output folders automatically.
- De-duplicate Berkeley comments (majority vote or mean score) before splitting, and treat "unclear" explicitly.
- Select checkpoints by hate-class F1 or macro-F1, and evaluate the saved best checkpoint on the test set. Handle class imbalance with class weights or threshold tuning.
- Feed transformers lightly normalised text (keep stopwords and casing where the tokenizer expects it).
- Add a real `evaluate.py` that writes metrics to JSON, and have the dashboard read that file.
- Fix or remove `setup.py`. Add tests for preprocessing and label mapping.

## Credits / Acknowledgements

- **Davidson et al. dataset**: T. Davidson, D. Warmsley, M. Macy, I. Weber. *Automated Hate Speech Detection and the Problem of Offensive Language*, ICWSM 2017. <https://github.com/t-davidson/hate-speech-and-offensive-language>
- **Measuring Hate Speech dataset**: UC Berkeley D-Lab. C. J. Kennedy, G. Bacon, A. Sahn, C. von Vacano, *Constructing interval variables via faceted Rasch measurement and multitask deep learning: a hate speech application*, arXiv:2009.10277 (2020). CC BY 4.0. <https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech>
- **Pretrained models** from the Hugging Face Hub: BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019), DistilBERT (Sanh et al., 2019), BERTweet (Nguyen et al., 2020).
- **NLTK** for tokenization, stopwords and lemmatization.

## Contributors

- Aliyah Seri
- Aly Adham (collaborator): [GitHub](https://github.com/AlyAdham)

## License

MIT, see [LICENSE](LICENSE). The datasets and pretrained models keep their own licenses.

## Author

**Aliyah Seri**

- GitHub: [`aliyahseri`](https://github.com/aliyahseri)
