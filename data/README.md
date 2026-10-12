# Data directory

No data is included in the repository. Everything under `data/` except this README is git-ignored.

## Datasets

| Dataset | Content | Size | License / terms |
|---|---|---|---|
| **Davidson et al. (2017)**, *Automated Hate Speech Detection and the Problem of Offensive Language* — [t-davidson/hate-speech-and-offensive-language](https://github.com/t-davidson/hate-speech-and-offensive-language) | Tweets labelled by crowd workers as hate speech (class 0), offensive (1) or neither (2) | ~24.8k tweets | See the source repository [TODO: confirm license] |
| **Measuring Hate Speech** (UC Berkeley D-Lab; Kennedy et al.) — [ucberkeley-dlab/measuring-hate-speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) | Social-media comments with one row **per annotator**, including a `hatespeech` rating (0 = no, 1 = unclear, 2 = yes) and a continuous `hate_speech_score` | ~136k annotation rows for ~40k unique comments [TODO: confirm counts] | CC BY 4.0 |

Both are turned into a **binary** label (`1` = hate speech, `0` = not):

- Davidson: `label = 1` only for class 0 (hate). Offensive-but-not-hateful tweets count as `0`.
- Berkeley: `label = 1` if `hatespeech > 0.5`, so both "unclear" (1) and "yes" (2) count as hate. Annotator rows are not de-duplicated (see the main README's Known issues).

## Producing the files `src/train.py` reads

`src/download_datasets.py` uses paths relative to `src/` (`../data/...`), so run it from inside `src/`. All other commands run from the repository root.

```bash
# 1. Download + binarize (writes data/github/labeled_data.csv, data/berkeley_raw.csv,
#    data/processed/davidson_processed.csv [tweet,label], data/processed/berkeley_processed.csv [text,label])
cd src
python download_datasets.py
cd ..

# 2. Clean the text (adds a processed_text column)
python src/preprocess.py --input data/processed/davidson_processed.csv --output data/processed/davidson_cleaned.csv --text-column tweet
python src/preprocess.py --input data/processed/berkeley_processed.csv --output data/processed/berkeley_cleaned.csv --text-column text
```

## Expected layout

```
data/
├── README.md
├── berkeley_raw.csv                 # full Hugging Face dump (large)
├── github/
│   └── labeled_data.csv             # original Davidson CSV
└── processed/
    ├── davidson_processed.csv       # tweet, label
    ├── berkeley_processed.csv       # text, label
    ├── davidson_cleaned.csv         # tweet, label, processed_text   <- read by train*.py
    └── berkeley_cleaned.csv         # text, label, processed_text    <- read by train*.py
```
