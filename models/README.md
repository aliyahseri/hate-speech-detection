# Model checkpoints

No checkpoints are included (everything under `models/` except this README is git-ignored). They are produced by
the training scripts, which save a plain `state_dict` of the Hugging Face sequence-classification model:

| Script | Base model | Saved to |
|---|---|---|
| `src/train.py` | `bert-base-uncased` | `models/bert/best_model.pt` |
| `src/train_roberta.py` | `roberta-base` | `models/roberta/best_model.pt` |
| `src/train_distilbert.py` | `distilbert-base-uncased` | `models/distilbert/best_model.pt` |
| `src/train_bertweet.py` | `vinai/bertweet-base` | `models/bertweet/best_model.pt` |

The scripts don't create these folders. Create them before training, or the first save fails:

```bash
# macOS / Linux
mkdir -p models/bert models/roberta models/distilbert models/bertweet
# Windows (PowerShell)
mkdir models/bert, models/roberta, models/distilbert, models/bertweet
```

The checkpoint is the epoch with the highest **validation accuracy**. Note that the test metrics printed at the end of
training come from the **last** epoch's weights, not from this saved checkpoint.

## Using a checkpoint

```bash
python src/inference.py --model bert --text "some sentence to classify"
python src/inference.py --model distilbert          # interactive mode
```

`--model` is one of `bert`, `roberta`, `distilbert`, `bertweet`. `--device` is optional (`cpu`, `cuda`, `mps`).

## Performance

No verified metrics have been recorded for these checkpoints yet. [TODO: confirm results] Record the
test-set accuracy, F1 and per-class report from your own training run here.
