# Terminal dashboard

`terminal_dashboard.py` prints a colour-coded comparison table of model metrics in the terminal. It uses
[`tabulate`](https://pypi.org/project/tabulate/) and [`colorama`](https://pypi.org/project/colorama/).

```bash
python dashboard/terminal_dashboard.py
```

## The numbers are not results

The metrics and dataset sizes are **hardcoded** in `ModelMetrics.models_data` and `ModelMetrics.dataset_info`.
They are not read from any training run or saved output. Several of the listed models (FastText, BERTweet-large,
DistilRoBERTa) have no training code in this repository. The text is also inconsistent: the summary calls
DistilRoBERTa the best model, but the "best model" highlight picks the highest F1 in the table, which is
BERTweet_large. Treat the dashboard as a display template and **don't** quote its values as project results.

To use it for real results, replace the dictionaries with metrics from your own runs (for example, the
test-set output printed at the end of `src/train.py`).
