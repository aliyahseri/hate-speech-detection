#to run code type this -> python src/inference.py --model type of model u wanna try

import os
import warnings
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from preprocess import preprocess_text

# Ignore warnings about classifier initialization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=r".*Some weights of .* were not initialized.*")

# Explicit hate phrases that should always be flagged (case-insensitive)
# Note: Explicit phrases have been removed for submission
EXPLICIT_HATE_PHRASES = set()

def contains_explicit_hate(text: str) -> Optional[tuple[float, float]]:
    """Check if text contains any explicit hate phrases.
    Returns (p_not_hate, p_hate) if matched, None otherwise."""
    text_lower = ' ' + text.lower() + ' '  # Add spaces for whole word matching
    for phrase in EXPLICIT_HATE_PHRASES:
        # Check for phrase as whole word
        if f' {phrase.lower()} ' in text_lower:
            return 0.0, 1.0  # Force label as hate speech with 100% confidence
        # Also check if phrase is at start/end of string
        if text_lower.startswith(f'{phrase} ') or text_lower.endswith(f' {phrase}'):
            return 0.0, 1.0
    return None
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
)


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    checkpoint_path: Path


def _get_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_model_spec(repo_root: Path, model_key: str) -> ModelSpec:
    specs: dict[str, ModelSpec] = {
        "bert": ModelSpec(
            model_name="bert-base-uncased",
            checkpoint_path=repo_root / "models" / "bert" / "best_model.pt",
        ),
        "roberta": ModelSpec(
            model_name="roberta-base",
            checkpoint_path=repo_root / "models" / "roberta" / "best_model.pt",
        ),
        "distilbert": ModelSpec(
            model_name="distilbert-base-uncased",
            checkpoint_path=repo_root / "models" / "distilbert" / "best_model.pt",
        ),
        "bertweet": ModelSpec(
            model_name="vinai/bertweet-base",
            checkpoint_path=repo_root / "models" / "bertweet" / "best_model.pt",
        ),
    }

    if model_key not in specs:
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {', '.join(specs.keys())}")
    return specs[model_key]


def load_model(model_key: str, device: torch.device) -> tuple[torch.nn.Module, object]:
    repo_root = Path(__file__).resolve().parents[1]
    spec = _get_model_spec(repo_root, model_key)

    if not spec.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{spec.checkpoint_path}'. "
            f"Train the model first or point to the correct repo."  # noqa: E501
        )

    if model_key == "bert":
        tokenizer = BertTokenizer.from_pretrained(spec.model_name)
        model = BertForSequenceClassification.from_pretrained(spec.model_name, num_labels=2)
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(spec.model_name, num_labels=2)

    state_dict = torch.load(spec.checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def predict(model: torch.nn.Module, tokenizer, text: str, device: torch.device, max_length: int = 128):
    # Check explicit phrases in the ORIGINAL text before any preprocessing
    explicit_probs = contains_explicit_hate(text)
    if explicit_probs is not None:
        p_not_hate, p_hate = explicit_probs
        label = "hate_speech"
        return label, p_hate, p_not_hate, p_hate

    # If no explicit match, preprocess and use the model
    processed_text = preprocess_text(text)
    enc = tokenizer(
        processed_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    probs = torch.softmax(out.logits, dim=-1)[0]
    pred = int(torch.argmax(probs).item())
    conf = float(probs[pred].item())
    label = "hate_speech" if pred == 1 else "not_hate_speech"
    p_not_hate = float(probs[0].item())
    p_hate = float(probs[1].item())
    return label, conf, p_not_hate, p_hate


def _format_prediction(label: str) -> str:
    return "Hate Speech" if label == "hate_speech" else "Not Hate Speech"


def _print_result(test_idx: int, original: str, processed: str, label: str, p_not_hate: float, p_hate: float) -> None:
    print(f"Test {test_idx}:")
    print(f"Original: {original}")
    print(f"Processed: {processed}")
    print(f"Prediction: {_format_prediction(label)}")
    print(f"Probabilities - Not Hate: {p_not_hate:.3f}, Hate: {p_hate:.3f}")
    print("----------------------------------------")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive hate speech detection")
    parser.add_argument(
        "--model",
        default="bert",
        choices=["bert", "roberta", "distilbert", "bertweet"],
        help="Which trained checkpoint to use",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, mps, etc.")
    parser.add_argument("--text", default=None, help="Run one prediction and exit")
    args = parser.parse_args()

    device = _get_device(args.device)
    model, tokenizer = load_model(args.model, device)

    if args.text is not None:
        original = args.text
        # Don't preprocess here - let predict() handle it
        label, _conf, p_not_hate, p_hate = predict(model, tokenizer, original, device)
        # Get processed text for display
        processed = preprocess_text(original)
        _print_result(1, original, processed, label, p_not_hate, p_hate)
        return 0

    print("Type a sentence and press Enter. Type 'exit' or an empty line to quit.")
    test_idx = 1
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break

        if text == "" or text.lower() == "exit":
            break

        original = text
        # Don't preprocess here - let predict() handle it after checking for explicit hate
        label, _conf, p_not_hate, p_hate = predict(model, tokenizer, original, device)
        # Get processed text for display
        processed = preprocess_text(original)
        _print_result(test_idx, original, processed, label, p_not_hate, p_hate)
        test_idx += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
