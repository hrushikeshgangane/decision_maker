"""Edit STATE, QUESTION and OPTIONS, then run: python run.py.

The model downloads once from Hugging Face and is cached automatically.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

# Once you publish the Hugging Face model, replace this with the final public
# repository ID, e.g. "your-name/decision_maker". Users can also pass --model.
MODEL_ID = "HrushikeshGangane/decision_maker"

# ----- Edit only these values -------------------------------------------------
STATE = "I was charged twice for the same purchase."
QUESTION = "Which issue is this?"
OPTIONS = ["delivery problem", "duplicate charge", "wrong item"]
# -----------------------------------------------------------------------------


def load_model(model_id: str):
    if model_id.startswith("YOUR_HUGGING_FACE_USERNAME/"):
        raise ValueError("Set MODEL_ID at the top of run.py or pass --model your-name/decision_maker")
    directory = Path(snapshot_download(repo_id=model_id))
    sys.path.insert(0, str(directory))
    decision_model = importlib.import_module("decision_model")
    data_utils = importlib.import_module("data_utils")
    config = json.loads((directory / "model_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer")
    model = decision_model.DecisionModel(decision_model.ModelConfig(**config["model_config"]))
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(load_file(directory / "model.safetensors"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), tokenizer, config, data_utils, device


def decide(model, tokenizer, config, data_utils, device, state: str, question: str, options: list[str]):
    row = {"id": "request", "state": state, "question": question, "options": options, "label": 0}
    batch = data_utils.make_collate(tokenizer, config["max_length"])([row])
    tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor) and key != "labels"}
    with torch.no_grad():
        probabilities = (model(**tensors)[0] / float(config["temperature"])).softmax(-1).cpu()
    return {
        "choice": options[int(probabilities.argmax())],
        "probabilities": {option: round(probabilities[index].item(), 6) for index, option in enumerate(options)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID, help="Hugging Face model ID")
    args = parser.parse_args()
    model, tokenizer, config, data_utils, device = load_model(args.model)
    print(json.dumps(decide(model, tokenizer, config, data_utils, device, STATE, QUESTION, OPTIONS), indent=2))
