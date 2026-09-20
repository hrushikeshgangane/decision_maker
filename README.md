# decision_maker

The smallest local runner for the `decision_maker` model. It starts no server
and needs no SDK: edit three variables, then run one Python file.

## Use

```powershell
git clone https://github.com/hrushikeshgangane/decision_maker
cd decision_maker
python -m pip install -r requirements.txt
python run.py --model HrushikeshGangane/decision_maker
```

### Optional NVIDIA GPU setup

The runner uses CUDA automatically when PyTorch can access an NVIDIA GPU.
Check the CUDA version supported by your installed driver with:

```powershell
nvidia-smi
```

If an NVIDIA GPU is available, install the PyTorch build for the CUDA version
shown by `nvidia-smi` before installing the remaining requirements. For example,
for CUDA 12.1:

```powershell
python -m pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The equivalent CUDA wheel line is also included, commented out, in
`requirements.txt`. Use the matching PyTorch CUDA index (such as `cu121` or
`cu124`) for your system; do not install a CUDA build that your NVIDIA driver
does not support.

The first run downloads and caches the model from Hugging Face. Later runs use
the local Hugging Face cache.

Edit these values at the top of `run.py`:

```python
STATE = "I was charged twice for the same purchase."
QUESTION = "Which issue is this?"
OPTIONS = ["delivery problem", "duplicate charge", "wrong item"]
```

The script prints the selected option and calibrated probability distribution.
Use `decide(...)` from `run.py` in your own Python loop if desired.

## Model architecture

`decision_maker` is a bidirectional decision encoder with query/key option
scoring. It is designed to rank a supplied set of choices, not to generate an
open-ended answer.

1. The state, question, and options are tokenized together with structural
   markers such as `[STATE]`, `[QUESTION]`, `[OPTION]`, and `[DECIDE]`.
2. A pretrained bidirectional Transformer encoder, such as BERT, reads the
   complete structured input so every token can use context from both
   directions.
3. Additional Transformer layers let the decision representation reason over
   the state, question, and option text together.
4. The hidden vector at `[DECIDE]` becomes a query. Each option representation
   becomes a key. Their scaled dot products produce one logit per option.
5. Padding slots are masked, and softmax converts the logits into a probability
   distribution over the real options.

In short, the model computes `score(question, state, option)` for each supplied
option and chooses the highest-scoring one. The number of options can vary from
request to request.

## Why use it for decisions?

This model is a focused, System 1-style decision component: it makes a fast
classification or ranking judgment from a compact structured input. Compared
with a general-purpose generative LLM, it can be a better fit when the answer
must be one of a known set of choices:

- **Predictable output:** it returns an option and probabilities instead of
  free-form text that must be parsed.
- **Lower serving cost and latency:** it performs one bounded encoder pass and
  a small scoring operation, with no token-by-token generation.
- **Consistent behavior:** the option set is explicit, making decisions easier
  to test, monitor, and reproduce.
- **Useful confidence scores:** probabilities can support thresholds,
  escalation, abstention, and human review workflows.
- **Local deployment:** the model can run without an external LLM API, which
  can help with privacy, availability, and predictable operating costs.

An LLM remains preferable when the task requires explanation, information
extraction from arbitrary documents, tool use, multi-step planning, or answers
outside a predefined option set. A practical system can use this model for the
high-volume first decision and route uncertain cases to a larger model or a
human.

## Example decisions

Each request is expressed as a `STATE`, a `QUESTION`, and an `OPTIONS` list.
The options should be mutually understandable labels for the decision being
made.

### Support triage

```python
STATE = "The customer received the order, but the package contained the wrong size."
QUESTION = "Which support category should handle this case?"
OPTIONS = ["refund", "replacement", "technical support", "delivery investigation"]
```

### Risk review

```python
STATE = "A new login came from an unfamiliar country two minutes after a password change."
QUESTION = "What action should be taken?"
OPTIONS = ["allow", "request verification", "temporarily block"]
```

### Ordered assessment

```python
STATE = "The applicant meets the core requirements and has several directly relevant projects."
QUESTION = "How strong is the match?"
OPTIONS = ["weak", "moderate", "strong"]
```

## Notes

- The model supports a variable number of categorical options, including yes/no
  and ordered score levels.
- Inputs have a 256-token structured limit.
- Temperature scaling changes probability confidence but never changes the
  winning option or option ranking.

## System 1 decision model

This model is intended to act as a fast System 1 layer for general-purpose
decision making: given a state, a question, and a known set of choices, it
produces a direct ranking instead of composing a long answer. Because it uses a
bounded encoder pass and scores only the supplied options, inference is usually
much cheaper than asking a general-purpose generative LLM to reason and produce
text for every request. It can therefore handle high-volume choices such as
triage, routing, risk flags, and approval steps with lower latency, simpler
monitoring, and predictable output costs. The exact savings depend on the model
size, hardware, batching, and traffic, so this should be verified with a
workload-specific benchmark. Complex cases can still be escalated to a larger
LLM or a human reviewer.
