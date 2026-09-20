# decision_maker runner

The smallest local runner for the `decision_maker` model. It starts no server
and needs no SDK: edit three variables, then run one Python file.

## Use

```powershell
git clone https://github.com/hrushikeshgangane/decision_maker_runner
cd decision_maker_runner
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

## Notes

- The model supports a variable number of categorical options, including yes/no
  and ordered score levels.
- Inputs have a 256-token structured limit.
- Temperature scaling changes probability confidence but never changes the
  winning option or option ranking.
