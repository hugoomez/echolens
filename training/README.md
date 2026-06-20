\# EchoLens — training module



Trains the baseline audio classifier (AudioCNN) on UrbanSound8K log-mel

spectrograms, and exports the artifacts the browser needs for inference.



\## Setup



```bash

py -3.11 -m venv .venv

.venv\\Scripts\\activate

pip install -e ".\[dev]"

```



If you have an NVIDIA GPU, install CUDA-enabled PyTorch \*\*before\*\* the

line above, matching your driver's supported CUDA version:

```bash

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

```



\## Download the dataset



```bash

python ../scripts/download\_data.py

```

Downloads and validates UrbanSound8K (\~6GB) into `data/urbansound8k/`.



\## Train



From the repo root (not from `training/`):

```bash

python -m echolens.train --config-name baseline

```

Runs the honest 10-fold cross-validation, then trains and saves the final

deployable model (trained on folds 1-9, validated on fold 10).



Outputs:

\- `training/checkpoint\_best.pt` — final model weights

\- `training/logs/cv\_results.json` — per-fold and mean CV accuracy

\- `web/public/models/norm\_stats.json` — normalization stats for inference



> Note: on Windows, `DataLoader` workers must stay at `num\_workers=0` —

> `soundata`'s internal objects aren't picklable, which breaks

> multiprocessing on this OS.



\## Evaluate



```bash

python -m echolens.evaluate

```

Generates `docs/metrics.md` and `docs/confusion\_matrix.png` from the

final model's held-out fold 10 predictions.



\## Run tests



```bash

ruff check .

ruff format --check .

mypy src

pytest -q

```

Dataset-dependent tests skip automatically if `data/urbansound8k/` isn't

present (e.g., in CI).

