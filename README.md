# SMoE

## Installation

```bash
pip install -r requirements.txt
bash scripts/fetch_histoformer.sh
```

Install the OpenCLIP dependencies from [DA-CLIP](https://github.com/Algolzw/daclip-uir).
Provide the restoration and conditioner checkpoints locally.


## Training

```bash
torchrun --standalone --nproc_per_node=2 train.py \
  --config configs/train_main.yml --data-root data/train \
  --train-list data/train_list.txt --quadrants data/quadrants.json \
  --context-cache /path/to/train_contexts.pt --output-dir runs/main
```

For continuation, use `configs/train_continuation.yml` or
`configs/train_low_lr.yml` and add `--checkpoint /path/to/smoe.pth`.
This folder-based launcher does not support exact replay of historical
BasicSR/LMDB training jobs.

## Inference

```bash
python infer.py --input-dir data/test/input --output-dir predictions \
  --checkpoint /path/to/smoe.pth --context-cache /path/to/test_contexts.pt \
  --test-list configs/test300.txt
```

## Evaluation

The target folder must contain exactly the same image stems as the predictions.

```bash
python evaluate.py --pred-dir predictions --gt-dir data/test/target300 \
  --out predictions/metrics.json --lpips
```

## Tests

```bash
python -m unittest discover -s tests -v
```
