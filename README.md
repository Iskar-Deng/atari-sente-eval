# Atari and Sente Evaluation

This repository provides controlled evaluation tasks for studying atari recognition and its relationship to move evaluation in language-based Go agents. It includes the evaluation dataset and code for test position construction, data annotation, model fine-tuning, and evaluation.

The evaluation set contains 1,000 X/Y pairs in `data/atari_sente/` and 500 positions in `data/non_sente_atari/`, with full move histories and candidate points labeled in the SGFs.

## Setup

The experiments require Ubuntu/Debian Linux x86_64, Python 3.10–3.12 with `venv` support, and an NVIDIA GPU with a driver supporting CUDA 12.1. All external inputs, intermediate files, and generated outputs are stored under `local/`.

Create a virtual environment and install Python dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Install KataGo's system dependencies and download the engine and its neural network:

```bash
bash setup_katago.sh
```

## 1. Construction

We construct evaluation samples by extracting and modifying positions from human game records. Download and extract SGF files from the [Database of Go Games](https://homepages.cwi.nl/~aeb/go/games/games/index.html) into `local/games/`, or place your own SGF files there. Run the following commands to generate evaluation samples:

```bash
.venv/bin/python scripts/construction/generate_atari_sente.py --count 1000
.venv/bin/python scripts/construction/generate_non_sente_atari.py --count 500
```

## 2. Annotation

We construct fine-tuning data from the [Tomlin et al. (2022) corpus](https://github.com/AndreHe02/go). Download the corpus and place its SGF files in `local/annotations/`. Run the following command to extract commented moves and add terminology and win rate labels:

```bash
bash scripts/annotation/run.sh
```

## 3. Fine-tuning

We fine-tune [LoGos-7B](https://huggingface.co/YichuanMa/LoGos-7B) using QLoRA with terminology, win rate, and commentary supervision. Set fine-tuning parameters in `config/fine_tuning.json`, then run:

```bash
.venv/bin/python scripts/fine_tuning/training.py
```

## 4. Evaluation

We evaluate the fine-tuned model on both tests, generating terminology and win rate predictions for each candidate move. To use the provided evaluation set directly, skip Construction and set `adapter_path` in `config/evaluation.json` to your fine-tuned adapter; the default input paths already point to `data/`. Run:

```bash
.venv/bin/python scripts/evaluation/evaluate.py --task atari_sente
.venv/bin/python scripts/evaluation/evaluate.py --task non_sente_atari
```
