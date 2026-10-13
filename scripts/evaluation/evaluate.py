#!/usr/bin/env python3
"""Generate terminology and win rate predictions for either evaluation task."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

from sgfmill import sgf, sgf_moves

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = "ABCDEFGHJKLMNOPQRST"
INSTRUCTION = (
    "Given the following Go game state and candidate move, "
    "identify the Go term that describes the candidate move "
    "and provide commentary explaining the move."
)


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def adapter_fingerprint(directory):
    directory = Path(directory)
    files = [directory / 'adapter_config.json']
    files.extend(sorted(directory.glob('adapter_model*.safetensors')))
    files.extend(sorted(directory.glob('adapter_model*.bin')))
    if not files[0].is_file() or len(files) == 1:
        raise ValueError(f'No saved LoRA adapter found in {directory}')
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def coordinate(row, col):
    return f"{COLUMNS[col]}{row + 1}"

def load_position(path):
    game = sgf.Sgf_game.from_bytes(path.read_bytes())
    board, moves = sgf_moves.get_setup_and_moves(game)
    if board.side != 19:
        raise ValueError("Expected a 19x19 board")

    history = []
    last_color = None
    for index, (color, move) in enumerate(moves, start=1):
        player = "X" if color == "b" else "O"
        last_color = color
        if move is None:
            history.append(f"{index}.{player}-PASS")
            continue
        row, col = move
        history.append(f"{index}.{player}-{coordinate(row, col)}")
        board.play(row, col, color)

    board_rows = []
    for row in range(18, -1, -1):
        values = []
        for col in range(19):
            stone = board.get(row, col)
            values.append("1" if stone == "b" else "-1" if stone == "w" else "0")
        board_rows.append(" ".join(values))
    root = game.get_root()
    player_to_play = root.get("PL") if root.has_property("PL") else None
    if last_color is not None:
        next_player = "O" if last_color == "b" else "X"
    else:
        next_player = "O" if player_to_play == "w" else "X"
    return {
        "move_history": " ".join(history),
        "board_state": "\n".join(board_rows),
        "history_length": len(moves),
        "next_player": next_player,
    }

def build_prompt(position, candidate_move):
    example_input = (
        INSTRUCTION
        + "\n\nMove history:\n" + position["move_history"]
        + "\n\nBoard state:\n" + position["board_state"]
        + "\n\nCandidate move:\n"
        + f"{position['next_player']}-{candidate_move}"
    )
    return "### Instruction:\n" + example_input + "\n\n### Response:\n"

def extract_term(output):
    match = re.search(r"(?im)^[ \t]*Term[ \t]*:[ \t]*\*?([^\n\r]+)", output)
    if not match:
        return None
    term = match.group(1).strip().rstrip(".:;, ")
    return term or None

def extract_comment(output):
    match = re.search(r"(?im)^[ \t]*Comment[ \t]*:[ \t]*", output)
    if not match:
        return None
    body = re.split(
        r"(?im)^\s*(?:Win\s*rate|Term)\s*:",
        output[match.end():], maxsplit=1,
    )[0].strip()
    return body or None

def extract_winrate(output):
    match = re.search(r"(?im)^[ \t]*Win[ \t]*rate[ \t]*:[ \t]*([0-9]*\.?[0-9]+)[ \t]*(%)?", output)
    if not match:
        return None
    value = float(match.group(1))
    if match.group(2):
        value /= 100
    return value if 0 <= value <= 1 else None

def load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=quant_config, device_map="auto",
        trust_remote_code=True, torch_dtype=compute_dtype,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return tokenizer, model

def generate(tokenizer, model, prompt):
    import torch

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            max_time=MAX_GENERATION_SECONDS,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)
    return generated.strip(), input_length


def save_results(path, results):
    rows = [row for example in results['examples']
            for predictions in example['predictions'].values() for row in predictions]
    results['metrics'] = {
        'examples': len(results['examples']),
        'predictions': len(rows),
        'terms_generated': sum(row['generated_term'] is not None for row in rows),
        'winrates_generated': sum(row['generated_winrate'] is not None for row in rows),
        'generation_errors': sum(bool(row['error']) for row in rows),
        'term_counts': dict(Counter(row['generated_term'] for row in rows
                                   if row['generated_term'] is not None)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    global BASE_MODEL, ADAPTER_PATH, MAX_INPUT_LENGTH, MAX_NEW_TOKENS, MAX_GENERATION_SECONDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=['atari_sente', 'non_sente_atari'])
    parser.add_argument('--config', type=Path, default=ROOT / 'config/evaluation.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    task = config['tasks'][args.task]
    inputs = project_path(task['input_dir'])
    output = project_path(task['output_file'])
    BASE_MODEL = config['base_model']
    ADAPTER_PATH = str(project_path(config['adapter_path']))
    MAX_INPUT_LENGTH = config['max_input_length']
    MAX_NEW_TOKENS = config['max_new_tokens']
    MAX_GENERATION_SECONDS = config['max_generation_seconds']
    cache = project_path(config['cache_dir'])
    for key, directory in {
        'HF_HOME': cache / 'huggingface',
        'HF_HUB_CACHE': cache / 'huggingface/hub',
        'HF_MODULES_CACHE': cache / 'huggingface/modules',
        'TORCH_HOME': cache / 'torch',
        'TRITON_CACHE_DIR': cache / 'triton',
        'TORCHINDUCTOR_CACHE_DIR': cache / 'torchinductor',
    }.items():
        os.environ[key] = str(directory)

    files = sorted(inputs.glob('*/sample.json'))
    if not files:
        parser.error(f'No samples found in {inputs}')
    if not Path(ADAPTER_PATH).is_dir():
        parser.error(f'Missing adapter: {ADAPTER_PATH}; run scripts/fine_tuning/training.py first')
    try:
        adapter_hash = adapter_fingerprint(ADAPTER_PATH)
    except ValueError as exc:
        parser.error(str(exc))
    sides = ['X', 'Y'] if args.task == 'atari_sente' else ['X']
    candidates = ['A', 'B', 'C'] if args.task == 'atari_sente' else ['A', 'B']
    fingerprint = hashlib.sha256()
    for path in files:
        fingerprint.update(path.parent.name.encode())
        for name in ['sample.json', *[f'{side}.sgf' for side in sides]]:
            fingerprint.update((path.parent / name).read_bytes())
    settings = {
        'base_model': BASE_MODEL, 'adapter_path': ADAPTER_PATH, 'adapter_sha256': adapter_hash,
        'input_directory': str(inputs), 'dataset_sha256': fingerprint.hexdigest(),
        'task': args.task, 'instruction': INSTRUCTION, 'decoding': 'greedy',
        'max_input_length': MAX_INPUT_LENGTH, 'max_new_tokens': MAX_NEW_TOKENS,
        'max_generation_seconds': MAX_GENERATION_SECONDS,
    }
    if output.exists():
        results = json.loads(output.read_text())
        if any(results.get(key) != value for key, value in settings.items()):
            parser.error('Existing output uses different data or settings; choose a new output_file')
    else:
        results = {**settings, 'examples': [], 'errors': [], 'metrics': {}}
    existing = {example['folder']: example for example in results['examples']}
    tokenizer = model = None
    try:
        for index, path in enumerate(files, 1):
            ident = path.parent.name
            try:
                sample = json.loads(path.read_text())
                positions = {side: load_position(path.parent / f'{side}.sgf') for side in sides}
                for side, position in positions.items():
                    if position['history_length'] != sample['move_number']:
                        raise ValueError(f'{side}: move history length differs from sample.json')
                    for candidate in candidates:
                        sample['moves'][candidate]
                        sample['results'][side]['winrate'][candidate]
            except (ValueError, KeyError, IndexError, OSError) as exc:
                results['errors'] = [e for e in results['errors'] if e['folder'] != ident]
                results['errors'].append({'folder': ident, 'error': str(exc)})
                print(f'Skipping {ident}: {exc}', flush=True)
                save_results(output, results)
                continue
            if ident not in existing:
                example = {
                    'folder': ident, 'source_sha256': sample.get('source_sha256'),
                    'move_number': sample['move_number'],
                    'candidate_moves': sample['moves'], 'X_A_targets': sample.get('X_A_targets'),
                    'reference_winrates': (
                        {s: sample['results'][s]['winrate'] for s in sides}
                        if len(sides) == 2 else sample['results']['X']['winrate']),
                    'predictions': {side: [] for side in sides},
                }
                results['examples'].append(example)
                existing[ident] = example
            example = existing[ident]
            for side in sides:
                rows = example['predictions'][side]
                for candidate in candidates:
                    if any(row['candidate'] == candidate and not row['error'] for row in rows):
                        continue
                    if model is None:
                        tokenizer, model = load_model()
                    move = sample['moves'][candidate]
                    prompt = build_prompt(positions[side], move)
                    print(f'{index}/{len(files)} | {ident} | {side}/{candidate}', flush=True)
                    try:
                        text, tokens = generate(tokenizer, model, prompt)
                        error = None
                    except (RuntimeError, ValueError) as exc:
                        text, tokens, error = '', None, str(exc)
                    rows[:] = [row for row in rows if row['candidate'] != candidate]
                    rows.append({
                        'candidate': candidate, 'move': move,
                        'move_count': positions[side]['history_length'], 'input_tokens': tokens,
                        'generated_term': extract_term(text), 'generated_winrate': extract_winrate(text),
                        'generated_comment': extract_comment(text), 'generated_output': text,
                        'reference_winrate': sample['results'][side]['winrate'][candidate],
                        'prompt': prompt, 'error': error,
                    })
                    save_results(output, results)
    except KeyboardInterrupt:
        print('Stopped. Run the same command to continue.', flush=True)
    finally:
        save_results(output, results)
    print(f'Saved predictions to: {output}')


if __name__ == '__main__':
    main()
