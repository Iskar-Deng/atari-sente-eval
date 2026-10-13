#!/usr/bin/env python3
"""Add candidate-player win rates using the shared KataGo installation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/construction'))
from generate_atari_sente import Engine, config_path, engine_command

SYMBOL_TO_COLOR = {'X': 'B', 'O': 'W'}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def extract_move_history(text):
    match = re.search(r'Move history:\s*(.*?)\s*Board state:', text, re.DOTALL)
    if not match:
        raise ValueError('Missing move history')
    return [(symbol.upper(), coord.upper()) for symbol, coord in re.findall(
        r'\d+\.\s*([XO])-([A-HJ-T]\d+|pass)', match[1], re.IGNORECASE)]


def extract_candidate_move(text):
    match = re.search(r'Candidate move:\s*([XO])-([A-HJ-T]\d+|pass)', text, re.IGNORECASE)
    if not match:
        raise ValueError('Missing candidate move')
    return match[1].upper(), match[2].upper()


def build_query(history, candidate):
    moves = [[SYMBOL_TO_COLOR[symbol], coord] for symbol, coord in history + [candidate]]
    next_player = 'W' if moves[-1][0] == 'B' else 'B'
    return dict(moves=moves, initialStones=[], initialPlayer='B',
                rules='chinese', komi=7.5, boardXSize=19, boardYSize=19,
                analyzeTurns=[len(moves)], _player=next_player)


def candidate_winrate(response, symbol):
    black = float(response['rootInfo']['winrate'])
    if not math.isfinite(black) or not 0 <= black <= 1:
        raise ValueError('Invalid KataGo win rate')
    return black if symbol == 'X' else 1 - black


def add_winrate(output, value):
    lines = [line for line in output.splitlines()
             if not re.match(r'^Win\s*rate\s*:', line, re.IGNORECASE)]
    for index, line in enumerate(lines):
        if line.startswith('Term:'):
            lines.insert(index + 1, f'Winrate: {value * 100:.2f}%')
            return '\n'.join(lines)
    raise ValueError('Missing Term field')


class Labeler:
    def __init__(self, out, visits):
        self.log = out / 'engine.log'
        self.visits = visits
        self.engine = None

    def query(self, history, candidate):
        for attempt in range(2):
            try:
                if self.engine is None:
                    self.engine = Engine(self.log, perspective='BLACK')
                response = self.engine.query(build_query(history, candidate), self.visits)
                return candidate_winrate(response, candidate[0])
            except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
                self.close()
                if attempt:
                    raise
                print(f'KataGo: {exc}; retrying once', flush=True)

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine = None


def process_file(source, out, labeler, max_history):
    raw = source.read_bytes()
    examples = json.loads(raw)
    output = out / source.name
    checkpoint = out / f'{source.stem}.state.json'
    settings = dict(input_sha256=hashlib.sha256(raw).hexdigest(),
                    visits=labeler.visits, max_history=max_history,
                    engine_command=engine_command(),
                    config_sha256=hashlib.sha256(config_path().read_bytes()).hexdigest(),
                    perspective='candidate player')
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state['settings'] != settings:
            raise ValueError(f'{checkpoint}: inputs or settings changed; use a new --out directory')
    else:
        state = dict(settings=settings, next_index=0, skipped=0, failures=[], examples=[])

    def save():
        # The checkpoint includes the output so an interrupted two-file save is recoverable.
        atomic_json(checkpoint, state)
        atomic_json(output, state['examples'])

    def progress():
        print(f'{source.stem}: {state["next_index"]}/{len(examples)} | '
              f'Saved {len(state["examples"])} | Skipped {state["skipped"]} | '
              f'Failed {len(state["failures"])}', flush=True)

    progress()
    last_report = time.monotonic()
    try:
        for index in range(state['next_index'], len(examples)):
            example = examples[index]
            try:
                history = extract_move_history(example['input'])
                # Retain the original training pipeline's history-length cutoff.
                if len(history) > max_history:
                    state['skipped'] += 1
                else:
                    candidate = extract_candidate_move(example['input'])
                    winrate = labeler.query(history, candidate)
                    state['examples'].append(dict(example, output=add_winrate(example['output'], winrate)))
            except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
                state['failures'].append(dict(index=index, error=str(exc)))
                print(f'{source.stem}: skipping sample {index}: {exc}', flush=True)
            state['next_index'] = index + 1
            if (index + 1) % 50 == 0 or time.monotonic() - last_report >= 30:
                save()
                progress()
                last_report = time.monotonic()
    finally:
        save()
    progress()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'local/finetuning/intermediate')
    parser.add_argument('--out', type=Path, default=ROOT / 'local/finetuning/data')
    parser.add_argument('--visits', type=int, default=500)
    parser.add_argument('--max-history', type=int, default=150)
    args = parser.parse_args()
    if args.visits < 1 or args.max_history < 0:
        parser.error('Visits must be positive and history length nonnegative')
    sources = [args.input_dir / f'{split}.json' for split in ['train', 'val', 'test']]
    for source in sources:
        if not source.is_file():
            parser.error(f'Missing input: {source}')
    if args.input_dir.resolve() == args.out.resolve():
        parser.error('Input and output directories must differ')
    args.out.mkdir(parents=True, exist_ok=True)
    labeler = Labeler(args.out, args.visits)

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        for source in sources:
            process_file(source, args.out, labeler, args.max_history)
    except KeyboardInterrupt:
        print('\nStopped. Run the same command to continue.', flush=True)
    finally:
        labeler.close()


if __name__ == '__main__':
    main()
