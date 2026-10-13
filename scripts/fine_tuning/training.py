#!/usr/bin/env python3
"""Fine-tune LoGos with terminology, win rate, and commentary supervision."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/fine_tuning.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    train_file = project_path(config['train_file'])
    validation_file = project_path(config['validation_file'])
    for path in [train_file, validation_file]:
        if not path.is_file():
            parser.error(f'Missing training data: {path}; run scripts/annotation/run.sh first')

    cache = project_path(config['cache_dir'])
    # Set cache locations before importing Hugging Face libraries.
    for name, directory in {
        'HF_HOME': cache / 'huggingface',
        'HF_HUB_CACHE': cache / 'huggingface/hub',
        'HF_DATASETS_CACHE': cache / 'huggingface/datasets',
        'HF_MODULES_CACHE': cache / 'huggingface/modules',
        'TORCH_HOME': cache / 'torch',
        'TRITON_CACHE_DIR': cache / 'triton',
        'TORCHINDUCTOR_CACHE_DIR': cache / 'torchinductor',
    }.items():
        os.environ[name] = str(directory)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        parser.error('Fine-tuning requires a CUDA-enabled PyTorch installation and an NVIDIA GPU')

    dataset = load_dataset('json', data_files={
        'train': str(train_file), 'validation': str(validation_file),
    })
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    def format_example(example):
        return (
            '### Instruction:\n'
            f"{example['input']}\n\n"
            '### Response:\n'
            f"{example['output']}"
            f'{tokenizer.eos_token}'
        )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        **config['quantization'], bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config['base_model'], quantization_config=quantization,
        device_map='auto', trust_remote_code=True, torch_dtype=dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    output = project_path(config['output_dir'])
    training_args = SFTConfig(
        **config['training'], output_dir=str(output),
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
    )
    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=dataset['train'], eval_dataset=dataset['validation'],
        formatting_func=format_example, processing_class=tokenizer,
        peft_config=LoraConfig(**config['lora']),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / 'fine_tuning.json').write_text(json.dumps(config, indent=2) + '\n')
    trainer.train()
    final = output / 'final'
    trainer.model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f'Saved adapter to: {final}')


if __name__ == '__main__':
    main()
