"""Prepare a small real-data baseline and check forward only; no training."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import datasets  # load Arrow before torch on Windows, as upstream does
import hashlib
import json
import random
import subprocess
from datetime import datetime
import requests
import torch
import transformers
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    run = ROOT / 'runs' / ('prepare_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    run.mkdir(parents=True, exist_ok=False)
    def save(name, obj):
        (run / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    result = dict(status='running', trained=False, optimizer_steps=0)
    save('packages.json', json.loads(subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--format=json'], text=True)))
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'model'), local_files_only=True)
        # Pin the dataset revision before reading its small prefix.
        api = 'https://huggingface.co/api/datasets/jingyaogong/minimind_dataset'
        response = requests.get(api, timeout=(15, 45)); response.raise_for_status()
        revision = response.json()['sha']
        base = f'https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/{revision}'
        card = requests.get(base + '/README.md', timeout=(15, 45)); card.raise_for_status()
        (run / 'dataset_card.md').write_text(card.text, encoding='utf-8')
        url = base + '/pretrain_t2t_mini.jsonl'
        rows, seen, seen_tokens = [], set(), set()
        with requests.get(url, stream=True, timeout=(15, 45)) as stream:
            stream.raise_for_status()
            for line_no, line in enumerate(stream.iter_lines(), 1):
                if not line:
                    continue
                text = json.loads(line)['text']
                if not isinstance(text, str) or not text.strip():
                    continue
                digest = sha(text.strip().encode('utf-8'))
                ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=338).input_ids
                token_digest = sha(json.dumps(ids).encode())
                if not ids or digest in seen or token_digest in seen_tokens:
                    continue
                seen.add(digest); seen_tokens.add(token_digest)
                rows.append(dict(text=text, source_line=line_no, sha256=digest, token_sha256=token_digest))
                if len(rows) == 1152:
                    break
                if line_no >= 20000:
                    raise RuntimeError('Insufficient unique examples in bounded prefix')
        if len(rows) != 1152:
            raise RuntimeError('Source ended before the sample budget was met')
        random.Random(42).shuffle(rows)
        manifest = dict(source=url, dataset_revision=revision, seed=42,
                        selection='first 1152 unique nonempty texts and truncated token sequences, then seeded shuffle',
                        limitation='prefix convenience sample; exact dedup only, not a representative benchmark', splits={})
        for split, selected in [('train', rows[:1024]), ('validation', rows[1024:])]:
            path = run / (split + '.jsonl')
            path.write_text(''.join(json.dumps({'text': r['text']}, ensure_ascii=False) + '\n' for r in selected), encoding='utf-8')
            manifest['splits'][split] = dict(count=len(selected), file_sha256=sha(path.read_bytes()),
                examples=[{k:v for k,v in r.items() if k != 'text'} for r in selected])
        save('data_manifest.json', manifest)
        config = dict(hidden_size=768, num_hidden_layers=8, use_moe=False, max_seq_len=340,
                      batch_size=2, accumulation_steps=8, epochs=1, learning_rate=5e-4,
                      seed=42, dtype='bfloat16', num_workers=0, from_weight='none',
                      planned_microbatches=512, planned_optimizer_steps=64)
        save('planned_config.json', config)
        train = PretrainDataset(str(run / 'train.jsonl'), tokenizer, max_length=340)
        valid = PretrainDataset(str(run / 'validation.jsonl'), tokenizer, max_length=340)
        counts = {name: sum(int((ds[i][1][1:] != -100).sum()) for i in range(len(ds)))
                  for name, ds in [('train', train), ('validation', valid)]}
        torch.manual_seed(42)
        model = MiniMindForCausalLM(MiniMindConfig()).cuda().eval()
        assert not any('fisher' in name.lower() for name, _ in model.named_parameters())
        inputs = torch.stack([train[i][0] for i in range(2)]).cuda()
        labels = torch.stack([train[i][1] for i in range(2)]).cuda()
        assert torch.cuda.is_bf16_supported()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(inputs, labels=labels)
        assert torch.isfinite(output.logits).all() and torch.isfinite(output.loss)
        result.update(status='passed', python=sys.version, torch=torch.__version__,
                      transformers=transformers.__version__, datasets=datasets.__version__,
                      gpu=torch.cuda.get_device_name(0), bf16=True,
                      vocab_size=len(tokenizer), token_ids=dict(bos=tokenizer.bos_token_id,eos=tokenizer.eos_token_id,pad=tokenizer.pad_token_id),
                      parameters=sum(p.numel() for p in model.parameters()),
                      logits_shape=list(output.logits.shape), forward_loss_finite=True,
                      valid_prediction_tokens=counts,
                      forward_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      memory_limitation='forward only; training peak not measured',
                      source_hashes={p:sha((ROOT/p).read_bytes()) for p in ['model/model_minimind.py','model/tokenizer.json','trainer/train_pretrain.py']})
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        save('result.json', result)
        print('RUN:', run, flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
