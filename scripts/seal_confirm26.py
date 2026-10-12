"""Freeze candidate implementation and source identities; no training."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=['model/model_minimind.py','model/model_fisher.py','model/model_moment.py','model/model_moment_controls.py','model/model_peft25.py','model/tokenizer.json','model/tokenizer_config.json','dataset/lm_dataset.py','report/26A_protocol.md','scripts/prepare_confirm26.py','scripts/seal_confirm26.py','dataset/external_csl_26a/manifest.json','dataset/external_csl_26a/eval.jsonl','runs/frozen_ready_20261002_124018/kmeans_fit.pt','runs/pair_ready_20260929_181458/train.jsonl','runs/pair_ready_20260929_181458/dev.jsonl','checkpoints/official_edba70e/download.pth']
hashes={str(ROOT/p):sha(ROOT/p) for p in files}
assert hashes[str(ROOT/'checkpoints/official_edba70e/download.pth')]=='0191958e7a96ef7cd1e443d379dc2ebea90cfea72cc4450fbcb4a7b87ff945f6'
manifest=json.loads((ROOT/'dataset/external_csl_26a/manifest.json').read_text())
assert sha(ROOT/'dataset/external_csl_26a/eval.jsonl')==manifest['data_sha256']
target=ROOT/'runs/confirm26_registry.json';assert not target.exists()
target.write_text(json.dumps({'status':'locked_before_new_seed_training_and_evaluation','candidates':['pure','k1','current','lora'],'seeds':[1729,2718,31415],'file_hashes':hashes,'confirmation_source':'CSL scientific abstracts; not general-domain Chinese','old_wikipedia_not_new_confirmation':True,'model_evaluations':0},indent=2),encoding='utf-8')
print(target)
