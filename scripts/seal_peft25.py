"""Normalize inherited preparation metadata without repeating smoke training."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'runs/peft25_ready_20261003_180008'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((run/'config.json').read_text())
assert json.loads((run/'preparation.json').read_text())['status']=='passed'
for p,d in cfg['file_hashes'].items():assert sha(p)==d,p
for key in ['index','k','dim','adapter_width','lora_rank','lora_alpha','code_hashes']:cfg.pop(key,None)
cfg['architectures']={'houlsby':{'layers':list(range(8)),'sites':['attention_output_before_residual','ffn_output_before_residual'],'width':6,'activation':'relu','bias':True,'original_norm_trainable':False},'lora':{'layers':list(range(8)),'targets':['q_proj','v_proj'],'rank':7,'alpha':7,'dropout':0,'bias_trainable':False,'auto_merge':False}}
cfg['primary_evaluation']='fp32_tf32_off_unmerged';cfg['auxiliary_evaluation']='bf16_unmerged'
for p in [Path(__file__),run/'config.json',run/'preparation.json']:cfg['file_hashes'][str(p)]=sha(p)
target=run/'locked_config.json';assert not target.exists();target.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
print(target)
