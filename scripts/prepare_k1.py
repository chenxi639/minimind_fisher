"""Derive K1 preparation from immutable K8 data and initialization."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,hashlib,shutil
from datetime import datetime
import torch
from model.model_minimind import MiniMindConfig
from model.model_fisher import FisherMiniMind

def main():
    source=ROOT/'runs/pair_ready_20260929_181458'
    target=ROOT/'runs'/('k1_ready_'+datetime.now().strftime('%Y%m%d_%H%M%S'));target.mkdir()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    cfg=json.loads((source/'config.json').read_text())
    for n,d in cfg['hashes'].items():assert sha(source/n)==d,n
    for n in ['train.jsonl','dev.jsonl','test.jsonl','order.json','backbone_init.pt','manifest.json']:
        shutil.copy2(source/n,target/n)
    state=torch.load(source/'fisher_init.pt',map_location='cpu',weights_only=True)
    prefix='model.layers.3.fisher.'
    state[prefix+'w']=state[prefix+'w'][:1].clone()
    state[prefix+'b']=state[prefix+'b'][:1].clone()
    assert not state[prefix+'output.weight'].count_nonzero()
    state[prefix+'output.weight']=torch.zeros(768,128)
    model=FisherMiniMind(MiniMindConfig(fisher_k=1))
    backbone=torch.load(target/'backbone_init.pt',map_location='cpu',weights_only=True)
    model.load_state_dict({**backbone,**state},strict=True)
    torch.save(state,target/'fisher_init.pt')
    cfg['k']=1
    cfg['hashes']={n:sha(target/n) for n in cfg['hashes']}
    cfg['code_hashes']={n:sha(ROOT/n) for n in ['model/model_minimind.py','model/model_fisher.py','scripts/train_pair_configured.py']}
    cfg['source_preparation']=str(source)
    cfg['initialization']='same backbone, norm, input projection; component 0 w/b from K8; zero output'
    (target/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    result={'status':'passed','parameters':sum(p.numel() for p in model.parameters()),'source':str(source)}
    (target/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(target);print(result)
if __name__=='__main__':main()
