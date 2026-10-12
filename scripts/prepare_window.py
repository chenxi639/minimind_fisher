import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,hashlib,shutil
from datetime import datetime
import torch
from model.model_minimind import MiniMindConfig
from model.model_fisher_window import WindowFisherMiniMind

def main():
    source=ROOT/'runs/pair_ready_20260929_181458'
    target=ROOT/'runs'/('w32_ready_'+datetime.now().strftime('%Y%m%d_%H%M%S'));target.mkdir()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    cfg=json.loads((source/'config.json').read_text())
    for n,d in cfg['hashes'].items():
        assert sha(source/n)==d,n
        shutil.copy2(source/n,target/n)
    shutil.copy2(source/'manifest.json',target/'manifest.json')
    model=WindowFisherMiniMind(MiniMindConfig(fisher_window=32))
    weights=torch.load(target/'backbone_init.pt',weights_only=True,map_location='cpu')
    weights.update(torch.load(target/'fisher_init.pt',weights_only=True,map_location='cpu'))
    model.load_state_dict(weights,strict=True)
    cfg['window']=32;cfg['source_preparation']=str(source)
    cfg['code_hashes']={n:sha(ROOT/n) for n in ['model/model_minimind.py','model/model_fisher.py','model/model_fisher_window.py','scripts/train_pair_window.py']}
    (target/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (target/'result.json').write_text(json.dumps({'status':'passed','parameters':sum(p.numel() for p in model.parameters()),'same_all_initial_parameters_as_k8':True}),encoding='utf-8')
    print(target)
if __name__=='__main__':main()
