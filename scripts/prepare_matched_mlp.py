import sys,json,shutil,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from model.model_minimind import MiniMindConfig
from model.model_matched_mlp import MatchedMiniMind,MatchedResidual

def main():
    source=ROOT/'runs/pair_ready_20260929_181458';target=ROOT/'runs'/('mlp_ready_'+datetime.now().strftime('%Y%m%d_%H%M%S'));target.mkdir()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    cfg=json.loads((source/'config.json').read_text())
    cfg['hashes'].pop('fisher_init.pt')
    for name,d in cfg['hashes'].items():
        assert sha(source/name)==d,name
        shutil.copy2(source/name,target/name)
    shutil.copy2(source/'manifest.json',target/'manifest.json')
    torch.manual_seed(43);branch=MatchedResidual(768,544)
    init={'model.layers.3.adapter.'+k:v for k,v in branch.state_dict().items()}
    assert sum(p.numel() for p in branch.parameters())==836352
    torch.save(init,target/'adapter_init.pt')
    cfg['hashes']['adapter_init.pt']=sha(target/'adapter_init.pt');cfg['adapter_width']=544
    cfg['code_hashes']={n:sha(ROOT/n) for n in ['model/model_minimind.py','model/model_matched_mlp.py','scripts/train_matched_mlp.py','scripts/prepare_matched_mlp.py']}
    model=MatchedMiniMind(MiniMindConfig())
    weights=torch.load(target/'backbone_init.pt',weights_only=True);weights.update(init);model.load_state_dict(weights,strict=True)
    (target/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (target/'preparation.json').write_text(json.dumps({'status':'passed','parameters':sum(p.numel() for p in model.parameters()),'branch_parameters':836352,'fisher_branch_parameters':837376,'relative_mismatch':1024/837376},indent=2),encoding='utf-8')
    print(target)
if __name__=='__main__':main()
