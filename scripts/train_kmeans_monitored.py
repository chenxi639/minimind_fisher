"""Run the unchanged paired trainer with read-only evaluation instrumentation."""
import sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datetime import datetime
import torch
import model.model_fisher as fisher
from scripts.assignment_monitor import AssignmentMonitor

folder=Path(sys.argv[sys.argv.index('--prepared')+1]).resolve()
health=folder/('assignment_'+('smoke_' if '--smoke' in sys.argv else 'train_')+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
assert not health.exists()
Original=fisher.FisherMiniMind
class ObservedFisher(Original):
    def __init__(self,config):
        super().__init__(config);self.observer=AssignmentMonitor(self,health)
    def train(self,mode=True):
        if hasattr(self,'observer'):
            if not mode:self.observer.start()
            elif self.observer.enabled:self.observer.finish()
        return super().train(mode)
    def forward(self,*args,**kwargs):
        if self.observer.enabled:
            y=kwargs['labels'];mask=torch.zeros_like(y,dtype=torch.bool);mask[:,:-1]=y[:,1:]!=-100
            self.observer.mask=mask
        return super().forward(*args,**kwargs)

fisher.FisherMiniMind=ObservedFisher
runpy.run_path(str(ROOT/'scripts/train_pair_configured.py'),run_name='__main__')
print('HEALTH',health,flush=True)
