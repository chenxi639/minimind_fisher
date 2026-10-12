"""Separate timing copies; primary checkpoints never modified."""
import sys,json,time,hashlib,statistics,subprocess
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind
from model.model_moment import MomentMiniMind
from model.model_matched_mlp import MatchedMiniMind
from model.model_peft25 import install,merged_lora
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    formal=Path(sys.argv[1]).resolve();comp=json.loads((formal/'comparison.json').read_text());assert comp['status']=='passed'
    manifest=json.loads((formal/'manifest.json').read_text());files=dict(manifest['file_hashes'])
    paths={'k1':ROOT/'runs/mechanism_pair_20261003_113236/k1/checkpoint.pt','moment':ROOT/'runs/mechanism_pair_20261003_113236/moment/checkpoint.pt','adapter96':ROOT/'runs/controls24_20261003_152008/adapter/checkpoint.pt','houlsby':formal/'houlsby/checkpoint.pt','lora':formal/'lora/checkpoint.pt'}
    previous=json.loads((ROOT/'runs/mechanism_pair_20261003_113236/comparison.json').read_text());control=json.loads((ROOT/'runs/controls24_20261003_152008/comparison.json').read_text())
    for name,p in paths.items():files[str(p)]=(previous[name] if name in previous else control['adapter'] if name=='adapter96' else comp[name])['checkpoint_sha256']
    files[str(Path(__file__))]=sha(__file__)
    def verify():
        for p,d in files.items():assert sha(p)==d,p
    verify();run=ROOT/'runs'/('cost25_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir();print(run,flush=True)
    def gpu():return subprocess.run(['nvidia-smi','--query-gpu=name,temperature.gpu,power.draw,utilization.gpu,memory.used','--format=csv'],capture_output=True,text=True).stdout
    r={'status':'running','files':files,'gpu_start':gpu(),'precision':'bf16','warmup':5,'repeats':20,'models':{}}
    def save():(run/'result.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    state=torch.load(manifest['backbone_path'],map_location='cpu',weights_only=True)
    tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True);ds=PretrainDataset(str(Path(manifest['source_preparation'])/'train.jsonl'),tok,max_length=340)
    order=json.loads((Path(manifest['source_preparation'])/'order.json').read_text());rows=[ds[i] for i in order[:16]]
    batches=[tuple(torch.stack([v[k] for v in rows[i:i+2]]).cuda() for k in [0,1]) for i in range(0,16,2)];x,y=batches[0]
    def forward(m,x,y=None,**kw):
        with torch.autocast('cuda',dtype=torch.bfloat16):return m(x,labels=y,**kw)
    def timing(fn):
        for _ in range(5):fn()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();values=[]
        for _ in range(20):
            torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();values.append((time.perf_counter()-t)*1000)
        q=statistics.quantiles(values,n=4,method='inclusive')
        return {'median_ms':statistics.median(values),'iqr_ms':q[2]-q[0],'samples_ms':values,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
    for name in ['pure','k1','moment','adapter96','houlsby','lora','lora_merged']:
        cfg=MiniMindConfig(fisher_k=1,fisher_dim=64,fisher_index=3,adapter_width=96)
        cls=FisherMiniMind if name=='k1' else MomentMiniMind if name=='moment' else MatchedMiniMind if name=='adapter96' else MiniMindForCausalLM
        m=cls(cfg);weights=dict(state)
        if name in ['k1','moment','adapter96']:
            branch='adapter' if name=='adapter96' else 'fisher';weights.update({f'model.layers.3.{branch}.'+k:v for k,v in torch.load(paths[name],weights_only=True,map_location='cpu')['branch'].items()});m.load_state_dict(weights,strict=True);m.requires_grad_(False)
            for p in getattr(m.model.layers[3],branch).parameters():p.requires_grad_(True)
        else:
            m.load_state_dict(weights,strict=True);m.requires_grad_(False)
            if name!='pure':
                kind='lora' if name=='lora_merged' else name;install(m,kind);m.load_state_dict({**state,**torch.load(paths[kind],weights_only=True,map_location='cpu')['adaptation']},strict=True)
            if name=='lora_merged':
                m.cuda().eval();merged=merged_lora(m).cuda()
                with torch.no_grad():
                    original=m(x).logits;combined=merged(x).logits
                    r['merge_fp32_max_abs_error']=float((original-combined).abs().max())
                    torch.testing.assert_close(original,combined,atol=2e-4,rtol=1e-4)
                del original,combined,m;m=merged;m.requires_grad_(False)
        m.cuda().eval();entry={'total_parameters':sum(p.numel() for p in m.parameters()),'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'gpu_before':gpu()}
        with torch.no_grad():
            entry['prefill']=timing(lambda:forward(m,x,use_cache=True))
            cache=forward(m,x,use_cache=True).past_key_values
            entry['decode_context340']=timing(lambda:forward(m,x[:,:1],past_key_values=cache,use_cache=True))
        del cache
        if name not in ['pure','lora_merged']:
            params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
            def step():
                m.train();opt.zero_grad(set_to_none=True)
                for tx,ty in batches:(forward(m,tx,ty).loss/8).backward()
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
            entry['train_update_batch2_accum8']=timing(step);entry['timing_copy_updates']=25
            del params,opt
        r['models'][name]=entry;save();print(name,entry['prefill']['median_ms'],entry['decode_context340']['median_ms'],flush=True)
        del m;torch.cuda.empty_cache()
    verify();r.update(status='passed',primary_checkpoints_unchanged=True,gpu_end=gpu());save()
if __name__=='__main__':main()
