"""Run unchanged upstream training, then verify its checkpoint and evaluate."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import datasets  # Windows Arrow before torch
import hashlib
import json
import math
import runpy
import time
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log
    def write(self, text):
        self.original.write(text); self.log.write(text); self.log.flush()
    def flush(self):
        self.original.flush(); self.log.flush()


def main():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run = ROOT / 'runs' / ('baseline_' + stamp)
    run.mkdir(exist_ok=False)
    source = ROOT / 'runs' / 'prepare_20260929_132426'
    def save(name, value):
        (run / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((source / 'data_manifest.json').read_text(encoding='utf-8'))
    for split in ['train', 'validation']:
        assert sha(source / (split + '.jsonl')) == manifest['splits'][split]['file_sha256']
    prefix = 'pretrain_' + stamp
    args = ['train_pretrain.py', '--data_path', str(source / 'train.jsonl'),
            '--save_dir', str(run / 'out'), '--save_weight', prefix,
            '--epochs','1','--batch_size','2','--accumulation_steps','8',
            '--learning_rate','0.0005','--hidden_size','768','--num_hidden_layers','8',
            '--max_seq_len','340','--dtype','bfloat16','--num_workers','0',
            '--seed','42','--from_weight','none','--from_resume','0',
            '--use_moe','0','--use_compile','0','--log_interval','32','--save_interval','512']
    save('config.json', dict(arguments=args, data_source=str(source),
         model_sha256=sha(ROOT/'model/model_minimind.py'),
         trainer_sha256=sha(ROOT/'trainer/train_pretrain.py'),
         torch=torch.__version__, seed=42, evaluation='token weighted float32 CE under BF16 autocast',
         generation=dict(seed=42,do_sample=False,max_new_tokens=64)))
    result = dict(status='running', run=str(run), trained=False)
    save('result.json', result)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    old_argv, old_cwd = sys.argv, Path.cwd()
    start = time.perf_counter()
    with (run/'train.log').open('w', encoding='utf-8') as log:
        sys.stdout, sys.stderr = Tee(original_stdout,log), Tee(original_stderr,log)
        try:
            print('RUN:', run, flush=True)
            os.chdir(ROOT/'trainer'); sys.argv=args
            torch.cuda.reset_peak_memory_stats()
            state = runpy.run_path(str(ROOT/'trainer/train_pretrain.py'), run_name='__main__')
            torch.cuda.synchronize()
            result.update(trained=True, training_seconds=time.perf_counter()-start,
                          training_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            model = state['model'].eval()
            steps = sorted(set(int(v['step'].item()) for v in state['optimizer'].state.values() if 'step' in v))
            assert steps == [64], steps
            result['optimizer_steps'] = steps[0]
            checkpoint = run/'out'/f'{prefix}_768.pth'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert all(torch.isfinite(v).all() for v in saved.values())
            assert all(torch.equal(v.detach().half().cpu(),saved[k]) for k,v in model.state_dict().items())
            # Upstream exports FP16; compare at the actual serialization precision.
            model.load_state_dict(saved,strict=True)
            loaded = MiniMindForCausalLM(MiniMindConfig()).cuda().eval()
            loaded.load_state_dict(saved,strict=True)
            tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
            ds=PretrainDataset(str(source/'validation.jsonl'),tokenizer,max_length=340)
            probe=ds[0][0][None].cuda()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                a=model(probe).logits; b=loaded(probe).logits
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            result.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                          reload_exact_at_saved_precision=True,
                          resume_checkpoint=str(ROOT/'checkpoints'/f'{prefix}_768_resume.pth'))
            # Verify optimizer checkpoint without performing extra training.
            resume=torch.load(result['resume_checkpoint'],map_location='cpu',weights_only=True)
            assert resume['step']==512 and resume['epoch']==0
            assert all(torch.equal(v,resume['model'][k]) for k,v in saved.items())
            assert {int(v['step'].item()) for v in resume['optimizer']['state'].values()}=={64}
            result['resume_checkpoint_verified']=True
            del resume, saved, model, state
            torch.cuda.empty_cache()
            summed, tokens=0.,0
            with torch.no_grad():
                for i in range(0,len(ds),2):
                    batch=[ds[j] for j in range(i,min(i+2,len(ds)))]
                    x=torch.stack([r[0] for r in batch]).cuda()
                    y=torch.stack([r[1] for r in batch]).cuda()[:,1:].contiguous()
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logits=loaded(x).logits[:,:-1].float().contiguous()
                    loss=F.cross_entropy(logits.view(-1,6400),y.view(-1),ignore_index=-100,reduction='sum')
                    assert torch.isfinite(loss)
                    summed+=loss.item();tokens+=int((y!=-100).sum())
            result.update(validation_tokens=tokens,validation_ce=summed/tokens,validation_ppl=math.exp(summed/tokens))
            samples=[]
            for prompt in ['清晨的阳光','机器学习是','The meaning of learning is']:
                torch.manual_seed(42)
                ids=tokenizer(tokenizer.bos_token+prompt,return_tensors='pt').input_ids.cuda()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    generated=loaded.generate(inputs=ids,max_new_tokens=64,do_sample=False,eos_token_id=tokenizer.eos_token_id)
                continuation=generated[0,ids.shape[1]:].tolist()
                samples.append(dict(prompt=prompt,generated_ids=continuation,text=tokenizer.decode(continuation,skip_special_tokens=True)))
            save('generation.json',samples)
            result.update(status='passed',total_seconds=time.perf_counter()-start)
            print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        except Exception as exc:
            result.update(status='failed',error=repr(exc))
            raise
        finally:
            save('result.json',result)
            sys.stdout,sys.stderr=original_stdout,original_stderr
            sys.argv=old_argv; os.chdir(old_cwd)


if __name__=='__main__':
    main()
