"""Create a pinned, stratified byte-window dataset and shared initialization."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import datasets
import json, hashlib, random
from datetime import datetime
import requests
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    run=ROOT/'runs'/('pair_ready_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir()
    def save(n,v):(run/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PREP',run,flush=True)
    result={'status':'running','formal_training':False};save('result.json',result)
    try:
        url='https://huggingface.co/datasets/jingyaogong/minimind_dataset/resolve/312afb4f76391145c6902f765bb51691c09a12f5/pretrain_t2t_mini.jsonl'
        r=requests.get(url,headers={'Range':'bytes=0-0'},timeout=(15,90));r.raise_for_status()
        if r.status_code!=206:raise RuntimeError('Server must support bounded range reads')
        total=int(r.headers['Content-Range'].split('/')[-1])
        tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
        seen=set();seen_ids=set();rows=[];windows=[]
        # Exclude historical pilot texts from all new splits.
        for name in ['train','validation']:
            for line in (ROOT/'runs/prepare_20260929_132426'/f'{name}.jsonl').read_text(encoding='utf-8').splitlines():
                text=json.loads(line)['text'];seen.add(hashlib.sha256(text.strip().encode()).hexdigest())
                seen_ids.add(tuple(tok(text,add_special_tokens=False,truncation=True,max_length=338).input_ids))
        for i in range(24):
            start=int(i*total/24);end=min(total-1,start+8*1024*1024-1)
            with requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(15,90)) as response:
                response.raise_for_status()
                if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                    raise RuntimeError('Range response mismatch')
                count=0
                for line_no,line in enumerate(response.iter_lines()):
                    if start and line_no==0:continue
                    if not line:continue
                    text=json.loads(line)['text']
                    if not isinstance(text,str) or not text.strip():continue
                    digest=hashlib.sha256(text.strip().encode()).hexdigest()
                    ids=tuple(tok(text,add_special_tokens=False,truncation=True,max_length=338).input_ids)
                    if not ids or digest in seen or ids in seen_ids:continue
                    seen.add(digest);seen_ids.add(ids)
                    rows.append({'text':text,'id':digest,'window':i,'window_line':line_no,'tokens':len(ids)+1})
                    count+=1
                    if count==768:break
                if count!=768:raise RuntimeError(f'Window {i} insufficient unique rows: {count}')
            windows.append({'start':start,'end':end,'selected':count})
            print('window',i+1,'/24',flush=True)
        random.Random(42).shuffle(rows)
        manifest={'url':url,'file_bytes':total,'windows':windows,'sampling':'equal quotas from 24 byte windows; not uniform by record','seed':42,'splits':{}}
        for name,selected in [('train',rows[:16384]),('dev',rows[16384:17408]),('test',rows[17408:])]:
            path=run/f'{name}.jsonl'
            path.write_text(''.join(json.dumps({'text':x['text']},ensure_ascii=False)+'\n' for x in selected),encoding='utf-8')
            manifest['splits'][name]={'sha256':sha(path),'count':len(selected),'prediction_tokens':sum(x['tokens'] for x in selected),'samples':[{k:v for k,v in x.items() if k!='text'} for x in selected]}
        save('manifest.json',manifest)
        torch.manual_seed(42);base=MiniMindForCausalLM(MiniMindConfig())
        torch.save(base.state_dict(),run/'backbone_init.pt')
        torch.manual_seed(43);fish=FisherMiniMind(MiniMindConfig())
        keys=fish.load_state_dict(base.state_dict(),strict=False)
        assert not keys.unexpected_keys and all('.fisher.' in x for x in keys.missing_keys)
        assert all(torch.equal(v,fish.state_dict()[k]) for k,v in base.state_dict().items())
        torch.save({k:v for k,v in fish.state_dict().items() if '.fisher.' in k},run/'fisher_init.pt')
        order=torch.randperm(16384,generator=torch.Generator().manual_seed(42)).tolist();save('order.json',order)
        save('config.json',{'seed':42,'batch':2,'accumulation':8,'steps':1024,'length':340,'lr':.0005,'weight_decay':.01,'eval_steps':[0,256,512,1024],'index':3,'k':8,'dim':64,
             'hashes':{name:sha(run/name) for name in ['backbone_init.pt','fisher_init.pt','order.json','train.jsonl','dev.jsonl','test.jsonl']},
             'code_hashes':{name:sha(ROOT/name) for name in ['model/model_minimind.py','model/model_fisher.py','scripts/train_pair.py']}})
        result.update(status='passed',rows=len(rows),parameters={'baseline':sum(p.numel() for p in base.parameters()),'fisher':sum(p.numel() for p in fish.parameters())})
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:save('result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
