import sys,json,hashlib,re,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pyarrow.parquet as pq
from transformers import AutoTokenizer
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s)).casefold()
def shingles(s):return {hashlib.blake2b(s[i:i+20].encode(),digest_size=8).digest() for i in range(max(0,len(s)-19))}
def main():
    out=ROOT/'dataset/external_wikipedia_20a';assert not (out/'manifest.json').exists()
    src=out/'source.parquet';assert sha(src)=='ec8f6c0dd1418b8fcd454278fb4f2bc7cd0ce8312f29c80b672533942601338f'
    tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    known_text=set();known_norm=set();known_ids=set();known_sh=set()
    for split in ['train','dev']:
        path=ROOT/'runs/pair_ready_20260929_181458'/(split+'.jsonl')
        for line in path.read_text(encoding='utf-8').splitlines():
            text=json.loads(line)['text'];ids=tok(text,add_special_tokens=False,truncation=True,max_length=338).input_ids
            known_text.add(text.strip());known_norm.add(norm(text));known_ids.add(tuple(ids));known_sh.update(shingles(norm(tok.decode(ids))))
    print('training/development overlap index ready',flush=True)
    rows=pq.read_table(src).to_pylist();rows.sort(key=lambda r:hashlib.sha256(('Stage20A:'+str(r['id'])).encode()).hexdigest())
    accepted=[];reject={};seen_id=set();seen_norm=set();seen_tokens=set()
    for row in rows:
        text=row['text'];ids=tok(text,add_special_tokens=False,truncation=True,max_length=338).input_ids;n=norm(text);key=tuple(ids)
        reason=None
        if len(ids)<128:reason='short'
        elif str(row['id']) in seen_id or n in seen_norm or key in seen_tokens:reason='within_duplicate'
        elif text.strip() in known_text or n in known_norm or key in known_ids:reason='known_exact'
        sh=shingles(norm(tok.decode(ids)));fraction=len(sh&known_sh)/len(sh) if sh else 0
        if reason is None and fraction>=.5:reason='known_shingle_union_ge_0.5'
        if reason:reject[reason]=reject.get(reason,0)+1;continue
        seen_id.add(str(row['id']));seen_norm.add(n);seen_tokens.add(key)
        accepted.append({'id':str(row['id']),'url':row['url'],'title':row['title'],'text':text,'content_ids':ids,'known_shingle_fraction':fraction})
        if len(accepted)==1024:break
    assert len(accepted)==1024
    data=out/'eval.jsonl';data.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted),encoding='utf-8')
    summary=json.loads((ROOT/'runs/frozen_multiseed_summary.json').read_text());models={'pure':{'backbone':str(ROOT/'checkpoints/official_edba70e/download.pth')}}
    for seed,folder in [(42,'frozen_pair_20261002_145450'),(2026,'frozen_pair_seed2026_20261002_204516'),(3407,'frozen_pair_seed3407_20261002_211149')]:
        for arm in ['fisher','mlp']:
            p=ROOT/'runs'/folder/arm/'checkpoint.pt';digest=sha(p);assert summary['checkpoint_hashes'][str(p)]==digest
            models[f'{arm}_{seed}']={'path':str(p),'sha256':digest}
    models['pure']['sha256']=sha(Path(models['pure']['backbone']))
    manifest={'status':'locked_before_evaluation','source_revision':'b04c8d1ceb2f5cd4588862100d08de323dccfbaa','source_file':'20231101.zh/train-00002-of-00006.parquet','source_sha256':sha(src),'license_card':['cc-by-sa-3.0','gfdl'],'source_rows':len(rows),'accepted':1024,'rejections':reject,'max_known_shingle_fraction':max(r['known_shingle_fraction'] for r in accepted),'data_sha256':sha(data),'prediction_tokens':sum(len(r['content_ids'])+1 for r in accepted),'models':models,'protocol_sha256':sha(ROOT/'report/20A_external_protocol.md'),'tokenizer_hashes':{n:sha(ROOT/'model'/n) for n in ['tokenizer.json','tokenizer_config.json']},'code_hashes':{n:sha(ROOT/n) for n in ['scripts/prepare_external20.py','scripts/evaluate_external20.py','model/model_minimind.py','model/model_fisher.py','model/model_matched_mlp.py']},'official_pretraining_overlap':'unknown; not claimed contamination-free','primary_precision':'fp32_tf32_off','bootstrap_seed':20261002,'bootstrap_repeats':2000}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':main()
