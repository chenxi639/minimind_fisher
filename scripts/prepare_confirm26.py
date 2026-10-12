"""Text-only new-source audit; no model loading or scoring."""
import sys,json,hashlib,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from transformers import AutoTokenizer
from scripts.prepare_external20 import norm,shingles
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'dataset/external_csl_26a';assert not (out/'manifest.json').exists()
    assert sha(out/'source.tsv')=='1db0be1d98d06dfa359460b23a3cb0f1fd9d4371caa95b0549a71ed4f56c3636'
    files={str(p):sha(p) for p in [out/'source.tsv',out/'source_readme.md',ROOT/'report/26A_protocol.md',Path(__file__),ROOT/'scripts/prepare_external20.py',ROOT/'model/tokenizer.json',ROOT/'model/tokenizer_config.json']}
    tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    known_text=set();known_norm=set();known_ids=set();known_sh=set();source_counts={}
    source=ROOT/'runs/pair_ready_20260929_181458'
    paths=[source/'train.jsonl',source/'dev.jsonl',ROOT/'dataset/external_wikipedia_20a/eval.jsonl']
    oldcfg=json.loads((source/'config.json').read_text());wiki=json.loads((ROOT/'dataset/external_wikipedia_20a/manifest.json').read_text())
    for p in paths:
        expected=wiki['data_sha256'] if p.name=='eval.jsonl' else oldcfg['hashes'][p.name]
        assert sha(p)==expected;files[str(p)]=expected;count=0
        for line in p.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);text=row['text'];ids=row.get('content_ids')
            if ids is None:ids=tok(text,add_special_tokens=False,truncation=True,max_length=338).input_ids
            known_text.add(text.strip());known_norm.add(norm(text));known_ids.add(tuple(ids));known_sh.update(shingles(norm(tok.decode(ids))));count+=1
        source_counts[str(p)]=count
    print('known-source index complete',source_counts,flush=True)
    with (out/'source.tsv').open(encoding='utf-8',newline='') as f:raw=list(csv.reader(f,delimiter='\t',quoting=csv.QUOTE_NONE))
    assert all(len(row)==3 and row[0]=='to title' for row in raw),'unexpected source schema'
    rows=[(i+1,row[1]) for i,row in enumerate(raw)]
    rows.sort(key=lambda r:hashlib.sha256(('Stage26A:'+str(r[0])).encode()).hexdigest())
    accepted=[];rejected=[];seen_norm=set();seen_ids=set();seen_sh=set()
    for row_id,text in rows:
        ids=tok(text,add_special_tokens=False,truncation=True,max_length=338).input_ids;n=norm(text);key=tuple(ids)
        sh=shingles(norm(tok.decode(ids)));fraction=len(sh&known_sh)/len(sh) if sh else 0;within=len(sh&seen_sh)/len(sh) if sh else 0
        reason=None
        if len(ids)<128:reason='short'
        elif text.strip() in known_text or n in known_norm or key in known_ids:reason='known_exact'
        elif n in seen_norm or key in seen_ids:reason='within_exact'
        elif fraction>=.5:reason='known_shingle_union_ge_0.5'
        elif within>=.5:reason='within_shingle_union_ge_0.5'
        if reason:rejected.append({'row':row_id,'reason':reason});continue
        seen_norm.add(n);seen_ids.add(key);seen_sh.update(sh)
        accepted.append({'id':f'csl-ts-train-line-{row_id}','source_row':row_id,'source_url':'https://github.com/ydli-ai/CSL/blob/0f1ac0626d584cd3c72f8be26d329eb4f967cfd0/benchmark/ts/train.tsv','text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),'content_ids':ids,'known_shingle_fraction':fraction,'within_shingle_fraction':within})
        if len(accepted)==1024:break
    counts={reason:sum(r['reason']==reason for r in rejected) for reason in sorted({r['reason'] for r in rejected})}
    audit={'source_rows':len(rows),'accepted':len(accepted),'rejections':counts,'rejected_rows':rejected,'known_source_counts':source_counts,'max_known_shingle_fraction':max(r['known_shingle_fraction'] for r in accepted),'max_within_shingle_fraction':max(r['within_shingle_fraction'] for r in accepted)}
    (out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');assert len(accepted)==1024
    data=out/'eval.jsonl';data.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in accepted),encoding='utf-8')
    for p,d in files.items():assert sha(p)==d,p
    manifest={'status':'locked_before_any_model_evaluation','revision':'0f1ac0626d584cd3c72f8be26d329eb4f967cfd0','source':'ydli-ai/CSL benchmark/ts/train.tsv abstract column, no header','license_statement':'repository README Apache-2.0; underlying publication rights not individually audited','files':files,'data_sha256':sha(data),'audit_sha256':sha(out/'audit.json'),'records':1024,'prediction_tokens':sum(len(r['content_ids'])+1 for r in accepted),'candidates':['pure','k1','current','lora'],'training_seeds':[1729,2718,31415],'official_pretraining_exposure':'unknown','model_evaluations':0,'optimizer_updates':0,'sealed_test_opened':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps({**audit,'rejected_rows':'see audit.json','prediction_tokens':manifest['prediction_tokens'],'data_sha256':manifest['data_sha256']},indent=2),flush=True)
if __name__=='__main__':main()
