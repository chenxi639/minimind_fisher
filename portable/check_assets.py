"""Read-only asset validation. Paths live only in an uncommitted local mapping."""
import argparse,hashlib,json,re
from pathlib import Path
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def check(manifest_path,mapping_path):
    m=json.loads(Path(manifest_path).read_text(encoding='utf-8'));mapping_path=Path(mapping_path).resolve()
    mapping=json.loads(mapping_path.read_text(encoding='utf-8'))
    if set(m)!={'schema_version','experiment_id','assets'} or m['schema_version']!=1:raise ValueError('unsupported manifest schema')
    if not isinstance(m['experiment_id'],str) or not m['experiment_id']:raise ValueError('missing experiment id')
    if not isinstance(m['assets'],dict) or not m['assets']:raise ValueError('empty assets')
    if not isinstance(mapping,dict) or set(mapping)!=set(m['assets']):raise ValueError('mapping must match all and only declared assets')
    result={}
    for name,a in m['assets'].items():
        if set(a)!={'sha256','role'} or not re.fullmatch('[0-9a-f]{64}',a['sha256']):raise ValueError('invalid asset entry')
        if not isinstance(mapping[name],str) or not isinstance(a['role'],str):raise ValueError('invalid role or path')
        p=Path(mapping[name]);p=p if p.is_absolute() else mapping_path.parent/p
        if not p.is_file():raise FileNotFoundError(name)
        actual=digest(p)
        if actual!=a['sha256']:raise ValueError('hash mismatch: '+name)
        result[name]={'sha256':actual,'bytes':p.stat().st_size,'status':'passed'}
    return {'status':'passed','schema_version':1,'experiment_id':m['experiment_id'],'assets':result,'meaning':'identity only; not license, dataset cleanliness or reproducibility certification'}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--mapping',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(check(args.manifest,args.mapping),indent=2))
