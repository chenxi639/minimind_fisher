"""Explicit allowlist; selection must never depend on the Git index."""
import json
from pathlib import Path,PurePosixPath
def select_sources(root,manifest):
    root=Path(root).resolve();data=json.loads(Path(manifest).read_text(encoding='utf-8'))
    if set(data)!={'schema_version','research_scripts'} or data['schema_version']!=1:raise ValueError('unsupported source manifest')
    names=data['research_scripts']
    if not isinstance(names,list) or not names or any(not isinstance(n,str) for n in names):raise ValueError('invalid script list')
    if len(set(names))!=len(names):raise ValueError('duplicate source')
    paths=[]
    for name in names:
        p=PurePosixPath(name)
        if len(p.parts)!=2 or p.parts[0]!='scripts' or p.suffix!='.py' or '\\' in name or ':' in name or p.as_posix()!=name:raise ValueError('invalid source path: '+name)
        target=(root/name).resolve()
        if not target.is_relative_to(root) or not target.is_file():raise ValueError('missing or escaping source: '+name)
        paths.append(target)
    return paths
