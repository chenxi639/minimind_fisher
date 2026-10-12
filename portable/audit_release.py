"""Static candidate checks. Not a comprehensive secrets/license certification."""
import argparse,ast,hashlib,json,re
from pathlib import Path
def audit(root):
    root=Path(root).resolve();manifest=json.loads((root/'EXPORT_MANIFEST.json').read_text());issues=[]
    def issue(code,path,line=None):issues.append({'code':code,'file':str(path),'line':line})
    for rel,h in manifest['generated_files'].items():
        p=root/rel
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:issue('manifest_mismatch',rel)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    for rel in actual-set(manifest['generated_files'])-{'EXPORT_MANIFEST.json'}:issue('unmanifested_file',rel)
    patterns={'private_user_path':r'[A-Za-z]:[/\\]+Users[/\\]+(?!<)',
              'private_key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
              'github_token':r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}',
              'provider_token':r'\bsk-[A-Za-z0-9_-]{24,}'}
    links=0
    for rel in sorted(actual):
        p=root/rel
        if p.suffix.lower() in ['.pt','.pth','.safetensors','.bin','.jsonl','.tsv','.log'] or '.local.' in p.name or p.name.startswith('.env'):issue('excluded_asset',rel)
        text=p.read_text(encoding='utf-8',errors='replace')
        for number,line in enumerate(text.splitlines(),1):
            for key,pattern in patterns.items():
                if re.search(pattern,line):issue(key,rel,number)
        if p.suffix=='.py':
            try:ast.parse(text)
            except SyntaxError:issue('python_syntax',rel)
        if p.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if '://' in target or target.startswith('#'):continue
                target=target.split('#')[0]
                if not target:continue
                links+=1
                if not (p.parent/target).is_file():issue('missing_document_link',rel)
    for rel in ['LICENSE','.gitignore','README.md','VALIDATION.md']:
        if rel not in actual:issue('required_file_missing',rel)
    return {'status':'passed' if not issues else 'needs_review','file_count':len(actual),'relative_links_checked':links,'issues':issues,'scope':'static patterns and integrity only; not legal review or clean installation','git_operations':0}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);a=p.parse_args();r=audit(a.candidate);print(json.dumps(r,indent=2));raise SystemExit(0 if r['status']=='passed' else 1)
