"""Read-only eval observer; no RNG use, parameter changes, or train-mode work."""
import math,json
import torch
import torch.nn.functional as F

class AssignmentMonitor:
    def __init__(self,model,path):
        self.model=model;self.path=path;self.rows=[];self.enabled=False;self.scale=1.
        self.branch=model.model.layers[3].fisher
        self.handle=self.branch.register_forward_hook(self.hook)
    def start(self):
        self.enabled=True;self.batch=0;self.q=[];self.cos=[];self.ratio=[];self.mask=None
    def hook(self,layer,args,output):
        delta,state=output
        if self.enabled:
            if self.batch%32==0:
                with torch.no_grad(),torch.autocast('cuda',enabled=False):
                    h=args[0].float();valid=self.mask.clone();valid[1:]=False
                    x=F.linear(layer.norm(h),layer.input.weight.float())
                    z=layer.w.float()[None,None]*(x[:,:,None]+layer.b.float()[None,None])
                    q=(-.5*z.square().sum(-1)).softmax(-1)
                    self.q.append(q[valid].cpu())
                    assert layer.window==1
                    a=q[...,None]*z;b=q[...,None]*(z.square()-1)/math.sqrt(2)
                    count=torch.ones(h.shape[1],device=h.device)
                    v=(torch.cat([a,b],-1)/count[None,:,None,None])[valid]
                    vn=F.normalize(v/torch.sqrt(v.abs()+1e-6),dim=-1)
                    sims=vn@vn.transpose(-1,-2);ix=torch.triu_indices(8,8,1,device=h.device)
                    self.cos.append(sims[:,ix[0],ix[1]].cpu().flatten())
                    self.ratio.append((delta.float()[valid].norm(dim=-1)/h[valid].norm(dim=-1).clamp_min(1e-12)).cpu())
            self.batch+=1
        return delta*self.scale,state
    def finish(self):
        self.enabled=False
        q=torch.cat(self.q);usage=q.mean(0)
        blocks=self.branch.output.weight.detach().float().reshape(768,2,8,64).permute(2,0,1,3).reshape(8,-1)
        norm=F.normalize(blocks,dim=-1);cos=norm@norm.T;ix=torch.triu_indices(8,8,1,device=cos.device)
        row={'evaluation_index':len(self.rows),'observed_batches':self.batch,'sample_indices':list(range(0,self.batch*2,64)),
             'tokens':len(q),'soft_occupancy':usage.tolist(),'hard_counts':torch.bincount(q.argmax(-1),minlength=8).tolist(),
             'mean_entropy':float(-(q*q.clamp_min(1e-30).log()).sum(-1).mean()),
             'mean_max_q':float(q.max(-1).values.mean()),'token_std':q.std(0).tolist(),
             'effective_components_mean_usage':float((-(usage*usage.log()).sum()).exp()),
             'statistic_cosine':float(torch.cat(self.cos).mean()),
             'output_block_cosine':float(cos[ix[0],ix[1]].mean()) if blocks.norm()>0 else None,
             'residual_hidden_ratio':float(torch.cat(self.ratio).mean())}
        self.rows.append(row)
        self.path.write_text(json.dumps(self.rows,indent=2),encoding='utf-8')
        print('ASSIGNMENT',row['evaluation_index'],row['mean_entropy'],row['statistic_cosine'],flush=True)
