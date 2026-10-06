"""Official I-JEPA classes, collator, optimizer/schedules: small synthetic training."""
import sys,json,time,copy,hashlib,subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4)
P=Path(__file__).resolve().parents[2];R=P/'复现';O=R/'outputs';REPO=P/'代码'/'ijepa'
sys.path.insert(0,str(REPO))
from src.models.vision_transformer import VisionTransformer,VisionTransformerPredictor
from src.masks.multiblock import MaskCollator
from src.utils.tensors import apply_masks,repeat_interleave_batch
from src.helper import init_opt
from src.utils.distributed import AllReduce
SEED=109;UPDATES=501

def make_models():
    enc=VisionTransformer(img_size=[64],patch_size=8,embed_dim=64,depth=2,num_heads=4)
    pred=VisionTransformerPredictor(num_patches=64,embed_dim=64,predictor_embed_dim=32,depth=2,num_heads=4)
    target=copy.deepcopy(enc)
    for p in target.parameters():p.requires_grad_(False)
    return enc,pred,target

def dataset(n,seed):
    g=torch.Generator().manual_seed(seed)
    yy,xx=torch.meshgrid(torch.linspace(0,1,64),torch.linspace(0,1,64),indexing='ij')
    output=[]
    for _ in range(n):
        center=.2+.6*torch.rand(2,generator=g);radius=.15+.15*torch.rand(1,generator=g)
        if torch.rand(1,generator=g)<.5:shape=((xx-center[0]).square()+(yy-center[1]).square()<radius.square()).float()
        else:shape=((xx-center[0]).abs()<radius)*((yy-center[1]).abs()<radius)
        color=.2+.8*torch.rand(3,1,1,generator=g)
        background=.15*torch.rand(3,1,1,generator=g)+.05*xx
        output.append(background+color*shape)
    return torch.stack(output)

def collator(nenc=1):
    return MaskCollator(input_size=64,patch_size=8,enc_mask_scale=(.85,1.),pred_mask_scale=(.15,.2),
                        aspect_ratio=(.75,1.5),nenc=nenc,npred=4,min_keep=4,allow_overlap=False)

def target_features(target,imgs,me,mp):
    with torch.no_grad():
        h=target(imgs);h=F.layer_norm(h,(h.size(-1),));h=apply_masks(h,mp)
        return repeat_interleave_batch(h,len(imgs),repeat=len(me))

@torch.no_grad()
def evaluate(enc,pred,target,imgs,me,mp):
    h=target_features(target,imgs,me,mp);z=pred(enc(imgs,me),me,mp)
    full=target(imgs);pooled=full.mean(1)
    return {'smooth_l1':F.smooth_l1_loss(z,h).item(),'prediction_mse':F.mse_loss(z,h).item(),
            'target_feature_std_across_images_mean':full.std(0).mean().item(),
            'pooled_feature_std_across_images_mean':pooled.std(0).mean().item()},z,h

def overlap_count(me,mp):
    return sum(int(torch.isin(e[i],p[i]).sum()) for e in me for p in mp for i in range(len(e)))

def train():
    O.mkdir(parents=True,exist_ok=True);torch.manual_seed(SEED)
    imgs=dataset(256,SEED);held=dataset(16,SEED+1);enc,pred,target=make_models()
    c=collator();evc=collator();_,me,mp=evc([x for x in held])
    initial,_,_=evaluate(enc,pred,target,held,me,mp)
    opt,scaler,lrs,wds=init_opt(enc,pred,iterations_per_epoch=1,start_lr=.0001,ref_lr=.001,warmup=20,
                               num_epochs=UPDATES,wd=.04,final_wd=.1,final_lr=.00001,use_bfloat16=False,ipe_scale=1.)
    history=[];overlaps=0;ema_error=0.;t0=time.time();grads={}
    for step in range(UPDATES):
        ix=torch.randint(256,(8,));batch,em,pm=c([imgs[i] for i in ix]);overlaps+=overlap_count(em,pm)
        lr=lrs.step();wd=wds.step();h=target_features(target,batch,em,pm)
        z=pred(enc(batch,em),em,pm);loss=AllReduce.apply(F.smooth_l1_loss(z,h))
        opt.zero_grad();loss.backward()
        if step==0:
            grads={'patch_embed':float(enc.patch_embed.proj.weight.grad.norm()),'mask_token':float(pred.mask_token.grad.norm()),
                   'predictor_proj':float(pred.predictor_proj.weight.grad.norm()),'target_has_any_grad':any(p.grad is not None for p in target.parameters())}
        opt.step();momentum=.996+step*.004/UPDATES
        with torch.no_grad():
            for j,(q,k) in enumerate(zip(enc.parameters(),target.parameters())):
                if j==1:expected=k.clone()*momentum+q.detach()*(1-momentum)
                k.data.mul_(momentum).add_((1-momentum)*q.detach().data)
                if j==1:ema_error=max(ema_error,float((k-expected).abs().max()))
        history.append({'step':step,'loss':loss.item(),'lr':lr,'wd':wd,'momentum':momentum,'context_tokens':em[0].shape[-1],'target_tokens':pm[0].shape[-1]})
        if step%100==0:print(step,loss.item(),flush=True)
    enc.eval();pred.eval();target.eval();final,z,h=evaluate(enc,pred,target,held,me,mp)
    # Masked target pixels do not enter the context transformer (fixed-size nonoverlap patches).
    changed=held.clone()
    for b in range(len(held)):
        for ids in mp:
            for index in ids[b]:
                y,x=divmod(int(index),8);changed[b,:,y*8:(y+1)*8,x*8:(x+1)*8]=torch.rand(3,8,8)
    with torch.no_grad():
        orig_context=enc(held,me);new_context=enc(changed,me)
    leakage=float((orig_context-new_context).abs().max());assert leakage==0
    assert not grads['target_has_any_grad'] and all(grads[k]>0 for k in ['patch_embed','mask_token','predictor_proj'])
    # Explicit identity trail for B=2, two contexts, four target lists.
    tagged=torch.stack([torch.arange(64),100+torch.arange(64)])[:,:,None].float()
    lists=[torch.tensor([[i],[i+1]]) for i in range(4)]
    expanded=repeat_interleave_batch(apply_masks(tagged,lists),2,repeat=2).flatten().tolist()
    expected=[x for i in range(4) for _ in range(2) for x in [float(i),float(101+i)]]
    assert expanded==expected
    torch.save({'encoder':enc.state_dict(),'predictor':pred.state_dict(),'target':target.state_dict()},O/'small_core_state.pt')
    torch.save({'images':held,'masks_enc':me,'masks_pred':mp,'prediction':z,'target':h},O/'fixture.pt')
    report={'label':'官方核心小配置合成训练；完整I-JEPA论文未复现','seed':SEED,'updates':UPDATES,'torch':torch.__version__,
            'parameters':{'encoder':sum(p.numel() for p in enc.parameters()),'predictor':sum(p.numel() for p in pred.parameters())},
            'initial_validation':initial,'final_validation':final,'shapes':{'context':list(orig_context.shape),'prediction':list(z.shape),'target':list(h.shape)},
            'initial_gradients':grads,'EMA_independent_formula_max_error':ema_error,'masked_pixel_context_max_difference':leakage,
            'context_target_overlap_index_count':overlaps,'mask_batch_identity_example':expanded,'wall_seconds':time.time()-t0,
            'repo_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
            'source_sha256':{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [REPO/'src/models/vision_transformer.py',REPO/'src/masks/multiblock.py',REPO/'src/helper.py',REPO/'src/utils/tensors.py',REPO/'src/utils/schedulers.py',REPO/'src/utils/distributed.py']}}
    (O/'结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (O/'训练曲线.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':train()
