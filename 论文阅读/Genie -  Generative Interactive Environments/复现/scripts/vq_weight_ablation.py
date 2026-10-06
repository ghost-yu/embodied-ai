"""One controlled second attempt: LAM external VQ loss weight .25 -> .01.
Same model init, batches and video corpus; no true labels in optimization.
"""
import json,time,torch
from genie_core_demo import *

def main():
    torch.manual_seed(SEED)
    tok,lam,dyn=make_models() # Consume the same initialization RNG as initial run.
    initial=torch.load(O/'small_core_state.pt',weights_only=True)
    tok.load_state_dict(initial['tokenizer'],strict=True);tok.eval()
    lam.vq_loss_w=.01
    # Match RNG state at start of original LAM training: tokenizer batch draws only.
    for _ in range(1201):torch.randint(512,(16,))
    videos,_=dataset(512,SEED);tests,labels=dataset(64,SEED+1)
    history=[];started=time.time()
    lam.train();opt=torch.optim.AdamW(lam.parameters(),lr=.001,weight_decay=0)
    for step in range(1801):
        ix=torch.randint(512,(16,));loss=lam(videos[ix]);opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(lam.parameters(),1);opt.step()
        history.append({'stage':'LAM','step':step,'loss':loss.item()})
    lam.eval()
    for m in [tok,lam]:
        for p in m.parameters():p.requires_grad_(False)
    with torch.no_grad():
        visual=tok(videos,return_only_codebook_ids=True);ids=lam(videos,return_only_codebook_ids=True);codes=action_codes(lam,ids)
    dyn.train();opt=torch.optim.AdamW(dyn.parameters(),lr=.001,weight_decay=0)
    for step in range(801):
        ix=torch.randint(512,(16,));loss=dyn(video_codebook_ids=visual[ix],actions=codes[ix]);opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(),1);opt.step()
        history.append({'stage':'dynamics','step':step,'loss':loss.item()})
    metrics,fixture=evaluate(tok,lam,dyn,tests,labels)
    metrics.update(LAM_vq_loss_weight=.01,wall_seconds=time.time()-started,
                   scope='same initialization and batches; only external LAM VQ loss weight changed; frozen original tokenizer')
    torch.save({n:m.state_dict() for n,m in [('tokenizer',tok),('LAM',lam),('dynamics',dyn)]},O/'vq001_state.pt')
    torch.save(fixture,O/'vq001_fixture.pt')
    (O/'VQ权重对照.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    (O/'VQ对照训练曲线.json').write_text(json.dumps(history),encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
