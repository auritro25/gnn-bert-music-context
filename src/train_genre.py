from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from .evaluation_genre import genre_metrics,save_genre_artifacts
from .factory import build_supervised_model
from .genre_datasets import GenreGraphTextDataset,build_genre_vocab,genre_collate
from .utils import ensure_dir,load_config,resolve_device,save_json,set_seed

def move(b,d):
    z=dict(b); z['graph']=b['graph'].to(d); z['input_ids']=b['input_ids'].to(d); z['attention_mask']=b['attention_mask'].to(d); z['labels']=b['labels'].to(d); return z

def optimizer_for(model,cfg):
    bert=[]; other=[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        (bert if 'text_encoder.transformer' in n else other).append(p)
    groups=[]
    if bert: groups.append({'params':bert,'lr':float(cfg['train']['lr_bert'])})
    if other: groups.append({'params':other,'lr':float(cfg['train']['lr_head'])})
    return AdamW(groups,weight_decay=float(cfg['train']['weight_decay']))

def run(model,loader,criterion,device,opt=None,use_amp=False):
    train=opt is not None; model.train(train); losses=[]; ys=[]; probs=[]; ids=[]
    for raw in tqdm(loader,leave=False):
        ids.extend(raw['track_id']); b=move(raw,device)
        if train: opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=(use_amp and device.type=='cuda')):
            o=model(b); loss=criterion(o['logits'],b['labels'])
        if train: loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        losses.append(loss.item()); ys.append(b['labels'].detach().cpu().numpy()); probs.append(torch.softmax(o['logits'],dim=-1).detach().cpu().numpy())
    return {'loss':float(np.mean(losses)),'y_true':np.concatenate(ys),'y_prob':np.concatenate(probs),'track_ids':ids}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--model',required=True,choices=['bert','gnn','concat','fusion']); a=p.parse_args()
    cfg=load_config(a.config); set_seed(int(cfg['seed'])); device=resolve_device(cfg.get('device','cuda:0')); genres=build_genre_vocab(cfg)
    ds=[GenreGraphTextDataset(cfg,s,genres) for s in ['train','val','test']]; bs=int(cfg['train']['batch_size']); nw=int(cfg['train']['num_workers'])
    mk=lambda x,sh:DataLoader(x,batch_size=bs,shuffle=sh,num_workers=nw,collate_fn=genre_collate,pin_memory=(device.type=='cuda'))
    trl,val,tel=mk(ds[0],True),mk(ds[1],False),mk(ds[2],False)
    model=build_supervised_model(cfg,a.model,len(genres)).to(device); criterion=nn.CrossEntropyLoss(); opt=optimizer_for(model,cfg); amp=bool(cfg['train'].get('use_amp',True))
    out=ensure_dir(Path(cfg['train']['output_dir'])/a.model); save_json({'genres':genres},out/'genres.json'); save_json(cfg,out/'resolved_config.json')
    hist=[]; best=-1.; state=None; stale=0
    for epoch in range(1,int(cfg['train']['epochs'])+1):
        tr=run(model,trl,criterion,device,opt,amp); va=run(model,val,criterion,device,None,amp); m=genre_metrics(va['y_true'],va['y_prob'].argmax(1)); hist.append({'epoch':epoch,'train_loss':tr['loss'],'val_loss':va['loss'],'val_macro_f1':m['macro_f1'],'val_accuracy':m['accuracy']}); print(f"Epoch {epoch:02d} | train_loss={tr['loss']:.4f} | val_loss={va['loss']:.4f} | val_macro_f1={m['macro_f1']:.4f}")
        if m['macro_f1']>best: best=m['macro_f1']; state=copy.deepcopy(model.state_dict()); torch.save({'model_state':state,'model_type':a.model,'genres':genres},out/'best_model.pt'); stale=0
        else:
            stale+=1
            if stale>=int(cfg['train']['patience']): print('Early stopping.'); break
    pd.DataFrame(hist).to_csv(out/'history.csv',index=False); model.load_state_dict(state); te=run(model,tel,criterion,device,None,amp); metrics=save_genre_artifacts(te['y_true'],te['y_prob'],genres,out,te['track_ids']); metrics['test_loss']=te['loss']; (out/'test_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
