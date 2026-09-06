from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from .cnn_model import CNNMelBaseline
from .evaluation_genre import genre_metrics,save_genre_artifacts
from .genre_datasets import build_genre_vocab
from .utils import ensure_dir,load_config,resolve_device,set_seed

def safe_id(x): return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in str(x))
class GenreMelDataset(Dataset):
    def __init__(self,cfg,split,genres):
        d=cfg['data']; df=pd.read_csv(d['metadata_csv']); self.df=df[df[d['split_col']].astype(str).str.lower()==split].reset_index(drop=True); self.d=d; self.mel_dir=Path(d['mel_dir']); self.g2i={g:i for i,g in enumerate(genres)}
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]; tid=safe_id(r[self.d['track_id_col']]); mel=np.load(self.mel_dir/f'{tid}.npy').astype(np.float32); y=self.g2i[str(r[self.d['labels_col']])]; return {'track_id':tid,'mel':torch.tensor(mel[None],dtype=torch.float32),'labels':torch.tensor(y)}
def run(model,loader,criterion,device,opt=None):
    train=opt is not None; model.train(train); losses=[]; ys=[]; probs=[]; ids=[]
    for b in tqdm(loader,leave=False):
        x,y=b['mel'].to(device),b['labels'].long().to(device); ids.extend(b['track_id'])
        if train: opt.zero_grad(set_to_none=True)
        logits=model(x); loss=criterion(logits,y)
        if train: loss.backward(); opt.step()
        losses.append(loss.item()); ys.append(y.detach().cpu().numpy()); probs.append(torch.softmax(logits,dim=-1).detach().cpu().numpy())
    return {'loss':float(np.mean(losses)),'y_true':np.concatenate(ys),'y_prob':np.concatenate(probs),'track_ids':ids}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args(); cfg=load_config(a.config); set_seed(int(cfg['seed'])); device=resolve_device(cfg.get('device','cuda:0')); genres=build_genre_vocab(cfg)
    ds=[GenreMelDataset(cfg,s,genres) for s in ['train','val','test']]; bs=int(cfg['train']['batch_size']); nw=int(cfg['train']['num_workers']); mk=lambda x,sh:DataLoader(x,batch_size=bs,shuffle=sh,num_workers=nw); trl,val,tel=mk(ds[0],True),mk(ds[1],False),mk(ds[2],False)
    model=CNNMelBaseline(len(genres),float(cfg['model']['dropout'])).to(device); criterion=nn.CrossEntropyLoss(); opt=AdamW(model.parameters(),lr=float(cfg['train']['lr_head']),weight_decay=float(cfg['train']['weight_decay'])); out=ensure_dir(Path(cfg['train']['output_dir'])/'cnn')
    best=-1.; state=None; stale=0; hist=[]
    for epoch in range(1,int(cfg['train']['epochs'])+1):
        tr=run(model,trl,criterion,device,opt); va=run(model,val,criterion,device); m=genre_metrics(va['y_true'],va['y_prob'].argmax(1)); hist.append({'epoch':epoch,'train_loss':tr['loss'],'val_loss':va['loss'],'val_macro_f1':m['macro_f1'],'val_accuracy':m['accuracy']}); print(f"Epoch {epoch:02d} | val_macro_f1={m['macro_f1']:.4f} | val_accuracy={m['accuracy']:.4f}")
        if m['macro_f1']>best: best=m['macro_f1']; state=copy.deepcopy(model.state_dict()); torch.save({'model_state':state,'genres':genres},out/'best_model.pt'); stale=0
        else:
            stale+=1
            if stale>=int(cfg['train']['patience']): break
    pd.DataFrame(hist).to_csv(out/'history.csv',index=False); model.load_state_dict(state); te=run(model,tel,criterion,device); metrics=save_genre_artifacts(te['y_true'],te['y_prob'],genres,out,te['track_ids']); metrics['test_loss']=te['loss']; (out/'test_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8'); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
