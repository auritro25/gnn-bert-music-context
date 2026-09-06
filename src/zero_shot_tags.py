from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import average_precision_score,f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from .contrastive import GraphTextContrastiveModel
from .datasets import PairedGraphTextDataset,graph_text_collate
from .factory import build_components
from .labels import load_label_vocab
from .utils import load_config,resolve_device

def move(b,d):
    z=dict(b); z['graph']=b['graph'].to(d); z['input_ids']=b['input_ids'].to(d); z['attention_mask']=b['attention_mask'].to(d); return z

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--split',default='test'); p.add_argument('--top-k',type=int,default=3); p.add_argument('--output',required=True); a=p.parse_args()
    cfg=load_config(a.config); device=resolve_device(cfg.get('device','cuda:0')); labels=load_label_vocab(cfg['data']['label_vocab'])
    gnn,text=build_components(cfg); model=GraphTextContrastiveModel(gnn,text,int(cfg['model']['hidden_dim']),int(cfg['contrastive']['projection_dim']),float(cfg['contrastive']['temperature'])).to(device)
    ck=torch.load(a.checkpoint,map_location=device,weights_only=False); model.load_state_dict(ck['model_state']); model.eval()
    tok=AutoTokenizer.from_pretrained(cfg['text']['model_name'],use_fast=True); prompts=[f'Music described as {x}.' for x in labels]; t=tok(prompts,padding=True,truncation=True,max_length=int(cfg['text']['max_length']),return_tensors='pt')
    with torch.no_grad(): tag_emb=model.encode_text(t['input_ids'].to(device),t['attention_mask'].to(device)).cpu().numpy()
    ds=PairedGraphTextDataset(cfg,a.split); dl=DataLoader(ds,batch_size=int(cfg['contrastive']['batch_size']),shuffle=False,num_workers=int(cfg['train']['num_workers']),collate_fn=graph_text_collate)
    scores=[]; ys=[]
    with torch.no_grad():
        for raw in dl:
            b=move(raw,device); g=model.encode_graph(b['graph']).cpu().numpy(); scores.append(g@tag_emb.T); ys.append(raw['labels'].numpy())
    score=np.concatenate(scores); y=np.concatenate(ys); pred=np.zeros_like(y,dtype=int); k=min(a.top_k,pred.shape[1]); top=np.argpartition(score,-k,axis=1)[:,-k:]
    for i,idx in enumerate(top): pred[i,idx]=1
    metrics={'split':a.split,'top_k':k,'macro_auc_pr':float(average_precision_score(y,score,average='macro')),'micro_auc_pr':float(average_precision_score(y,score,average='micro')),'macro_f1_topk':float(f1_score(y,pred,average='macro',zero_division=0)),'micro_f1_topk':float(f1_score(y,pred,average='micro',zero_division=0))}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(metrics,indent=2),encoding='utf-8'); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
