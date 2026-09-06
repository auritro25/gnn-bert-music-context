from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from transformers import AutoTokenizer

def _safe_id(x): return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in str(x))

def build_genre_vocab(cfg):
    d=cfg['data']; df=pd.read_csv(d['metadata_csv']); tr=df[df[d['split_col']].astype(str).str.lower()=='train']
    genres=sorted(g for g in tr[d['labels_col']].dropna().astype(str).unique() if g.strip())
    if not genres: raise ValueError('No training genres found.')
    return genres

class GenreGraphTextDataset(Dataset):
    def __init__(self,cfg,split,genre_vocab):
        d=cfg['data']; t=cfg['text']; df=pd.read_csv(d['metadata_csv'])
        self.df=df[df[d['split_col']].astype(str).str.lower()==split.lower()].reset_index(drop=True)
        self.d=d; self.graph_dir=Path(d['graph_dir']); self.g2i={g:i for i,g in enumerate(genre_vocab)}
        self.tokenizer=AutoTokenizer.from_pretrained(t['model_name'],use_fast=True); self.max_length=int(t['max_length'])
        unknown=sorted(set(self.df[d['labels_col']].dropna().astype(str))-set(genre_vocab))
        if unknown: raise ValueError(f'{split} contains genres absent from training vocabulary: {unknown}')
    def __len__(self): return len(self.df)
    def __getitem__(self,idx):
        r=self.df.iloc[idx]; tid=_safe_id(r[self.d['track_id_col']]); gp=self.graph_dir/f'{tid}.pt'
        g=torch.load(gp,map_location='cpu',weights_only=False); text=str(r[self.d['text_col']])
        tok=self.tokenizer(text,truncation=True,padding='max_length',max_length=self.max_length,return_tensors='pt')
        return {'track_id':tid,'graph':g,'input_ids':tok['input_ids'].squeeze(0),'attention_mask':tok['attention_mask'].squeeze(0),'labels':torch.tensor(self.g2i[str(r[self.d['labels_col']])],dtype=torch.long),'text':text}

def genre_collate(items):
    return {'track_id':[x['track_id'] for x in items],'graph':Batch.from_data_list([x['graph'] for x in items]),'input_ids':torch.stack([x['input_ids'] for x in items]),'attention_mask':torch.stack([x['attention_mask'] for x in items]),'labels':torch.stack([x['labels'] for x in items]),'text':[x['text'] for x in items]}
