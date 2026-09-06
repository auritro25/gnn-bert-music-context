from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

def add_multi(rows,task,dataset,base,models):
    for model in models:
        p=Path(base)/model/'test_metrics.json'
        if p.exists():
            m=json.loads(p.read_text()); rows.append({'task':task,'dataset':dataset,'model':model,'macro_f1':m.get('macro_f1'),'micro_f1':m.get('micro_f1'),'macro_auc_pr':m.get('macro_auc_pr'),'micro_auc_pr':m.get('micro_auc_pr'),'accuracy':None})
def add_genre(rows,base,models):
    for model in models:
        p=Path(base)/model/'test_metrics.json'
        if p.exists():
            m=json.loads(p.read_text()); rows.append({'task':'Task 2: genre classification','dataset':'FMA-small','model':model,'macro_f1':m.get('macro_f1'),'micro_f1':None,'macro_auc_pr':None,'micro_auc_pr':None,'accuracy':m.get('accuracy')})
def main():
    rows=[]; add_multi(rows,'Task 1: leakage-controlled caption tagging','MusicCaps','results/musiccaps_masked',['bert']); add_genre(rows,'results/fma_small_genre',['cnn','gnn']); add_multi(rows,'Task 3: GNN-BERT fusion','FMA-medium','results/fma_medium_tags',['bert','gnn','concat','fusion'])
    df=pd.DataFrame(rows); Path('results').mkdir(exist_ok=True); df.to_csv('results/strict_compliance_comparison.csv',index=False); print(df.to_string(index=False) if len(df) else 'No strict-compliance result files found yet.'); print('\nSaved results/strict_compliance_comparison.csv')
if __name__=='__main__': main()
