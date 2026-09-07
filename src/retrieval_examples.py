from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--embeddings',required=True); p.add_argument('--metadata',required=True); p.add_argument('--output',required=True); p.add_argument('--num-queries',type=int,default=10); p.add_argument('--top-k',type=int,default=3); p.add_argument('--track-id-col',default='track_id'); p.add_argument('--text-col',default='caption'); a=p.parse_args()
    z=np.load(a.embeddings,allow_pickle=True); graph=z['graph']; text=z['text']; ids=[str(x) for x in z['track_ids']]; sim=text@graph.T
    meta=pd.read_csv(a.metadata); lookup={str(r[a.track_id_col]):str(r[a.text_col]) for _,r in meta.iterrows()}
    rows=[]; n=min(a.num_queries,len(ids))
    for qi in range(n):
        order=np.argsort(sim[qi])[::-1][:a.top_k]; qid=ids[qi]
        for rank,j in enumerate(order,1):
            rows.append({'query_index':qi,'query_track_id':qid,'query_caption':lookup.get(qid,''),'rank':rank,'retrieved_track_id':ids[j],'retrieved_caption_reference':lookup.get(ids[j],''),'similarity':float(sim[qi,j]),'is_exact_pair':bool(qi==j)})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False); out.with_suffix('.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8'); print(f'Saved {n} qualitative queries x top-{a.top_k} retrievals -> {out}')
if __name__=='__main__': main()
