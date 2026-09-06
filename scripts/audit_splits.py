from __future__ import annotations
import argparse
import pandas as pd

p=argparse.ArgumentParser(); p.add_argument('--csv',required=True); p.add_argument('--split-col',default='split'); p.add_argument('--group-col',default='artist_id'); a=p.parse_args()
df=pd.read_csv(a.csv)
req={'train','val','test'}; actual=set(df[a.split_col].astype(str).str.lower().unique())
if req-actual: raise SystemExit(f"FAIL: missing split(s): {sorted(req-actual)}")
print(df[a.split_col].value_counts())
if a.group_col not in df.columns:
    print(f"Group leakage audit: SKIPPED ({a.group_col!r} unavailable)"); raise SystemExit(0)
clean=df.dropna(subset=[a.group_col]).copy(); clean=clean[clean[a.group_col].astype(str).str.strip()!='']
g={s:set(clean.loc[clean[a.split_col].astype(str).str.lower()==s,a.group_col].astype(str)) for s in ['train','val','test']}
failed=False
for x,y in [('train','val'),('train','test'),('val','test')]:
    n=len(g[x]&g[y]); print(f"{x} vs {y}: {n} overlapping {a.group_col}(s)"); failed|=bool(n)
if failed: raise SystemExit('FAIL: group leakage detected.')
print('Group leakage audit: PASS')
