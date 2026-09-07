from pathlib import Path
CHECKS=[
('Task 1 masked MusicCaps BERT metrics',Path('results/musiccaps_masked/bert/test_metrics.json')),
('Task 2 FMA-small GNN metrics',Path('results/fma_small_genre/gnn/test_metrics.json')),
('Task 2 FMA-small CNN metrics',Path('results/fma_small_genre/cnn/test_metrics.json')),
('Task 3 FMA-medium BERT metrics',Path('results/fma_medium_tags/bert/test_metrics.json')),
('Task 3 FMA-medium GNN metrics',Path('results/fma_medium_tags/gnn/test_metrics.json')),
('Task 3 FMA-medium concat metrics',Path('results/fma_medium_tags/concat/test_metrics.json')),
('Task 3 FMA-medium fusion metrics',Path('results/fma_medium_tags/fusion/test_metrics.json')),
('Task 3 fusion t-SNE',Path('results/fma_medium_tags/fusion/tsne.png')),
('Task 3 case studies',Path('results/fma_medium_tags/fusion/case_studies/case_studies.json')),
('Task 4 retrieval metrics',Path('results/musiccaps_task4/contrastive/test_retrieval_metrics.json')),
('Task 4 10 retrieval examples',Path('results/musiccaps_task4/contrastive/retrieval_examples.csv')),
('Task 4 zero-shot tag metrics',Path('results/musiccaps_task4/contrastive/zero_shot_tags.json')),
('Strict comparison table',Path('results/strict_compliance_comparison.csv')),
('Demo notebook',Path('notebooks/demo_context.ipynb'))]
ok=0
for label,p in CHECKS:
    e=p.exists(); ok+=int(e); print(('PASS' if e else 'MISS')+f' | {label} | {p}')
print(f'\n{ok}/{len(CHECKS)} artifact checks currently pass.')
if ok!=len(CHECKS): print('Missing items require running the corresponding experiment; code support is present.')
