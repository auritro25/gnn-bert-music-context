from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score

def genre_metrics(y_true,y_pred):
    return {'accuracy':float(accuracy_score(y_true,y_pred)),'macro_f1':float(f1_score(y_true,y_pred,average='macro',zero_division=0)),'weighted_f1':float(f1_score(y_true,y_pred,average='weighted',zero_division=0)),'macro_precision':float(precision_score(y_true,y_pred,average='macro',zero_division=0)),'macro_recall':float(recall_score(y_true,y_pred,average='macro',zero_division=0))}

def save_genre_artifacts(y_true,y_prob,labels,out_dir,track_ids=None):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); pred=y_prob.argmax(1); metrics=genre_metrics(y_true,pred)
    (out/'test_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    np.savez_compressed(out/'test_predictions.npz',y_true=y_true,y_prob=y_prob,track_ids=np.array(track_ids or []))
    cm=confusion_matrix(y_true,pred,labels=np.arange(len(labels)))
    plt.figure(figsize=(9,8)); plt.imshow(cm); plt.xticks(range(len(labels)),labels,rotation=90,fontsize=8); plt.yticks(range(len(labels)),labels,fontsize=8); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('FMA-small genre confusion matrix'); plt.tight_layout(); plt.savefig(out/'confusion_matrix.png',dpi=220); plt.close()
    return metrics
