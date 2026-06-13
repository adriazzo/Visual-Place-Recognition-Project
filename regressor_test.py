import os 
import argparse
import torch
from pathlib import Path
from glob import glob
from tqdm import tqdm
from util import read_file_preds
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import classification_report, roc_auc_score




def build_dataset(preds_dir, inliers_dir):
    txt_files = glob(os.path.join(preds_dir, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    X,y = [],[]

    for txt_file in tqdm(txt_files):
        q_num = Path(txt_file).stem
        torch_file = Path(inliers_dir) / f"{q_num}.torch"
        _, pred_paths, pos_paths = read_file_preds(txt_file)
        results = torch.load(torch_file, weights_only = False)
        inliers = results[0]["num_inliers"]
   
        for i, r in enumerate(results):
            X.append(r["num_inliers"])
            label = int(pred_paths[i] in pos_paths)
            y.append(label)

    return np.array(X), np.array(y)
 
def test_model(X, y, model_path):
    bundle = joblib.load(model_path)
    clf = bundle['clf']

    probs = clf.predict_proba(X)[:,1]
    preds = clf.predict(X)
    print(f"  AUC-ROC: {roc_auc_score(y, probs):.3f}")
    print(classification_report(y, preds))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_base", required = True, help="Cartella col dataset")
    parser.add_argument("--model_dir", required = True, help="Modello trainato")
    parser.add_argument("--vpr_list", required = True)
    parser.add_argument("--matchers_list", required = True)
    return parser.parse_args()
def main(args):
    test_base = Path(args.test_base)
    model_path = Path(args.model_path)
    vpr_list = list(args.vpr_list)
    matchers_list = list(args.matchers_list)
    
    for vpr_dir in test_base.iterdir() if vpr_dir.name in vpr_list else None:
      vpr_method = vpr_dir.name

      matcher_dirs = sorted(
          d for d in vpr_dir.iterdir()
          if d.is_dir() and d.name.startswith("preds_")
      )
      for matcher_dir in sorted(matcher_dirs) if matcher_dir.name.replace("preds_","") in matchers_list else None:
          matcher = matcher_dir.name.replace("preds_","")

          X, y = build_dataset(str(vpr_method / "preds"), str(matcher_dir))
          test_model(X, y, model_path)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
