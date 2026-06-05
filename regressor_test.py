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
   
        X.append(inliers)

        label = int(pred_paths[0] in pos_paths)
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


def find_run_dir(vpr_dir: Path):
    for sub in sorted(vpr_dir.iterdir()):
        if sub.is_dir() and (sub / "preds").exists():
            return sub
    return None


def main(args):
    test_base = Path(args.test_base)
    model_dir = Path(args.model_dir)

    for vpr_dir in sorted(test_base.iterdir()):
        if not vpr_dir.is_dir():
            continue

        vpr_method = vpr_dir.name
        run_dir = find_run_dir(vpr_dir)
        if run_dir is None:
            continue
        matcher_dirs = []
        for d in run_dir.iterdir():
            if d.is_dir() and d.name.startswith("preds_"):
                matcher_dirs.append(d)

        for matcher_dir in sorted(matcher_dirs):
            matcher = matcher_dir.name.replace("preds_","")
            combo = f"{vpr_method}_{matcher}"
            model_path = model_dir / f"{combo}.pkl"
    X, y = build_dataset(str(run_dir / "preds"), str(matcher_dir))
    test_model(X, y, model_path)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
