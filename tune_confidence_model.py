import os
import torch
import numpy as np
import argparse
from glob import glob
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from tqdm import tqdm
import joblib
from util import read_file_predsimport os
import torch
import numpy as np
import argparse
from glob import glob
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from tqdm import tqdm
import joblib
from util import read_file_preds


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
 

def tune(X_train, y_train, X_val, y_val, model_path):
    if len(X_train.shape) == 1:
        X_train = X_train.reshape(-1, 1)
        X_val = X_val.reshape(-1, 1)

    X_combinato = np.vstack((X_train, X_val))
    y_combinato = np.concatenate((y_train, y_val))

    split_indices = np.zeros(X_combinato.shape[0])
    split_indices[:X_train.shape[0]] = -1
    pds = PredefinedSplit(test_fold=split_indices)

    param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                  'class_weight': ['balanced', None],
                  "max_iter": [100, 1000, 10000]
                  }

    clf = LogisticRegression(random_state=42)
    grid = GridSearchCV(
        estimator = clf,
        param_grid=param_grid,
        cv=pds,
        scoring = "roc_auc",
        n_jobs = -1
    )

    grid.fit(X_combinato, y_combinato)
    best_model = grid.best_estimator_
    best_auc = grid.best_score_
    print(f"  Migliori parametri: {grid.best_params_}")
    print(f"  Miglior AUC su Val: {best_auc:.4f}")
    joblib.dump(best_model, model_path)

    return best_model, grid.best_params_


def main(args):
    """
    Struttura attesa:

        train_base_dir/
        └── <vpr_method>/
              ├── preds/
              └── preds_<matcher>/

        val_base_dir/  (stesso layout, dataset diverso)
    """
    train_base = Path(args.train_base_dir)
    val_base   = Path(args.val_base_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    X_grande = np.array([])
    y_grande = np.array([])
    X_grande_val = np.array([])
    y_grande_val = np.array([])
    for vpr_dir in sorted(train_base.iterdir()):
        vpr_method = vpr_dir.name
        if vpr_dir.iterdir()

        matcher_dirs = [d for d in vpr_dir.iterdir()
                        if d.is_dir() and d.name.startswith("preds_")]

        for matcher_dir in sorted(matcher_dirs):
            matcher = matcher_dir.name.replace("preds_", "")
            combo   = f"{vpr_method}__{matcher}"

            val_vpr_dir = val_base / vpr_method

            preds_val   = val_vpr_dir / "preds"
            matched_val = val_vpr_dir / matcher_dir.name

            print(f"\n=== {combo} ===")
            X_train, y_train = build_dataset(str(run_train / "preds"), str(matcher_dir))
            X_grande = X_grande.vstack(X_train)
            y_grande = y_grande.vstack(y_train)
            X_val,   y_val   = build_dataset(str(preds_val),           str(matched_val))
            X_grande_val = X_grande_val.vstack(X_val)
            y_grande_val = y_grande_val.vstack(y_val)
            print(f"  Train: {len(X_grande)} | Val: {len(y_grande)}")
    model_path = model_dir / f"{train_base.name}.pkl"
    tune(X_grande, y_grande, X_grande_val, y_grande_val, model_path)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_base_dir", required=True)
    parser.add_argument("--val_base_dir",   required=True)
    parser.add_argument("--model_dir", required = True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


def find_run_dir(vpr_dir: Path):
    for sub in sorted(vpr_dir.iterdir()):
        if sub.is_dir() and (sub / "preds").exists():
            return sub
    return None

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
 

def tune(X_train, y_train, X_val, y_val, model_path):
    if len(X_train.shape) == 1:
        X_train = X_train.reshape(-1, 1)
        X_val = X_val.reshape(-1, 1)

    X_combinato = np.vstack((X_train, X_val))
    y_combinato = np.concatenate((y_train, y_val))

    split_indices = np.zeros(X_combinato.shape[0])
    split_indices[:X_train.shape[0]] = -1
    pds = PredefinedSplit(test_fold=split_indices)

    param_grid = {'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                  'class_weight': ['balanced', None],
                  "max_iter": [100, 1000, 10000]
                  }

    clf = LogisticRegression(random_state=42)
    grid = GridSearchCV(
        estimator = clf,
        param_grid=param_grid,
        cv=pds,
        scoring = "roc_auc",
        n_jobs = -1
    )

    grid.fit(X_combinato, y_combinato)
    best_model = grid.best_estimator_
    best_auc = grid.best_score_
    print(f"  Migliori parametri: {grid.best_params_}")
    print(f"  Miglior AUC su Val: {best_auc:.4f}")
    joblib.dump(best_model, model_path)

    return best_model, grid.best_params_


def main(args):
    """
    Struttura attesa:

        train_base_dir/
        └── <vpr_method>/
            └── <timestamp>/
                ├── preds/
                └── preds_<matcher>/

        val_base_dir/  (stesso layout, dataset diverso)
    """
    train_base = Path(args.train_base_dir)
    val_base   = Path(args.val_base_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for vpr_dir in sorted(train_base.iterdir()):
        vpr_method = vpr_dir.name

        run_train = find_run_dir(vpr_dir)
        if run_train is None:
            continue

        matcher_dirs = [d for d in run_train.iterdir()
                        if d.is_dir() and d.name.startswith("preds_")]

        for matcher_dir in sorted(matcher_dirs):
            matcher = matcher_dir.name.replace("preds_", "")
            combo   = f"{vpr_method}__{matcher}"

            val_vpr_dir = val_base / vpr_method
            run_val     = find_run_dir(val_vpr_dir) if val_vpr_dir.exists() else None
            if run_val is None:
                print(f"\n=== {combo} === SKIP (val set mancante)")
                continue

            preds_val   = run_val / "preds"
            matched_val = run_val / matcher_dir.name
            if not preds_val.exists() or not matched_val.exists():
                print(f"\n=== {combo} === SKIP (val set mancante)")
                continue

            print(f"\n=== {combo} ===")
            X_train, y_train = build_dataset(str(run_train / "preds"), str(matcher_dir))
            X_val,   y_val   = build_dataset(str(preds_val),           str(matched_val))
            print(f"  Train: {len(X_train)} | Val: {len(X_val)}")

            model_path = model_dir / f"{combo}.pkl"
            tune(X_train, y_train, X_val, y_val, model_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_base_dir", required=True)
    parser.add_argument("--val_base_dir",   required=True)
    parser.add_argument("--model_path", required = True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
