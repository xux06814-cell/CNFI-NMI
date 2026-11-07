"""Five-fold outer evaluation; inner tuning is not fully nested (see docs)."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from cnfi_nmi import CNFINMISelector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--positive-label", required=True,
                        help="Exact target value counted as class 1 for recall/F1/AUC.")
    parser.add_argument("--sep", default=",")
    parser.add_argument("--drop", nargs="*", default=[])
    parser.add_argument("--categorical", nargs="*", default=[],
                        help="Integer-coded categories that must not be quantile-binned.")
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument(
        "--cnfi-features", type=int, default=None,
        help="Compatibility option: must equal the total input feature count; "
             "both branches evaluate all inputs. --features controls final retention.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def encode_without_leakage(train: pd.DataFrame, test: pd.DataFrame):
    train = train.copy()
    test = test.copy()
    for name in train.columns:
        if pd.api.types.is_numeric_dtype(train[name]):
            median = train[name].median()
            if pd.isna(median):
                median = 0.0
            train[name] = train[name].fillna(median)
            test[name] = test[name].fillna(median)
        else:
            values = train[name].astype("string").fillna("<NA>")
            categories = {value: index for index, value in enumerate(pd.unique(values))}
            train[name] = values.map(categories).astype(int)
            test[name] = (
                test[name]
                .astype("string")
                .fillna("<NA>")
                .map(categories)
                .fillna(-1)
                .astype(int)
            )
    return train, test


def specificity(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if tn + fp else 0.0


def main():
    args = parse_args()
    frame = pd.read_csv(args.csv, sep=args.sep, low_memory=False)
    if args.target not in frame:
        raise ValueError(f"Target column not found: {args.target}")
    y = frame.pop(args.target)
    if y.isna().any():
        raise ValueError("The target contains missing values.")
    labels = y.astype(str)
    if labels.nunique() != 2 or args.positive_label not in labels.unique():
        raise ValueError("Expected two labels including --positive-label.")
    y = (labels == args.positive_label).astype(int)
    frame = frame.drop(columns=args.drop)
    sources = {"Emergency service(s)", "Emergency services"} & set(frame.columns)
    if sources:
        raise ValueError(f"Exclude target source columns with --drop: {sorted(sources)}")
    for name in args.categorical:
        frame[name] = frame[name].astype("string")
    if y.value_counts().min() < 5:
        raise ValueError("Each class must contain at least five samples.")
    if args.threads < 1:
        raise ValueError("--threads must be positive.")

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    folds = []
    for fold, (train_index, test_index) in enumerate(outer.split(frame, y), start=1):
        X_train, X_test = frame.iloc[train_index], frame.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        selector = CNFINMISelector(
            n_features_to_select=args.features,
            cnfi_features=args.cnfi_features,
            alpha=0.5,
            min_support=0.01,
            min_confidence=0.1,
            n_bins=5,
            kappa=1.0,
            theta=0.0,
            graph_mode="normalized",
        )
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        # Mine original categories and compute NMI on original observations.
        # Numeric encoding is only for the downstream classifier.
        X_train_selected, X_test_selected = encode_without_leakage(
            X_train_selected, X_test_selected
        )

        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed + fold)
        search = GridSearchCV(
            LGBMClassifier(random_state=args.seed, verbosity=-1,
                           n_jobs=args.threads),
            {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "n_estimators": [200],
            },
            scoring="roc_auc",
            cv=inner,
            n_jobs=1,
        )
        search.fit(X_train_selected, y_train)
        prediction = search.predict(X_test_selected)
        probability = search.predict_proba(X_test_selected)[:, 1]
        folds.append(
            {
                "fold": fold,
                "train_indices": train_index.tolist(),
                "test_indices": test_index.tolist(),
                "selected_features": selector.selected_features_,
                "best_parameters": search.best_params_,
                "accuracy": accuracy_score(y_test, prediction),
                "precision": precision_score(y_test, prediction, zero_division=0),
                "recall": recall_score(y_test, prediction, zero_division=0),
                "f1": f1_score(y_test, prediction, zero_division=0),
                "specificity": specificity(y_test, prediction),
                "auc": roc_auc_score(y_test, probability),
            }
        )

    metrics = ["accuracy", "precision", "recall", "f1", "specificity", "auc"]
    summary = {
        metric: {
            "mean": float(np.mean([fold[metric] for fold in folds])),
            "sample_std": float(np.std([fold[metric] for fold in folds], ddof=1)),
        }
        for metric in metrics
    }
    result = {
        "protocol": "5-fold outer holdout; inner tuning is not fully nested",
        "graph_mode": "normalized",
        "input_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "target": args.target,
        "positive_label": args.positive_label,
        "seed": args.seed,
        "features": args.features,
        "dropped": args.drop,
        "categorical": args.categorical,
        "selector_parameters": selector.get_params(),
        "versions": {"python": platform.python_version(), **{
            package: version(package) for package in
            ["cnfi-nmi", "numpy", "pandas", "scikit-learn", "networkx", "mlxtend", "lightgbm"]
        }},
        "folds": folds,
        "summary": summary,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
