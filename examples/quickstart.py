"""Fit CNFI-NMI on a user-supplied processed CSV."""
import argparse
from pathlib import Path
import pandas as pd
from cnfi_nmi import CNFINMISelector

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--sep", default=",")
    parser.add_argument("--drop", nargs="*", default=[])
    parser.add_argument("--categorical", nargs="*", default=[])
    args = parser.parse_args()
    frame = pd.read_csv(args.csv, sep=args.sep, low_memory=False)
    if args.target not in frame:
        raise ValueError(f"Target column not found: {args.target}")
    y = frame.pop(args.target)
    if y.isna().any():
        raise ValueError("Target contains missing values.")
    X = frame.drop(columns=args.drop)
    sources = {"Emergency service(s)", "Emergency services"} & set(X.columns)
    if sources:
        raise ValueError(f"Exclude target source columns with --drop: {sorted(sources)}")
    for name in args.categorical:
        X[name] = X[name].astype("string")
    selector = CNFINMISelector(n_features_to_select=args.features).fit(X, y)
    print("Selected features:", selector.selected_features_)
    print("Fused scores:", selector.feature_scores_)
    print("Transformed shape:", selector.transform(X).shape)

if __name__ == "__main__":
    main()
