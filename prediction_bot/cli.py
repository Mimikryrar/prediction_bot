from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd

from prediction_bot.data.pipeline import DataPipeline
from prediction_bot.model.evaluator import evaluate
from prediction_bot.model.features import FEATURE_COLS, build_feature_dataframe
from prediction_bot.model.model_store import DEFAULT_STORE_DIR
from prediction_bot.model.predictor import Predictor
from prediction_bot.model.splitter import SplitResult, temporal_split
from prediction_bot.model.trainer import build_labels, train


_DEFAULT_LOOKBACK_DAYS = 180


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _default_artifact_path(model_version: str) -> Path:
    return DEFAULT_STORE_DIR / f"{model_version}.joblib"


def _default_output_path(model_version: str, suffix: str) -> Path:
    return DEFAULT_STORE_DIR / f"{model_version}_{suffix}.json"


def _split_symbols(value: str) -> list[str]:
    return [s.strip() for s in value.split(",") if s.strip()]


def _write_json_output(payload: dict, output_path: Path | None) -> None:
    rendered = json.dumps(payload, indent=2)
    if output_path is None:
        print(rendered)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n")
    print(f"Saved output to {output_path}")


def _to_multiindex_split(split: SplitResult) -> SplitResult:
    return SplitResult(
        train=split.train.set_index(["symbol", "date"]),
        val=split.val.set_index(["symbol", "date"]),
        test=split.test.set_index(["symbol", "date"]),
        train_end_date=split.train_end_date,
        val_end_date=split.val_end_date,
    )


def _build_pipeline(args: argparse.Namespace) -> DataPipeline:
    return DataPipeline(stock_symbols=args.stock_symbols, crypto_symbols=args.crypto_symbols)


def _build_split_features(args: argparse.Namespace) -> tuple[list[str], pd.DataFrame, SplitResult]:
    symbols = _split_symbols(args.symbols)
    pipeline = _build_pipeline(args)
    records = pipeline.get_ohlcv(symbols, args.start_date, args.end_date)
    features = build_feature_dataframe(records).reset_index()
    if features.empty:
        raise ValueError("No features generated for the requested window")
    split = temporal_split(features, train_frac=args.train_frac, val_frac=args.val_frac)
    return symbols, features, split


def cmd_train(args: argparse.Namespace) -> int:
    symbols, _, split = _build_split_features(args)
    split_mi = _to_multiindex_split(split)
    artifact_path = args.artifact_path or _default_artifact_path(args.model_version)
    train(
        split_mi,
        horizon_days=args.horizon_days,
        model_version=args.model_version,
        symbol_universe=symbols,
        artifact_path=artifact_path,
    )
    payload = {
        "artifact_path": str(artifact_path),
        "train_rows": len(split.train),
        "val_rows": len(split.val),
        "test_rows": len(split.test),
        "train_end_date": split.train_end_date.date().isoformat(),
        "val_end_date": split.val_end_date.date().isoformat(),
    }
    _write_json_output(payload, args.output_path)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    symbols = _split_symbols(args.symbols)
    if len(symbols) != 1:
        raise ValueError("predict currently supports exactly one symbol; pass a single symbol")
    symbol = symbols[0]

    lookback_days = max(args.lookback_days, 30)
    start_date = date.fromordinal(args.prediction_date.toordinal() - lookback_days)
    pipeline = _build_pipeline(args)
    records = pipeline.get_ohlcv([symbol], start_date, args.prediction_date)
    features = build_feature_dataframe(records).reset_index()
    if features.empty:
        raise ValueError("No features available for prediction")

    latest = features[features["symbol"] == symbol].sort_values("date").iloc[-1]
    feature_row = {col: float(latest[col]) for col in FEATURE_COLS}
    predictor = Predictor(args.artifact_path)
    result = predictor.predict(symbol, feature_row, args.prediction_date)
    _write_json_output(result.model_dump(mode="json"), args.output_path)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    symbols, _, split = _build_split_features(args)
    split_mi = _to_multiindex_split(split)
    predictor = Predictor(args.artifact_path)
    partition = {"train": split_mi.train, "val": split_mi.val, "test": split_mi.test}[args.partition]
    labels = build_labels(partition, horizon_days=args.horizon_days)
    valid_mask = labels.notna()
    scored = partition.loc[valid_mask]
    if scored.empty:
        raise ValueError(f"No scorable rows found in {args.partition} partition")

    feature_rows = scored[FEATURE_COLS].to_dict(orient="records")
    pred_dates = [pd.Timestamp(d).date() for d in scored.index.get_level_values("date")]
    pred_symbols = list(scored.index.get_level_values("symbol"))
    results = predictor.predict_many(pred_symbols, feature_rows, pred_dates)
    paired = list(zip(results, labels.loc[valid_mask].astype(int).tolist()))
    metrics = evaluate(paired, calibration_bins=args.calibration_bins)

    payload = {
        "partition": args.partition,
        "symbols": symbols,
        "n_samples": metrics.n_samples,
        "accuracy": metrics.accuracy,
        "auc_roc": metrics.auc_roc,
        "brier_score": metrics.brier_score,
        "calibration_pred": metrics.calibration_pred,
        "calibration_true": metrics.calibration_true,
        "per_quarter": {
            q: {
                "accuracy": m.accuracy,
                "auc_roc": m.auc_roc,
                "brier_score": m.brier_score,
                "n_samples": m.n_samples,
            }
            for q, m in metrics.per_quarter.items()
        },
    }
    _write_json_output(payload, args.output_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prediction-bot", description="Train and run the prediction bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    default_stock_symbols = _split_symbols(os.getenv("STOCK_SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,NVDA"))
    default_crypto_symbols = _split_symbols(os.getenv("CRYPTO_SYMBOLS", "bitcoin,ethereum,solana,bnb,ripple"))

    train_parser = subparsers.add_parser("train", help="Train a model artifact")
    train_parser.add_argument("--symbols", required=True, help="Comma-separated symbols to train on")
    train_parser.add_argument("--start-date", type=_parse_date, required=True, help="Start date (YYYY-MM-DD)")
    train_parser.add_argument("--end-date", type=_parse_date, required=True, help="End date (YYYY-MM-DD)")
    train_parser.add_argument("--model-version", default="v1")
    train_parser.add_argument("--horizon-days", type=int, default=5)
    train_parser.add_argument("--train-frac", type=float, default=0.70)
    train_parser.add_argument("--val-frac", type=float, default=0.15)
    train_parser.add_argument("--artifact-path", type=Path, default=None)
    train_parser.add_argument("--output-path", type=Path, default=None)
    train_parser.add_argument("--stock-symbols", nargs="*", default=default_stock_symbols)
    train_parser.add_argument("--crypto-symbols", nargs="*", default=default_crypto_symbols)
    train_parser.set_defaults(func=cmd_train)

    predict_parser = subparsers.add_parser("predict", help="Run a single-symbol prediction")
    predict_parser.add_argument("--symbols", required=True, help="Single symbol to score")
    predict_parser.add_argument("--prediction-date", type=_parse_date, required=True, help="Prediction date (YYYY-MM-DD)")
    predict_parser.add_argument("--artifact-path", type=Path, required=True)
    predict_parser.add_argument("--lookback-days", type=int, default=_DEFAULT_LOOKBACK_DAYS)
    predict_parser.add_argument("--output-path", type=Path, default=None)
    predict_parser.add_argument("--stock-symbols", nargs="*", default=default_stock_symbols)
    predict_parser.add_argument("--crypto-symbols", nargs="*", default=default_crypto_symbols)
    predict_parser.set_defaults(func=cmd_predict)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model on a split partition")
    eval_parser.add_argument("--symbols", required=True, help="Comma-separated symbols to evaluate on")
    eval_parser.add_argument("--start-date", type=_parse_date, required=True, help="Start date (YYYY-MM-DD)")
    eval_parser.add_argument("--end-date", type=_parse_date, required=True, help="End date (YYYY-MM-DD)")
    eval_parser.add_argument("--artifact-path", type=Path, required=True)
    eval_parser.add_argument("--partition", choices=["train", "val", "test"], default="test")
    eval_parser.add_argument("--horizon-days", type=int, default=5)
    eval_parser.add_argument("--train-frac", type=float, default=0.70)
    eval_parser.add_argument("--val-frac", type=float, default=0.15)
    eval_parser.add_argument("--calibration-bins", type=int, default=10)
    eval_parser.add_argument("--output-path", type=Path, default=None)
    eval_parser.add_argument("--stock-symbols", nargs="*", default=default_stock_symbols)
    eval_parser.add_argument("--crypto-symbols", nargs="*", default=default_crypto_symbols)
    eval_parser.set_defaults(func=cmd_evaluate)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
