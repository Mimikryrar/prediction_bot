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
from prediction_bot.model.regime import (
    attach_regime_features,
    compute_market_returns,
    fit_regime_hmm,
    predict_regimes,
    regime_feature_columns,
)
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


def _attach_regime_to_split(split_mi: SplitResult, hmm, n_states: int) -> tuple[SplitResult, list[str]]:
    """Attach lagged regime one-hot columns to each partition of a MultiIndex split."""
    regime_cols = regime_feature_columns(n_states)

    def _attach(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.assign(**{c: 0.0 for c in regime_cols})
        market_returns = compute_market_returns(df)
        states = predict_regimes(hmm, market_returns)
        return attach_regime_features(df, states, n_states)

    return (
        SplitResult(
            train=_attach(split_mi.train),
            val=_attach(split_mi.val),
            test=_attach(split_mi.test),
            train_end_date=split_mi.train_end_date,
            val_end_date=split_mi.val_end_date,
        ),
        regime_cols,
    )


def cmd_train(args: argparse.Namespace) -> int:
    symbols, _, split = _build_split_features(args)
    split_mi = _to_multiindex_split(split)
    artifact_path = args.artifact_path or _default_artifact_path(args.model_version)

    regime_hmm = None
    feature_names = list(FEATURE_COLS)
    if args.regime_states > 0:
        train_market_returns = compute_market_returns(split_mi.train)
        regime_hmm = fit_regime_hmm(train_market_returns, n_states=args.regime_states)
        split_mi, regime_cols = _attach_regime_to_split(split_mi, regime_hmm, args.regime_states)
        feature_names = list(FEATURE_COLS) + regime_cols

    train(
        split_mi,
        horizon_days=args.horizon_days,
        model_version=args.model_version,
        symbol_universe=symbols,
        artifact_path=artifact_path,
        feature_names=feature_names,
        regime_hmm=regime_hmm,
        regime_n_states=args.regime_states,
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

    predictor = Predictor(args.artifact_path)
    if predictor.regime_n_states > 0 and predictor.regime_hmm is not None:
        features_mi = features.set_index(["symbol", "date"])
        market_returns = compute_market_returns(features_mi)
        states = predict_regimes(predictor.regime_hmm, market_returns)
        features = attach_regime_features(features_mi, states, predictor.regime_n_states).reset_index()
        if features.empty:
            raise ValueError("No features available for prediction after regime attachment")

    latest = features[features["symbol"] == symbol].sort_values("date").iloc[-1]
    feature_row = {col: float(latest[col]) for col in predictor.feature_cols}
    result = predictor.predict(symbol, feature_row, args.prediction_date)
    _write_json_output(result.model_dump(mode="json"), args.output_path)
    return 0


def _score_partition(partition: pd.DataFrame, predictor: Predictor, horizon_days: int) -> tuple[list, list[int], pd.DataFrame]:
    if predictor.regime_n_states > 0 and predictor.regime_hmm is not None:
        market_returns = compute_market_returns(partition)
        states = predict_regimes(predictor.regime_hmm, market_returns)
        partition = attach_regime_features(partition, states, predictor.regime_n_states)

    labels = build_labels(partition, horizon_days=horizon_days)
    valid_mask = labels.notna()
    scored = partition.loc[valid_mask]
    if scored.empty:
        raise ValueError("No scorable rows found in selected partition")

    feature_rows = scored[list(predictor.feature_cols)].to_dict(orient="records")
    pred_dates = [pd.Timestamp(d).date() for d in scored.index.get_level_values("date")]
    pred_symbols = list(scored.index.get_level_values("symbol"))
    results = predictor.predict_many(pred_symbols, feature_rows, pred_dates)
    actuals = labels.loc[valid_mask].astype(int).tolist()
    return results, actuals, scored


def cmd_evaluate(args: argparse.Namespace) -> int:
    symbols, _, split = _build_split_features(args)
    split_mi = _to_multiindex_split(split)
    predictor = Predictor(args.artifact_path)
    partition = {"train": split_mi.train, "val": split_mi.val, "test": split_mi.test}[args.partition]
    results, actuals, _ = _score_partition(partition, predictor, args.horizon_days)
    metrics = evaluate(list(zip(results, actuals)), calibration_bins=args.calibration_bins)

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


def cmd_backtest(args: argparse.Namespace) -> int:
    symbols, _, split = _build_split_features(args)
    split_mi = _to_multiindex_split(split)
    predictor = Predictor(args.artifact_path)
    partition = {"train": split_mi.train, "val": split_mi.val, "test": split_mi.test}[args.partition]
    results, actuals, scored = _score_partition(partition, predictor, args.horizon_days)

    prediction_rows = []
    strategy_returns: list[float] = []
    equity = 1.0
    equity_curve: list[float] = []

    closes = scored["close"].to_numpy(dtype=float)
    future_closes = closes.copy()
    if isinstance(scored.index, pd.MultiIndex) and "symbol" in scored.index.names:
        future_close_series = scored["close"].groupby(level="symbol", sort=False).shift(-args.horizon_days)
        future_closes = future_close_series.to_numpy(dtype=float)

    for result, actual, (_, row), future_close in zip(results, actuals, scored.iterrows(), future_closes):
        p_up = result.p_up
        signal = 1 if p_up >= args.threshold else 0
        raw_return = 0.0 if pd.isna(future_close) else float(future_close / float(row["close"]) - 1.0)
        strategy_return = signal * raw_return
        strategy_returns.append(strategy_return)
        equity *= (1.0 + strategy_return)
        equity_curve.append(equity)
        prediction_rows.append({
            "symbol": result.symbol,
            "prediction_date": result.prediction_date.isoformat(),
            "p_up": p_up,
            "confidence": result.confidence,
            "actual": actual,
            "signal": signal,
            "close": float(row["close"]),
            "future_close": None if pd.isna(future_close) else float(future_close),
            "raw_return": raw_return,
            "strategy_return": strategy_return,
            "equity": equity,
        })

    pred_path = args.predictions_output_path
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_rows).to_csv(pred_path, index=False)

    equity_series = pd.Series(equity_curve, dtype=float)
    running_max = equity_series.cummax()
    drawdown = ((equity_series / running_max) - 1.0).min() if not equity_series.empty else 0.0
    trades = sum(row["signal"] for row in prediction_rows)
    wins = sum(1 for row in prediction_rows if row["signal"] == 1 and row["strategy_return"] > 0)
    hit_rate = float(wins / trades) if trades else 0.0
    summary = {
        "partition": args.partition,
        "symbols": symbols,
        "strategy": "long_only_threshold",
        "threshold": args.threshold,
        "n_predictions": len(prediction_rows),
        "n_trades": trades,
        "hit_rate": hit_rate,
        "cumulative_return": float(equity - 1.0),
        "max_drawdown": float(drawdown),
        "predictions_output_path": str(pred_path),
    }
    summary_path = args.output_path or pred_path.with_name("backtest_summary.json")
    _write_json_output(summary, summary_path)
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
    train_parser.add_argument(
        "--regime-states",
        type=int,
        default=0,
        help="Number of HMM regime states to fit on the train window and add as one-hot features. 0 disables.",
    )
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

    backtest_parser = subparsers.add_parser("backtest", help="Run a simple long-only threshold backtest")
    backtest_parser.add_argument("--symbols", required=True, help="Comma-separated symbols to backtest")
    backtest_parser.add_argument("--start-date", type=_parse_date, required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", type=_parse_date, required=True, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--artifact-path", type=Path, required=True)
    backtest_parser.add_argument("--partition", choices=["train", "val", "test"], default="test")
    backtest_parser.add_argument("--horizon-days", type=int, default=5)
    backtest_parser.add_argument("--train-frac", type=float, default=0.70)
    backtest_parser.add_argument("--val-frac", type=float, default=0.15)
    backtest_parser.add_argument("--threshold", type=float, default=0.55)
    backtest_parser.add_argument("--output-path", type=Path, default=None)
    backtest_parser.add_argument("--predictions-output-path", type=Path, required=True)
    backtest_parser.add_argument("--stock-symbols", nargs="*", default=default_stock_symbols)
    backtest_parser.add_argument("--crypto-symbols", nargs="*", default=default_crypto_symbols)
    backtest_parser.set_defaults(func=cmd_backtest)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
