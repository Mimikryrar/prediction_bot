from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

import runpy

from prediction_bot.cli import build_parser, cmd_evaluate, cmd_predict, cmd_train
from prediction_bot.model.splitter import SplitResult
from prediction_bot.shared.schemas import PredictionResult


class DummyPipeline:
    def __init__(self, records):
        self._records = records

    def get_ohlcv(self, symbols, start, end):
        return self._records


class DummyPredictor:
    def __init__(self, artifact_path):
        self.artifact_path = artifact_path

    def predict(self, symbol, feature_row, prediction_date):
        return PredictionResult(
            symbol=symbol,
            prediction_date=prediction_date,
            horizon_days=5,
            p_up=0.75,
            confidence=0.75,
            model_version="cli_test",
        )

    def predict_many(self, symbols, feature_rows, prediction_dates):
        return [self.predict(symbol, feature_row, prediction_date) for symbol, feature_row, prediction_date in zip(symbols, feature_rows, prediction_dates)]


def test_parser_builds_subcommands():
    parser = build_parser()
    args = parser.parse_args([
        "train",
        "--symbols",
        "AAPL",
        "--start-date",
        "2023-01-01",
        "--end-date",
        "2023-06-01",
    ])
    assert args.command == "train"


def test_module_entrypoint_smoke(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prediction_bot", "--help"])
    try:
        runpy.run_module("prediction_bot", run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0


def test_cmd_train_runs(monkeypatch, tmp_path, capsys):
    records = [object()]
    monkeypatch.setattr("prediction_bot.cli.DataPipeline", lambda **kwargs: DummyPipeline(records))
    monkeypatch.setattr(
        "prediction_bot.cli.build_feature_dataframe",
        lambda records: pd.DataFrame([
            {"symbol": "AAPL", "date": pd.Timestamp("2023-01-01"), "close": 100.0, "log_return_1d": 0.1, "momentum_5d": 0.2, "momentum_20d": 0.3, "volatility_20d": 0.4, "sma_cross_5_20": 1.0, "volume_zscore": 0.5},
            {"symbol": "AAPL", "date": pd.Timestamp("2023-01-02"), "close": 101.0, "log_return_1d": 0.1, "momentum_5d": 0.2, "momentum_20d": 0.3, "volatility_20d": 0.4, "sma_cross_5_20": 1.0, "volume_zscore": 0.5},
            {"symbol": "AAPL", "date": pd.Timestamp("2023-01-03"), "close": 102.0, "log_return_1d": 0.1, "momentum_5d": 0.2, "momentum_20d": 0.3, "volatility_20d": 0.4, "sma_cross_5_20": 1.0, "volume_zscore": 0.5},
        ])
    )
    monkeypatch.setattr(
        "prediction_bot.cli.temporal_split",
        lambda df, train_frac, val_frac: SplitResult(df.iloc[:1], df.iloc[1:2], df.iloc[2:], pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")),
    )
    monkeypatch.setattr("prediction_bot.cli.train", lambda *args, **kwargs: None)

    args = build_parser().parse_args([
        "train",
        "--symbols", "AAPL",
        "--start-date", "2023-01-01",
        "--end-date", "2023-06-01",
        "--artifact-path", str(tmp_path / "model.joblib"),
    ])
    assert cmd_train(args) == 0
    assert "artifact_path" in capsys.readouterr().out


def test_cmd_predict_runs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("prediction_bot.cli.DataPipeline", lambda **kwargs: DummyPipeline([object()]))
    monkeypatch.setattr(
        "prediction_bot.cli.build_feature_dataframe",
        lambda records: pd.DataFrame([
            {"symbol": "AAPL", "date": pd.Timestamp("2023-01-03"), "close": 102.0, "log_return_1d": 0.1, "momentum_5d": 0.2, "momentum_20d": 0.3, "volatility_20d": 0.4, "sma_cross_5_20": 1.0, "volume_zscore": 0.5},
        ])
    )
    monkeypatch.setattr("prediction_bot.cli.Predictor", DummyPredictor)

    artifact = tmp_path / "model.joblib"
    artifact.write_text("stub")
    args = build_parser().parse_args([
        "predict",
        "--symbols", "AAPL",
        "--prediction-date", "2023-06-01",
        "--artifact-path", str(artifact),
    ])
    assert cmd_predict(args) == 0
    assert '"p_up": 0.75' in capsys.readouterr().out


def test_cmd_predict_writes_output_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("prediction_bot.cli.DataPipeline", lambda **kwargs: DummyPipeline([object()]))
    monkeypatch.setattr(
        "prediction_bot.cli.build_feature_dataframe",
        lambda records: pd.DataFrame([
            {"symbol": "AAPL", "date": pd.Timestamp("2023-01-03"), "close": 102.0, "log_return_1d": 0.1, "momentum_5d": 0.2, "momentum_20d": 0.3, "volatility_20d": 0.4, "sma_cross_5_20": 1.0, "volume_zscore": 0.5},
        ])
    )
    monkeypatch.setattr("prediction_bot.cli.Predictor", DummyPredictor)

    artifact = tmp_path / "model.joblib"
    artifact.write_text("stub")
    output = tmp_path / "prediction.json"
    args = build_parser().parse_args([
        "predict",
        "--symbols", "AAPL",
        "--prediction-date", "2023-06-01",
        "--artifact-path", str(artifact),
        "--output-path", str(output),
    ])
    assert cmd_predict(args) == 0
    assert output.exists()
    assert "Saved output to" in capsys.readouterr().out


def test_cmd_evaluate_runs(monkeypatch, tmp_path, capsys):
    df = pd.DataFrame([
        {"symbol": "AAPL", "date": pd.Timestamp("2023-01-01"), "close": 100.0, "log_return_1d": 0.1, "momentum_5d": 0.2, "momentum_20d": 0.3, "volatility_20d": 0.4, "sma_cross_5_20": 1.0, "volume_zscore": 0.5},
        {"symbol": "AAPL", "date": pd.Timestamp("2023-01-02"), "close": 101.0, "log_return_1d": 0.1, "momentum_5d": 0.2, "momentum_20d": 0.3, "volatility_20d": 0.4, "sma_cross_5_20": 1.0, "volume_zscore": 0.5},
        {"symbol": "AAPL", "date": pd.Timestamp("2023-01-03"), "close": 102.0, "log_return_1d": 0.1, "momentum_5d": 0.2, "momentum_20d": 0.3, "volatility_20d": 0.4, "sma_cross_5_20": 1.0, "volume_zscore": 0.5},
    ])
    monkeypatch.setattr("prediction_bot.cli.DataPipeline", lambda **kwargs: DummyPipeline([object()]))
    monkeypatch.setattr("prediction_bot.cli.build_feature_dataframe", lambda records: df)
    monkeypatch.setattr(
        "prediction_bot.cli.temporal_split",
        lambda df, train_frac, val_frac: SplitResult(df.iloc[:3], df.iloc[:0], df.iloc[:0], pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-03")),
    )
    monkeypatch.setattr("prediction_bot.cli.Predictor", DummyPredictor)
    monkeypatch.setattr("prediction_bot.cli.build_labels", lambda part, horizon_days: pd.Series([0.0, 1.0, float('nan')], index=part.index))

    artifact = tmp_path / "model.joblib"
    artifact.write_text("stub")
    args = build_parser().parse_args([
        "evaluate",
        "--symbols", "AAPL",
        "--start-date", "2023-01-01",
        "--end-date", "2023-06-01",
        "--artifact-path", str(artifact),
        "--partition", "train",
    ])
    assert cmd_evaluate(args) == 0
    out = capsys.readouterr().out
    assert '"accuracy"' in out
