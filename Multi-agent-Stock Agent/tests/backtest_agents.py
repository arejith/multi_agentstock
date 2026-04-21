import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Agents.analysis_team.decision import DecisionLayer
from models.risk.risk_model import RiskModel
from models.sentiment.sentiment_model import SentimentModel
from models.Transformer.predict import TransformerPredictor
from tools.data_sources.stock_data import FIXED_COMPANY_ORDER, SECTOR_REPLACEMENT_MAP


CANONICAL_TICKER_SECTOR_PAIRS = [
    ("AAPL", "technology"),
    ("AMZN", "consumer_discretionary"),
    ("CAT", "industrials"),
    ("DOW", "materials"),
    ("GOOGL", "communication_services"),
    ("JNJ", "healthcare"),
    ("JPM", "financials"),
    ("MSFT", "software"),
    ("WMT", "consumer_staples"),
    ("XOM", "energy"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Backtest agent outputs on historical data. "
            "Transformer accuracy uses Yahoo Finance price history. "
            "Sentiment/risk/final decision accuracy can also be evaluated from an optional snapshot file."
        )
    )
    parser.add_argument(
        "--period",
        default="2y",
        help="Yahoo Finance period used for price backtesting. Default: 2y",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=30,
        help="Sequence length passed to the transformer model. Default: 30",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=60,
        help="Minimum number of return rows before a prediction is scored. Default: 60",
    )
    parser.add_argument(
        "--tickers",
        default=",".join(ticker for ticker, _ in CANONICAL_TICKER_SECTOR_PAIRS),
        help="Comma-separated tickers to evaluate from the supported model universe.",
    )
    parser.add_argument(
        "--snapshot-file",
        help=(
            "Optional JSON/JSONL file with dated news/fundamentals snapshots. "
            "If expected labels are included, sentiment/risk/final decision accuracy is computed too."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(ROOT / ".runtime_cache" / "yfinance"),
        help="Workspace-local yfinance cache directory.",
    )
    return parser.parse_args()


def normalize_returns(returns_frame: pd.DataFrame):
    return (returns_frame - returns_frame.mean()) / (returns_frame.std() + 1e-8)


def classify_signal(value: float):
    if value > 0.01:
        return "BUY"
    if value < -0.01:
        return "SELL"
    return "HOLD"


def load_snapshot_records(path: Path):
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            records = payload.get("records")
            if isinstance(records, list):
                return records
        raise ValueError("JSON snapshot file must contain either a list or {'records': [...]} ")

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records


def build_close_frame(period: str):
    downloaded = yf.download(
        FIXED_COMPANY_ORDER,
        period=period,
        auto_adjust=True,
        progress=False,
    )
    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no market data for the supported ticker universe")

    closes = downloaded["Close"] if isinstance(downloaded.columns, pd.MultiIndex) else downloaded
    closes = closes[FIXED_COMPANY_ORDER].ffill().dropna()
    if closes.empty:
        raise RuntimeError("No close-price frame could be built from Yahoo Finance data")
    return closes


def run_transformer_backtest(close_prices: pd.DataFrame, selected_pairs: list[tuple[str, str]], seq_len: int, min_history: int):
    returns = close_prices.pct_change().dropna()
    predictor = TransformerPredictor()

    rows = []
    confusion = Counter()
    per_ticker = defaultdict(
        lambda: {
            "samples": 0,
            "signal_matches": 0,
            "direction_matches": 0,
            "mae": [],
            "pred_counts": Counter(),
            "actual_counts": Counter(),
        }
    )

    for ticker, sector in selected_pairs:
        replaced_column = SECTOR_REPLACEMENT_MAP[sector]
        for index in range(min_history - 1, len(returns) - 1):
            history = returns.iloc[: index + 1]
            normalized = normalize_returns(history).tail(seq_len)
            if len(normalized) < seq_len:
                continue

            prepared_input = {
                "ticker": ticker,
                "sector": sector,
                "source_timestamp": str(returns.index[index]),
                "columns": list(normalized.columns),
                "replaced_column": replaced_column,
                "sequence_length": seq_len,
                "normalized_returns": normalized.values.tolist(),
            }

            prediction = predictor.predict_prepared(prepared_input)
            predicted_value = float(prediction["prediction"])
            predicted_signal = prediction["signal"]
            actual_return = float(returns.iloc[index + 1][ticker])
            actual_signal = classify_signal(actual_return)

            predicted_direction = "UP" if predicted_value > 0 else "DOWN" if predicted_value < 0 else "FLAT"
            actual_direction = "UP" if actual_return > 0 else "DOWN" if actual_return < 0 else "FLAT"

            row = {
                "ticker": ticker,
                "sector": sector,
                "as_of": returns.index[index].date().isoformat(),
                "target_date": returns.index[index + 1].date().isoformat(),
                "predicted_value": predicted_value,
                "predicted_signal": predicted_signal,
                "actual_return": actual_return,
                "actual_signal": actual_signal,
                "signal_match": predicted_signal == actual_signal,
                "direction_match": predicted_direction == actual_direction,
                "abs_error": abs(predicted_value - actual_return),
            }
            rows.append(row)

            confusion[(predicted_signal, actual_signal)] += 1
            stats = per_ticker[ticker]
            stats["samples"] += 1
            stats["signal_matches"] += int(row["signal_match"])
            stats["direction_matches"] += int(row["direction_match"])
            stats["mae"].append(row["abs_error"])
            stats["pred_counts"][predicted_signal] += 1
            stats["actual_counts"][actual_signal] += 1

    if not rows:
        raise RuntimeError("Transformer backtest produced no scored samples")

    return {
        "rows": rows,
        "confusion": confusion,
        "per_ticker": per_ticker,
        "returns": returns,
    }


def evaluate_snapshot_records(records: list[dict], returns: pd.DataFrame):
    if not records:
        return None

    sentiment_model = SentimentModel()
    risk_model = RiskModel()
    decision_layer = DecisionLayer()

    metrics = {
        "scored_records": 0,
        "sentiment_correct": 0,
        "risk_correct": 0,
        "recommendation_correct": 0,
        "transformer_signal_correct": 0,
        "transformer_direction_correct": 0,
    }
    missing_expectations = Counter()
    examples = []

    for record in records:
        ticker = record["ticker"]
        sector = record["sector"]
        as_of = pd.Timestamp(record["as_of"])
        news_items = record.get("news", [])
        fundamentals = dict(record.get("fundamentals") or {})
        expected = dict(record.get("expected") or {})

        texts = []
        for item in news_items:
            if isinstance(item, dict):
                text = (item.get("text") or item.get("summary") or item.get("title") or "").strip()
            else:
                text = str(item).strip()
            if text:
                texts.append(text)

        sentiment = sentiment_model.analyze_batch(texts)
        fundamentals.setdefault("ticker", ticker)
        risk = risk_model.analyze(fundamentals)

        transformer_signal = None
        transformer_prediction = None
        transformer_direction_match = None
        transformer_signal_match = None

        if ticker in returns.columns and as_of in returns.index:
            index = returns.index.get_loc(as_of)
            if isinstance(index, slice):
                index = index.start
            if isinstance(index, int) and index >= 29 and index < len(returns) - 1:
                history = returns.iloc[: index + 1]
                normalized = normalize_returns(history).tail(30)
                prepared_input = {
                    "ticker": ticker,
                    "sector": sector,
                    "source_timestamp": str(as_of),
                    "columns": list(normalized.columns),
                    "replaced_column": SECTOR_REPLACEMENT_MAP[sector],
                    "sequence_length": 30,
                    "normalized_returns": normalized.values.tolist(),
                }
                prediction = TransformerPredictor().predict_prepared(prepared_input)
                transformer_prediction = float(prediction["prediction"])
                transformer_signal = prediction["signal"]
                actual_return = float(returns.iloc[index + 1][ticker])
                actual_signal = classify_signal(actual_return)
                predicted_direction = "UP" if transformer_prediction > 0 else "DOWN" if transformer_prediction < 0 else "FLAT"
                actual_direction = "UP" if actual_return > 0 else "DOWN" if actual_return < 0 else "FLAT"
                transformer_signal_match = transformer_signal == actual_signal
                transformer_direction_match = predicted_direction == actual_direction

        analysis_phase = {
            "news_analysis": {"sentiment": sentiment},
            "risk_analysis": {"risk": risk},
            "transformer_analysis": {
                "transformer": {
                    "ticker": ticker,
                    "prediction": transformer_prediction,
                    "signal": transformer_signal or "UNKNOWN",
                }
            },
        }
        final_report = decision_layer.decide(ticker=ticker, sector=sector, analysis_phase=analysis_phase)

        metrics["scored_records"] += 1

        sentiment_score = sentiment.get("score", 0.0)
        sentiment_label = "positive" if sentiment_score > 0.2 else "negative" if sentiment_score < -0.2 else "neutral"

        if "sentiment_label" in expected:
            metrics["sentiment_correct"] += int(sentiment_label == expected["sentiment_label"])
        else:
            missing_expectations["sentiment_label"] += 1

        if "risk_level" in expected:
            metrics["risk_correct"] += int(risk["risk_level"] == expected["risk_level"])
        else:
            missing_expectations["risk_level"] += 1

        if "recommendation" in expected:
            metrics["recommendation_correct"] += int(final_report["recommendation"] == expected["recommendation"])
        else:
            missing_expectations["recommendation"] += 1

        if transformer_signal_match is not None:
            metrics["transformer_signal_correct"] += int(transformer_signal_match)
            metrics["transformer_direction_correct"] += int(transformer_direction_match)
        else:
            missing_expectations["transformer_market_alignment"] += 1

        if len(examples) < 3:
            examples.append(
                {
                    "ticker": ticker,
                    "as_of": str(as_of.date()),
                    "sentiment": sentiment_label,
                    "risk": risk["risk_level"],
                    "recommendation": final_report["recommendation"],
                    "transformer_signal": transformer_signal or "UNKNOWN",
                }
            )

    return {
        "metrics": metrics,
        "missing_expectations": missing_expectations,
        "examples": examples,
    }


def print_transformer_report(result: dict):
    rows = result["rows"]
    confusion = result["confusion"]
    per_ticker = result["per_ticker"]

    signal_accuracy = mean(int(row["signal_match"]) for row in rows)
    directional_accuracy = mean(int(row["direction_match"]) for row in rows)
    mae = mean(row["abs_error"] for row in rows)

    print("=== TRANSFORMER BACKTEST ===")
    print(f"Samples: {len(rows)}")
    print(
        "Signal accuracy (BUY/SELL/HOLD vs next-day move with +/-1% thresholds): "
        f"{signal_accuracy:.4f}"
    )
    print(f"Directional accuracy (prediction sign vs next-day return sign): {directional_accuracy:.4f}")
    print(f"Mean absolute error vs next-day raw return: {mae:.6f}")
    print()

    print("Confusion counts:")
    for predicted_signal in ["BUY", "HOLD", "SELL"]:
        for actual_signal in ["BUY", "HOLD", "SELL"]:
            print(f"  {predicted_signal:>4} -> {actual_signal:<4}: {confusion[(predicted_signal, actual_signal)]}")
    print()

    print("Per ticker:")
    for ticker in sorted(per_ticker):
        stats = per_ticker[ticker]
        print(
            f"  {ticker}: samples={stats['samples']}, "
            f"signal_acc={stats['signal_matches'] / stats['samples']:.4f}, "
            f"direction_acc={stats['direction_matches'] / stats['samples']:.4f}, "
            f"mae={mean(stats['mae']):.6f}, "
            f"pred(B/H/S)="
            f"{stats['pred_counts']['BUY']}/{stats['pred_counts']['HOLD']}/{stats['pred_counts']['SELL']}, "
            f"actual(B/H/S)="
            f"{stats['actual_counts']['BUY']}/{stats['actual_counts']['HOLD']}/{stats['actual_counts']['SELL']}"
        )
    print()


def safe_accuracy(correct: int, total: int):
    if total == 0:
        return None
    return correct / total


def print_snapshot_report(result: dict | None):
    if result is None:
        print("=== SNAPSHOT EVALUATION ===")
        print("No snapshot file provided, so sentiment/risk/final-decision accuracy was not scored.")
        print()
        return

    metrics = result["metrics"]
    missing = result["missing_expectations"]

    sentiment_total = metrics["scored_records"] - missing["sentiment_label"]
    risk_total = metrics["scored_records"] - missing["risk_level"]
    recommendation_total = metrics["scored_records"] - missing["recommendation"]
    transformer_total = metrics["scored_records"] - missing["transformer_market_alignment"]

    print("=== SNAPSHOT EVALUATION ===")
    print(f"Snapshot records processed: {metrics['scored_records']}")

    sentiment_accuracy = safe_accuracy(metrics["sentiment_correct"], sentiment_total)
    risk_accuracy = safe_accuracy(metrics["risk_correct"], risk_total)
    recommendation_accuracy = safe_accuracy(metrics["recommendation_correct"], recommendation_total)
    transformer_signal_accuracy = safe_accuracy(metrics["transformer_signal_correct"], transformer_total)
    transformer_direction_accuracy = safe_accuracy(metrics["transformer_direction_correct"], transformer_total)

    print(
        "Sentiment accuracy: "
        + (f"{sentiment_accuracy:.4f} on {sentiment_total} labeled records" if sentiment_accuracy is not None else "not scored")
    )
    print(
        "Risk accuracy: "
        + (f"{risk_accuracy:.4f} on {risk_total} labeled records" if risk_accuracy is not None else "not scored")
    )
    print(
        "Recommendation accuracy: "
        + (
            f"{recommendation_accuracy:.4f} on {recommendation_total} labeled records"
            if recommendation_accuracy is not None
            else "not scored"
        )
    )
    print(
        "Transformer market alignment from snapshots: "
        + (
            f"signal={transformer_signal_accuracy:.4f}, direction={transformer_direction_accuracy:.4f} on {transformer_total} records"
            if transformer_signal_accuracy is not None
            else "not scored"
        )
    )

    if missing:
        print("Missing labels/targets:")
        for key, count in sorted(missing.items()):
            print(f"  {key}: {count}")

    if result["examples"]:
        print("Example snapshot outputs:")
        for example in result["examples"]:
            print(
                f"  {example['ticker']} @ {example['as_of']}: "
                f"sentiment={example['sentiment']}, "
                f"risk={example['risk']}, "
                f"recommendation={example['recommendation']}, "
                f"transformer={example['transformer_signal']}"
            )
    print()


def resolve_selected_pairs(raw_tickers: str):
    requested = {ticker.strip().upper() for ticker in raw_tickers.split(",") if ticker.strip()}
    pairs = [(ticker, sector) for ticker, sector in CANONICAL_TICKER_SECTOR_PAIRS if ticker in requested]
    if not pairs:
        supported = ", ".join(ticker for ticker, _ in CANONICAL_TICKER_SECTOR_PAIRS)
        raise ValueError(f"No supported tickers selected. Supported universe: {supported}")
    return pairs


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))

    selected_pairs = resolve_selected_pairs(args.tickers)
    closes = build_close_frame(args.period)
    transformer_result = run_transformer_backtest(
        close_prices=closes,
        selected_pairs=selected_pairs,
        seq_len=args.seq_len,
        min_history=args.min_history,
    )

    snapshot_result = None
    if args.snapshot_file:
        snapshot_path = Path(args.snapshot_file)
        records = load_snapshot_records(snapshot_path)
        snapshot_result = evaluate_snapshot_records(records, transformer_result["returns"])

    print(f"Date range: {closes.index.min().date()} -> {closes.index.max().date()}")
    print(f"Tickers: {', '.join(ticker for ticker, _ in selected_pairs)}")
    print()
    print_transformer_report(transformer_result)
    print_snapshot_report(snapshot_result)


if __name__ == "__main__":
    main()
