import pandas as pd
import torch
import yfinance as yf


FIXED_COMPANY_ORDER = [
    "AAPL", "AMZN", "CAT", "DOW", "GOOGL",
    "JNJ", "JPM", "MSFT", "WMT", "XOM",
]

SECTOR_REPLACEMENT_MAP = {
    "technology": "AAPL",
    "consumer_discretionary": "AMZN",
    "industrials": "CAT",
    "materials": "DOW",
    "communication_services": "GOOGL",
    "healthcare": "JNJ",
    "financials": "JPM",
    "software": "MSFT",
    "consumer_staples": "WMT",
    "energy": "XOM",
}


def _download_close_prices(symbols, period="6mo"):
    data = yf.download(
        symbols,
        period=period,
        auto_adjust=True,
        progress=False,
    )["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    return data.ffill().dropna()


def _normalize_returns(returns_frame):
    return (returns_frame - returns_frame.mean()) / (returns_frame.std() + 1e-8)


def _index_value_to_iso(timestamp_value):
    if timestamp_value is None:
        raise ValueError("Missing market timestamp for transformer input")

    if isinstance(timestamp_value, pd.Timestamp):
        return timestamp_value.isoformat()

    return pd.Timestamp(timestamp_value).isoformat()


def build_transformer_input(ticker: str, sector: str, seq_len: int = 30):
    base_prices = _download_close_prices(FIXED_COMPANY_ORDER)
    returns = base_prices.pct_change().dropna()

    if len(returns) < seq_len:
        raise ValueError("Not enough base market history to build transformer input")

    new_prices = _download_close_prices(ticker)
    new_returns = new_prices.pct_change().dropna()
    if ticker in new_returns.columns:
        replacement_series = new_returns[ticker]
    else:
        replacement_series = new_returns.squeeze()

    normalized_sector = sector.strip().lower()
    replace_column = SECTOR_REPLACEMENT_MAP.get(normalized_sector)
    if not replace_column:
        supported = ", ".join(sorted(SECTOR_REPLACEMENT_MAP))
        raise ValueError(f"Unsupported sector '{sector}'. Expected one of: {supported}")

    aligned_returns = returns.copy()
    aligned_returns[replace_column] = replacement_series.reindex(returns.index).fillna(0)
    aligned_returns = aligned_returns[FIXED_COMPANY_ORDER]

    normalized = _normalize_returns(aligned_returns).tail(seq_len)
    if len(normalized) < seq_len:
        raise ValueError("Not enough aligned history to build a 30-step tensor")

    return {
        "ticker": ticker,
        "sector": normalized_sector,
        "source_timestamp": _index_value_to_iso(normalized.index[-1]),
        "columns": list(normalized.columns),
        "replaced_column": replace_column,
        "sequence_length": seq_len,
        "normalized_returns": normalized.values.tolist(),
    }


def prepared_input_to_tensor(prepared_input: dict):
    matrix = prepared_input["normalized_returns"]
    tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)

    expected_shape = (1, prepared_input["sequence_length"], len(FIXED_COMPANY_ORDER))
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Prepared tensor shape {tuple(tensor.shape)} does not match expected {expected_shape}"
        )

    return tensor
