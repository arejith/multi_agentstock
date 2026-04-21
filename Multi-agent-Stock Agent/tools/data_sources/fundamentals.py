from datetime import datetime

import yfinance as yf


def get_fundamentals(ticker="AAPL"):
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    timestamp = datetime.utcnow().isoformat()

    return {
        "ticker": ticker,
        "timestamp": timestamp,
        "date": timestamp.split("T")[0],
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "eps": info.get("trailingEps"),
        "revenue": info.get("totalRevenue"),
        "profit_margin": info.get("profitMargins"),
        "debt_to_equity": info.get("debtToEquity"),
        "return_on_equity": info.get("returnOnEquity"),
        "growth": info.get("earningsGrowth"),
    }
