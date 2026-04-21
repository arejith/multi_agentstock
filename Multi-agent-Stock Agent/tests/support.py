import pandas as pd

import models.sentiment.sentiment_model as sentiment_model_module
import tools.data_sources.fundamentals as fundamentals_source_module
import tools.data_sources.news as news_source_module
import tools.data_sources.stock_data as stock_data_source_module


TEST_TICKER = "AAPL"
TEST_SECTOR = "technology"
TEST_USER_REQUEST = "Analyze Apple stock in technology and tell me if it looks strong"


class FakeSentimentPipeline:
    def __call__(self, text):
        if "strong" in text.lower() or "growth" in text.lower():
            return [{"label": "POSITIVE", "score": 0.91}]
        return [{"label": "NEGATIVE", "score": 0.67}]


def fake_get_news(ticker="AAPL", limit=5, sector=None):
    if "INVALID" in ticker:
        return []
    return [
        {
            "ticker": ticker,
            "title": f"{ticker} outlook improves",
            "text": f"{ticker} delivered strong growth in services and margins.",
            "summary": "Strong growth and better margins.",
            "source": "test",
            "url": "https://example.com/news",
            "timestamp": "2026-04-05T10:00:00",
            "date": "2026-04-05",
        }
    ][:limit]


def fake_get_fundamentals(ticker="AAPL"):
    return {
        "ticker": ticker,
        "timestamp": "2026-04-05T10:05:00",
        "date": "2026-04-05",
        "market_cap": 250_000_000_000,
        "pe_ratio": 24.5,
        "forward_pe": 20.4,
        "eps": 6.12,
        "revenue": 100_000_000_000,
        "profit_margin": 0.24,
        "debt_to_equity": 45.0,
        "return_on_equity": 0.31,
        "growth": 0.14,
    }


def fake_download(symbols, period="6mo", auto_adjust=True, progress=False):
    index = pd.date_range("2026-01-01", periods=60, freq="B")
    if isinstance(symbols, str):
        values = pd.Series(range(100, 160), index=index, name=symbols)
        return pd.DataFrame({"Close": values})

    close_frame = pd.DataFrame(
        {
            symbol: pd.Series(range(100 + i, 160 + i), index=index)
            for i, symbol in enumerate(symbols)
        }
    )
    return pd.concat({"Close": close_frame}, axis=1)


def install_fakes():
    sentiment_model_module.SentimentModel._pipeline = FakeSentimentPipeline()
    news_source_module.get_news = fake_get_news
    fundamentals_source_module.get_fundamentals = fake_get_fundamentals
    stock_data_source_module.yf.download = fake_download
