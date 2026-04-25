import os
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")
DEFAULT_LOOKBACK_DAYS = 3
MIN_TEXT_LENGTH = 40

COMPANY_MAP = {
    "AAPL": "apple",
    "MSFT": "microsoft",
    "TSLA": "tesla",
    "GOOGL": "google",
    "GOOG": "google",
    "AMZN": "amazon",
    "NVDA": "nvidia",
}


def _format_timestamp(raw_value):
    try:
        return datetime.fromtimestamp(raw_value, tz=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _format_date(raw_value):
    try:
        return datetime.fromtimestamp(raw_value, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _is_relevant(text: str, ticker: str, sector: str | None = None, keywords: list[str] | None = None):
    normalized_text = text.lower()
    company_name = COMPANY_MAP.get(ticker.upper(), ticker.lower())
    active_keywords = list(keywords or [ticker.lower(), company_name])

    if sector:
        active_keywords.append(sector.lower())

    return any(keyword in normalized_text for keyword in active_keywords if keyword)


def _normalize_signature_value(value: str | None):
    return " ".join((value or "").strip().lower().split())


def _article_signature(ticker: str, title: str, url: str | None, text: str):
    normalized_title = _normalize_signature_value(title)
    if normalized_title:
        return ("title", ticker.upper(), normalized_title)
    normalized_url = (url or "").strip().lower()
    if normalized_url:
        return ("url", ticker.upper(), normalized_url)
    return (
        "content",
        ticker.upper(),
        _normalize_signature_value(title),
        _normalize_signature_value(text),
    )


def _extract_article_text(url: str | None):
    if not url:
        return None

    try:
        from newspaper import Article
    except Exception:
        return None

    try:
        article = Article(url)
        article.download()
        article.parse()
        text = (article.text or "").strip()
        return text or None
    except Exception:
        return None


def get_news(ticker="AAPL", limit=5, sector: str | None = None, keywords: list[str] | None = None):
    if not API_KEY:
        raise ValueError("NEWS_API_KEY is required to fetch news")

    today = datetime.now(timezone.utc).date()
    start_date = today - timedelta(days=DEFAULT_LOOKBACK_DAYS)

    response = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={
            "symbol": ticker,
            "from": start_date.isoformat(),
            "to": today.isoformat(),
            "token": API_KEY,
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list):
        raise ValueError(f"Finnhub error: {data}")

    ranked_articles = sorted(data, key=lambda item: item.get("datetime") or 0, reverse=True)
    articles = []
    seen_signatures = set()
    active_keywords = [keyword.strip().lower() for keyword in (keywords or []) if keyword and keyword.strip()]

    for article in ranked_articles:
        title = (article.get("headline") or "").strip()
        description = (article.get("summary") or "").strip()
        text = f"{title} {description}".strip()

        if not text or len(text) < MIN_TEXT_LENGTH:
            continue

        if not _is_relevant(text, ticker, sector, active_keywords):
            continue

        published_at = article.get("datetime")
        timestamp = _format_timestamp(published_at)
        article_text = _extract_article_text(article.get("url"))
        final_text = article_text if article_text and len(article_text) >= 100 else text
        signature = _article_signature(ticker, title, article.get("url"), final_text)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        articles.append(
            {
                "ticker": ticker,
                "title": title or f"{ticker} news",
                "description": description,
                "summary": description,
                "text": final_text,
                "source": article.get("source") or "finnhub",
                "url": article.get("url"),
                "timestamp": timestamp,
                "date": _format_date(published_at),
                "keywords": active_keywords,
            }
        )

        if len(articles) >= limit:
            break

    return articles
