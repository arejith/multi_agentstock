import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.stores.news_store import NewsVectorStore
from memory.writers.data_writer import DataWriter
from tools.data_sources.news import _article_signature


class EmptyStore:
    def add(self, *_args, **_kwargs):
        return None


def run_news_dedup_test():
    news_store = NewsVectorStore()
    writer = DataWriter(news_store, EmptyStore(), EmptyStore())

    duplicate_batch = [
        {
            "ticker": "AAPL",
            "title": "Apple shares rise",
            "text": "Apple shares rise after services revenue improves.",
            "source": "test",
            "url": "https://example.com/apple-rise",
            "timestamp": "2026-04-05T10:00:00",
            "date": "2026-04-05",
        },
        {
            "ticker": "AAPL",
            "title": "Apple shares rise",
            "text": "Apple shares rise after services revenue improves.",
            "source": "test",
            "url": "https://example.com/apple-rise",
            "timestamp": "2026-04-05T10:01:00",
            "date": "2026-04-05",
        },
        {
            "ticker": "AAPL",
            "title": "Apple margin improves",
            "text": "Apple margin improves as services revenue keeps growing.",
            "source": "test",
            "url": "https://example.com/apple-margin",
            "timestamp": "2026-04-05T10:02:00",
            "date": "2026-04-05",
        },
    ]

    first_count = writer.write_news(duplicate_batch)
    if first_count != 2:
        raise AssertionError(f"Expected two unique articles from first batch, got {first_count}")

    second_count = writer.write_news(duplicate_batch)
    if second_count != 0:
        raise AssertionError(f"Expected repeated batch to store zero new articles, got {second_count}")

    results = news_store.query("Apple", ticker="AAPL", k=5)
    titles = [result.metadata["title"] for result in results]
    if titles.count("Apple shares rise") != 1:
        raise AssertionError(f"Query results should dedupe repeated headlines: {titles}")
    if len(results) != 2:
        raise AssertionError(f"Query should return exactly two unique stored articles, got {len(results)}")

    signature_with_title = _article_signature(
        "AAPL",
        "Apple shares rise",
        "https://example.com/apple-rise",
        "Different body text",
    )
    if signature_with_title != ("title", "AAPL", "apple shares rise"):
        raise AssertionError(f"Signatures should prefer stable article titles: {signature_with_title}")

    signature_with_url = _article_signature(
        "AAPL",
        "",
        "https://example.com/apple-rise",
        "Different body text",
    )
    if signature_with_url != ("url", "AAPL", "https://example.com/apple-rise"):
        raise AssertionError(f"URL signatures should be used when titles are missing: {signature_with_url}")

    print("News deduplication validated successfully.")


if __name__ == "__main__":
    run_news_dedup_test()
