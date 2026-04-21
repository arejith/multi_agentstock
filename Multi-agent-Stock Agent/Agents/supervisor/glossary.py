COMPANY_GLOSSARY = [
    ("AAPL", "Apple Inc.", "Information Technology", "technology"),
    ("AMZN", "Amazon.com Inc.", "Consumer Discretionary", "consumer_discretionary"),
    ("CAT", "Caterpillar Inc.", "Industrials", "industrials"),
    ("DOW", "Dow Inc.", "Materials", "materials"),
    ("GOOGL", "Alphabet Inc.", "Communication Services", "communication_services"),
    ("JNJ", "Johnson & Johnson", "Health Care", "healthcare"),
    ("JPM", "JPMorgan Chase & Co.", "Financials", "financials"),
    ("MSFT", "Microsoft Corp.", "Information Technology", "software"),
    ("WMT", "Walmart Inc.", "Consumer Staples", "consumer_staples"),
    ("XOM", "Exxon Mobil Corp.", "Energy", "energy"),
    ("NVDA", "NVIDIA Corp.", "Information Technology", None),
    ("TSLA", "Tesla Inc.", "Consumer Discretionary", None),
]

COMPANY_LOOKUP = {
    ticker: {
        "ticker": ticker,
        "company": company,
        "sector": sector,
        "tool_sector": tool_sector,
    }
    for ticker, company, sector, tool_sector in COMPANY_GLOSSARY
}


def glossary_text() -> str:
    return "\n".join(f"{ticker} | {company} | {sector}" for ticker, company, sector, _ in COMPANY_GLOSSARY)


def normalize_text(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
