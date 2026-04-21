from langchain.tools import tool
from pydantic import BaseModel, Field

from tools.data_sources.news import COMPANY_MAP, get_news

GOLDEN_RULE = "Data tools write data. Analysis tools read stored data and analyze only."

class NewsKeywordSet(BaseModel):
    keywords: list[str] = Field(description="Short keyword list for filtering the latest company-specific news.")


def _build_keywords(llm, ticker: str, sector: str = "", company_name: str = ""):
    base_keywords = [ticker.upper(), ticker.lower()]
    resolved_company = (company_name or COMPANY_MAP.get(ticker.upper()) or ticker).strip()
    if resolved_company:
        base_keywords.append(resolved_company.lower())
    if sector:
        base_keywords.append(sector.lower())

    deduped_base = list(dict.fromkeys(keyword for keyword in base_keywords if keyword))
    if llm is None or not hasattr(llm, "with_structured_output"):
        return deduped_base

    try:
        structured_llm = llm.with_structured_output(NewsKeywordSet)
        result = structured_llm.invoke(
            "You support a write-only stock news retrieval tool.\n"
            f"Golden rule: {GOLDEN_RULE}\n"
            "Generate a compact keyword list for filtering the latest company-relevant stock news.\n"
            "Do not analyze sentiment. Do not make a recommendation. Focus only on retrieval relevance.\n"
            "Return 4 to 8 short keywords or phrases.\n"
            f"Ticker: {ticker}\n"
            f"Company: {resolved_company}\n"
            f"Sector: {sector or 'unknown'}\n"
        )
        generated = [keyword.strip().lower() for keyword in result.keywords if keyword and keyword.strip()]
        return list(dict.fromkeys(deduped_base + generated))
    except Exception:
        return deduped_base


def create_news_write_tool(writer, llm=None):
    @tool
    def fetch_and_store_news(ticker: str, sector: str = "", company_name: str = ""):
        """
        Fetch financial news externally and store it in news memory.
        """
        keywords = _build_keywords(llm, ticker, sector=sector, company_name=company_name)
        try:
            news_list = get_news(ticker, sector=sector or None, keywords=keywords)
        except TypeError:
            news_list = get_news(ticker)

        if not news_list:
            return {
                "status": "no_data",
                "ticker": ticker,
                "stored_count": 0,
                "keywords": keywords,
            }

        writer.write_news(news_list)

        return {
            "status": "stored",
            "ticker": ticker,
            "stored_count": len(news_list),
            "latest_date": news_list[0].get("date"),
            "keywords": keywords,
        }

    return fetch_and_store_news
