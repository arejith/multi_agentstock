from langchain.tools import tool

from tools.data_sources.fundamentals import get_fundamentals


def create_fundamentals_write_tool(writer):
    @tool
    def fetch_and_store_fundamentals(ticker: str):
        """
        Fetch company fundamentals externally and store them in memory.
        """
        data = get_fundamentals(ticker)

        if not data:
            return {
                "status": "no_data",
                "ticker": ticker,
            }

        writer.write_fundamentals(data)

        return {
            "status": "stored",
            "ticker": ticker,
            "date": data.get("date"),
            "timestamp": data.get("timestamp"),
        }

    return fetch_and_store_fundamentals
