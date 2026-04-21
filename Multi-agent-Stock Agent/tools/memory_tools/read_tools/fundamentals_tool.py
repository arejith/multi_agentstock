from langchain.tools import tool


def create_fundamentals_tool(fundamentals_retriever):
    @tool
    def get_fundamentals(ticker: str):
        """
        Read the latest stored fundamentals for a ticker from memory.
        """
        data = fundamentals_retriever.get_latest(ticker)

        if not data:
            return {"ticker": ticker, "found": False}

        return data

    return get_fundamentals
