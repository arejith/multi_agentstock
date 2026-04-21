from langchain.tools import tool


def create_transformer_input_tool(transformer_retriever):
    @tool
    def get_transformer_input(ticker: str):
        """
        Read the latest stored transformer input for a ticker from memory.
        """
        data = transformer_retriever.get_latest(ticker)

        if not data:
            return {"ticker": ticker, "found": False}

        return data

    return get_transformer_input
