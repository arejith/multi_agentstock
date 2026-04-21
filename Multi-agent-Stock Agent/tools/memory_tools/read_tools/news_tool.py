from langchain.tools import tool


def create_news_tool(news_retriever):

    @tool
    def get_news(query: str, ticker: str):
        """
        Fetch relevant financial news for a given ticker.
        """

        docs = news_retriever.get_news(query, ticker)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in docs
        ]

    return get_news
