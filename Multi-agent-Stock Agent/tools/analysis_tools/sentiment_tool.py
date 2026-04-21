from langchain.tools import tool

from models.sentiment.sentiment_model import SentimentModel


def create_sentiment_tool(news_retriever):
    model = SentimentModel()

    @tool
    def get_sentiment_analysis(ticker: str):
        """
        Analyze sentiment using news already stored in memory.
        """
        docs = news_retriever.get_news(ticker, ticker)
        texts = [doc.page_content for doc in docs if getattr(doc, "page_content", "").strip()]

        result = model.analyze_batch(texts)
        result["ticker"] = ticker
        return result

    return get_sentiment_analysis
