class NewsRetriever:
    def __init__(self, news_store):
        self.store = news_store

    def get_news(self, query, ticker):
        return self.store.query(query, ticker=ticker)