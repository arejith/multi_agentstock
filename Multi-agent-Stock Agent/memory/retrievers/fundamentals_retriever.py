class FundamentalsRetriever:
    def __init__(self, fundamentals_store):
        self.store = fundamentals_store

    def get_latest(self, ticker):
        return self.store.get_latest(ticker)

    def get_fundamentals(self, ticker):
        return self.get_latest(ticker)

    def get_history(self, ticker):
        return self.store.get_history(ticker)
