class TransformerInputRetriever:
    def __init__(self, transformer_store):
        self.store = transformer_store

    def get_latest(self, ticker: str):
        return self.store.get_latest(ticker)

    def get_prepared_input(self, ticker: str):
        return self.get_latest(ticker)

    def get_history(self, ticker: str):
        return self.store.get_history(ticker)
