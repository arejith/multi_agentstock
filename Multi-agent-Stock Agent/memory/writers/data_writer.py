class DataWriter:
    def __init__(self, news_store, fundamentals_store, transformer_store):
        self.news_store = news_store
        self.fundamentals_store = fundamentals_store
        self.transformer_store = transformer_store

    def write_news(self, news_list):
        return self.news_store.build(news_list, role="data_team")

    def write_fundamentals(self, fundamentals):
        self.fundamentals_store.add(fundamentals, role="data_team")

    def write_transformer_input(self, prepared_input):
        self.transformer_store.add(prepared_input, role="data_team")
