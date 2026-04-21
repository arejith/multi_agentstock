from datetime import datetime


class TransformerInputStore:
    def __init__(self):
        self.data = {}

    def add(self, prepared_input: dict, role="data_team"):
        if role != "data_team":
            raise PermissionError("Only Data Team can write")

        ticker = prepared_input.get("ticker")
        if not ticker:
            raise ValueError("Ticker missing")

        stored_input = dict(prepared_input)
        stored_at = stored_input.get("source_timestamp") or datetime.utcnow().isoformat()
        stored_input["source_timestamp"] = stored_at
        stored_input["date"] = stored_input.get("date") or stored_at.split("T")[0]

        if ticker not in self.data:
            self.data[ticker] = []

        self.data[ticker].append(stored_input)

    def get_latest(self, ticker: str):
        if ticker not in self.data:
            return None

        return sorted(
            self.data[ticker],
            key=lambda item: item["source_timestamp"],
        )[-1]

    def get_history(self, ticker: str):
        return self.data.get(ticker, [])
