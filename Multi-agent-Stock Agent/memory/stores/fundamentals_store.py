from datetime import datetime


class FundamentalsStore:
    def __init__(self):
        self.data = {}

    def add(self, fundamentals: dict, role="data_team"):
        if role != "data_team":
            raise PermissionError("Only Data Team can write")

        ticker = fundamentals.get("ticker")
        if not ticker:
            raise ValueError("Ticker missing")

        stored_at = fundamentals.get("timestamp") or datetime.utcnow().isoformat()
        fundamentals["timestamp"] = stored_at
        fundamentals["date"] = fundamentals.get("date") or stored_at.split("T")[0]

        if ticker not in self.data:
            self.data[ticker] = []

        self.data[ticker].append(fundamentals)

    def get_latest(self, ticker: str):
        if ticker not in self.data:
            return None

        return sorted(
            self.data[ticker],
            key=lambda x: x["timestamp"]
        )[-1]

    def get_history(self, ticker: str):
        return self.data.get(ticker, [])
