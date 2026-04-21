from langchain.tools import tool

from models.risk.risk_model import RiskModel


def create_risk_tool(fundamentals_retriever):
    model = RiskModel()

    @tool
    def get_risk_analysis(ticker: str):
        """
        Analyze risk using fundamentals already stored in memory.
        """
        fundamentals = fundamentals_retriever.get_latest(ticker)
        if not fundamentals:
            return {
                "ticker": ticker,
                "risk_score": None,
                "risk_level": "unknown",
                "reasons": ["No stored fundamentals available"],
            }

        return model.analyze(fundamentals)

    return get_risk_analysis
