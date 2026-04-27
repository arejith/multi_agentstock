from typing import Literal

from pydantic import BaseModel, Field


def normalize_analysis_outputs(analysis_outputs: dict) -> dict:
    if "news_analysis" not in analysis_outputs:
        return analysis_outputs
    return {
        "news": analysis_outputs.get("news_analysis", {}).get("news"),
        "sentiment": analysis_outputs.get("news_analysis", {}).get("sentiment"),
        "fundamentals": analysis_outputs.get("risk_analysis", {}).get("fundamentals"),
        "risk": analysis_outputs.get("risk_analysis", {}).get("risk"),
        "prepared_input": analysis_outputs.get("transformer_analysis", {}).get("prepared_input"),
        "transformer": analysis_outputs.get("transformer_analysis", {}).get("transformer"),
    }


class DecisionLayer:
    def __init__(self, llm=None):
        self.llm = llm
        if llm is None or not hasattr(llm, "with_structured_output"):
            raise ValueError("DecisionLayer requires an LLM with structured output support")
        self.structured_llm = llm.with_structured_output(DecisionOutput)

    def decide(
        self,
        ticker: str,
        company: str | None = None,
        sector: str | None = None,
        analysis_outputs: dict | None = None,
        **kwargs,
    ):
        if analysis_outputs is None and "analysis_phase" in kwargs:
            analysis_outputs = kwargs["analysis_phase"]
        if isinstance(sector, dict) and analysis_outputs is None:
            analysis_outputs = sector
            sector = company
            company = ticker

        analysis_outputs = normalize_analysis_outputs(analysis_outputs or {})
        if self.llm is None:
            raise ValueError("DecisionLayer requires an LLM")

        transformer = analysis_outputs.get("transformer") or {}
        prompt = (
            "You are the final decision layer for a stock-analysis pipeline. "
            "Use the transformer forecast as a next-day return percentage forecast, not as a fixed BUY/SELL/HOLD label. "
            "Decide the final recommendation by weighing sentiment, risk, fundamentals, and the transformer forecast. "
            "Explain why you made the decision in specific, user-readable terms. "
            "Return structured output. "
            "The reasoning value must be an array of concise strings explaining why this decision was made.\n\n"
            f"Ticker: {ticker}\n"
            f"Company: {company}\n"
            f"Sector: {sector}\n"
            f"Sentiment: {analysis_outputs.get('sentiment')}\n"
            f"Risk: {analysis_outputs.get('risk')}\n"
            f"Fundamentals: {analysis_outputs.get('fundamentals')}\n"
            f"Transformer forecast: {transformer.get('signal')}\n"
            f"Transformer next-day return prediction: {transformer.get('prediction')}\n"
        )

        payload = self.structured_llm.invoke(prompt)

        return {
            "ticker": ticker,
            "company": company,
            "sector": sector,
            "decision": payload.decision,
            "recommendation": payload.recommendation,
            "confidence": payload.confidence,
            "reasoning": payload.reasoning,
            "risks": payload.risks,
            "sentiment_score": (analysis_outputs.get("sentiment") or {}).get("score"),
            "risk_level": (analysis_outputs.get("risk") or {}).get("risk_level"),
            "transformer_signal": transformer.get("signal"),
            "prediction": transformer.get("prediction"),
        }

class DecisionOutput(BaseModel):
    decision: Literal["BUY", "HOLD", "SELL"]
    recommendation: Literal["strong_buy", "buy", "hold", "watchlist_buy", "sell", "strong_sell"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: list[str]
    risks: list[str]
