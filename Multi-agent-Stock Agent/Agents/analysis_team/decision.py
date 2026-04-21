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


def fallback_decision(analysis_outputs: dict):
    sentiment_score = float((analysis_outputs.get("sentiment") or {}).get("score") or 0.0)
    risk_level = (analysis_outputs.get("risk") or {}).get("risk_level") or "unknown"
    transformer_signal = (analysis_outputs.get("transformer") or {}).get("signal") or "UNKNOWN"

    score = 0
    if sentiment_score > 0.2:
        score += 1
    elif sentiment_score < -0.2:
        score -= 1
    if risk_level == "low":
        score += 1
    elif risk_level == "high":
        score -= 1
    if transformer_signal == "BUY":
        score += 1
    elif transformer_signal == "SELL":
        score -= 1

    decision = "BUY" if score >= 2 else "SELL" if score <= -1 else "HOLD"
    confidence = round(min(0.95, 0.55 + 0.12 * abs(score)), 2)
    reasoning = [
        f"Sentiment score: {sentiment_score}",
        f"Risk level: {risk_level}",
        f"Transformer signal: {transformer_signal}",
    ]
    risks = []
    if risk_level in {"medium", "high", "unknown"}:
        risks.append(f"Company risk is {risk_level}.")
    if transformer_signal == "SELL":
        risks.append("Price model signal is bearish.")
    if sentiment_score < 0:
        risks.append("Recent news sentiment is negative.")
    if not risks:
        risks.append("No major risk flag from the current tool outputs.")
    return decision, confidence, reasoning, risks


class DecisionLayer:
    def __init__(self, llm=None):
        self.llm = llm

    def decide(self, ticker: str, company: str | None = None, sector: str | None = None, analysis_outputs: dict | None = None):
        if isinstance(sector, dict) and analysis_outputs is None:
            analysis_outputs = sector
            sector = company
            company = ticker

        analysis_outputs = normalize_analysis_outputs(analysis_outputs or {})
        decision, confidence, reasoning, risks = fallback_decision(analysis_outputs)

        return {
            "ticker": ticker,
            "company": company,
            "sector": sector,
            "decision": decision,
            "recommendation": {"BUY": "buy", "HOLD": "hold", "SELL": "sell"}.get(decision, "hold"),
            "confidence": confidence,
            "reasoning": reasoning,
            "risks": risks,
            "sentiment_score": (analysis_outputs.get("sentiment") or {}).get("score"),
            "risk_level": (analysis_outputs.get("risk") or {}).get("risk_level"),
            "transformer_signal": (analysis_outputs.get("transformer") or {}).get("signal"),
            "prediction": (analysis_outputs.get("transformer") or {}).get("prediction"),
        }
