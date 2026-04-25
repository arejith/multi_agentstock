import json


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


def score_decision(analysis_outputs: dict):
    sentiment_score = float((analysis_outputs.get("sentiment") or {}).get("score") or 0.0)
    risk_level = (analysis_outputs.get("risk") or {}).get("risk_level") or "unknown"
    transformer = analysis_outputs.get("transformer") or {}
    transformer_forecast = transformer.get("signal") or "UNKNOWN"
    transformer_prediction = transformer.get("prediction")
    transformer_value = float(transformer_prediction or 0.0)

    score = 0
    if sentiment_score > 0.2:
        score += 1
    elif sentiment_score < -0.2:
        score -= 1
    if risk_level == "low":
        score += 1
    elif risk_level == "high":
        score -= 1
    if transformer_value > 0.01:
        score += 1
    elif transformer_value < -0.01:
        score -= 1

    decision = "BUY" if score >= 2 else "SELL" if score <= -1 else "HOLD"
    confidence = round(min(0.95, 0.55 + 0.12 * abs(score)), 2)
    reasoning = [
        f"Sentiment score: {sentiment_score}",
        f"Risk level: {risk_level}",
        f"Transformer forecast: {transformer_forecast}",
    ]
    risks = []
    if risk_level in {"medium", "high", "unknown"}:
        risks.append(f"Company risk is {risk_level}.")
    if transformer_value < -0.01:
        risks.append("Price model signal is bearish.")
    if sentiment_score < 0:
        risks.append("Recent news sentiment is negative.")
    if not risks:
        risks.append("No major risk flag from the current tool outputs.")
    return decision, confidence, reasoning, risks


ALLOWED_DECISIONS = {"BUY", "HOLD", "SELL"}
ALLOWED_RECOMMENDATIONS = {"strong_buy", "buy", "hold", "watchlist_buy", "sell", "strong_sell"}


class DecisionLayer:
    def __init__(self, llm=None):
        self.llm = llm

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
        llm_decision = self._decide_with_llm(ticker, company, sector, analysis_outputs)
        if llm_decision:
            return llm_decision

        decision, confidence, reasoning, risks = score_decision(analysis_outputs)

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

    def _decide_with_llm(self, ticker: str, company: str | None, sector: str | None, analysis_outputs: dict):
        if self.llm is None:
            return None

        transformer = analysis_outputs.get("transformer") or {}
        prompt = (
            "You are the final decision layer for a stock-analysis pipeline. "
            "Use the transformer forecast as a next-day return percentage forecast, not as a fixed BUY/SELL/HOLD label. "
            "Decide the final recommendation by weighing sentiment, risk, fundamentals, and the transformer forecast. "
            "Explain why you made the decision in specific, user-readable terms. "
            "Return only valid JSON with keys: decision, recommendation, confidence, reasoning, risks. "
            "decision must be one of BUY, HOLD, SELL. "
            "recommendation must be one of strong_buy, buy, hold, watchlist_buy, sell, strong_sell. "
            "The reasoning value must be an array of concise strings explaining why this decision was made.\n\n"
            f"Ticker: {ticker}\n"
            f"Company: {company}\n"
            f"Sector: {sector}\n"
            f"Sentiment: {json.dumps(analysis_outputs.get('sentiment'), default=str)}\n"
            f"Risk: {json.dumps(analysis_outputs.get('risk'), default=str)}\n"
            f"Fundamentals: {json.dumps(analysis_outputs.get('fundamentals'), default=str)}\n"
            f"Transformer forecast: {transformer.get('signal')}\n"
            f"Transformer next-day return prediction: {transformer.get('prediction')}\n"
        )

        try:
            response = self.llm.invoke(prompt)
            content = getattr(response, "content", response)
            payload = _parse_json_object(str(content))
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None
        recommendation = str(payload.get("recommendation") or "hold").lower()
        decision = payload.get("decision")
        if isinstance(decision, str):
            decision = decision.upper()
        if decision not in ALLOWED_DECISIONS:
            decision = {"strong_buy": "BUY", "buy": "BUY", "watchlist_buy": "BUY", "hold": "HOLD", "sell": "SELL", "strong_sell": "SELL"}.get(
                recommendation,
                "HOLD",
            )
        if recommendation not in ALLOWED_RECOMMENDATIONS:
            recommendation = {"BUY": "buy", "HOLD": "hold", "SELL": "sell"}.get(decision, "hold")
        reasoning = payload.get("reasoning") or []
        if isinstance(reasoning, str):
            reasoning = [reasoning]
        risks = payload.get("risks") or []
        if isinstance(risks, str):
            risks = [risks]

        return {
            "ticker": ticker,
            "company": company,
            "sector": sector,
            "decision": decision,
            "recommendation": recommendation,
            "confidence": payload.get("confidence"),
            "reasoning": reasoning,
            "risks": risks,
            "sentiment_score": (analysis_outputs.get("sentiment") or {}).get("score"),
            "risk_level": (analysis_outputs.get("risk") or {}).get("risk_level"),
            "transformer_signal": transformer.get("signal"),
            "prediction": transformer.get("prediction"),
        }


def _parse_json_object(content: str):
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])
