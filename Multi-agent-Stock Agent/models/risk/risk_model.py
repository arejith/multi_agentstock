class RiskModel:
    def analyze(self, fundamentals: dict):
        if not fundamentals:
            return {
                "ticker": None,
                "risk_score": None,
                "risk_level": "unknown",
                "reasons": ["No fundamentals available"],
            }

        score = 0
        reasons = []

        pe_ratio = fundamentals.get("pe_ratio")
        debt_to_equity = fundamentals.get("debt_to_equity")
        profit_margin = fundamentals.get("profit_margin")
        return_on_equity = fundamentals.get("return_on_equity")
        growth = fundamentals.get("growth")
        market_cap = fundamentals.get("market_cap")

        if pe_ratio is None:
            score += 1
            reasons.append("Missing P/E ratio")
        elif pe_ratio > 35:
            score += 2
            reasons.append("High P/E ratio")

        if debt_to_equity is None:
            score += 1
            reasons.append("Missing debt-to-equity ratio")
        elif debt_to_equity > 150:
            score += 2
            reasons.append("High debt-to-equity ratio")

        if profit_margin is None:
            score += 1
            reasons.append("Missing profit margin")
        elif profit_margin < 0.05:
            score += 2
            reasons.append("Low profit margin")

        if return_on_equity is None:
            score += 1
            reasons.append("Missing return on equity")
        elif return_on_equity < 0.08:
            score += 1
            reasons.append("Weak return on equity")

        if growth is None:
            score += 1
            reasons.append("Missing earnings growth")
        elif growth < 0:
            score += 2
            reasons.append("Negative earnings growth")

        if market_cap is not None and market_cap < 5_000_000_000:
            score += 1
            reasons.append("Smaller market capitalization")

        if score <= 2:
            risk_level = "low"
        elif score <= 5:
            risk_level = "medium"
        else:
            risk_level = "high"

        if not reasons:
            reasons.append("Fundamentals look stable across tracked metrics")

        return {
            "ticker": fundamentals.get("ticker"),
            "risk_score": score,
            "risk_level": risk_level,
            "reasons": reasons,
        }
