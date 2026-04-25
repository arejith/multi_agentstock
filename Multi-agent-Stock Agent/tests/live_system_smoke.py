import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Agents.supervisor.runtime import run_stock_analysis


LIVE_REQUESTS = [
    "Analyze Apple stock and figure out the right sector, then tell me the outlook.",
    "Review JPMorgan as a stock pick and infer the ticker and sector before analysis.",
    "Check Exxon stock, determine the correct ticker and sector, and summarize the signal.",
    "Just fetch news for Apple stock.",
    "Just fetch fundamentals for Microsoft.",
    "Do sentiment analysis for Amazon stock.",
    "Run risk analysis for Walmart stock.",
    "Run transformer analysis for Exxon stock.",
]


def validate_result(result: dict):
    required_top_keys = {
        "request",
        "flow",
        "data_phase",
        "analysis_phase",
        "final_report",
        "request_resolution",
        "request_route",
    }
    missing = required_top_keys - set(result)
    if missing:
        raise AssertionError(f"Missing top-level keys: {sorted(missing)}")

    if not result["request_resolution"]["ticker"]:
        raise AssertionError("Resolved ticker is empty")
    if not result["request_resolution"]["sector"]:
        raise AssertionError("Resolved sector is empty")

    action = result["request_route"]["action"]
    data_phase = result["data_phase"]
    analysis_phase = result["analysis_phase"]
    final_report = result["final_report"]

    if action == "full_analysis":
        if data_phase["news"]["status"] != "stored":
            raise AssertionError(f"News step failed: {data_phase['news']}")
        if data_phase["fundamentals"]["status"] != "stored":
            raise AssertionError(f"Fundamentals step failed: {data_phase['fundamentals']}")
        if data_phase["transformer_input"]["status"] != "stored":
            raise AssertionError(f"Transformer input step failed: {data_phase['transformer_input']}")
        if analysis_phase["news_analysis"]["sentiment"]["count"] < 1:
            raise AssertionError(f"Sentiment did not analyze any stored news: {analysis_phase['news_analysis']}")
        if analysis_phase["risk_analysis"]["risk"]["risk_level"] not in {"low", "medium", "high", "unknown"}:
            raise AssertionError(f"Unexpected risk level: {analysis_phase['risk_analysis']['risk']}")
        transformer_signal = analysis_phase["transformer_analysis"]["transformer"]["signal"]
        if transformer_signal != "UNKNOWN" and "predicted daily return" not in transformer_signal:
            raise AssertionError(
                f"Unexpected transformer signal: {analysis_phase['transformer_analysis']['transformer']}"
            )
        if final_report["recommendation"] not in {
            "strong_buy",
            "buy",
            "hold",
            "watchlist_buy",
            "sell",
            "strong_sell",
        }:
            raise AssertionError(f"Unexpected recommendation: {final_report}")
        return

    if action == "fetch_news":
        if data_phase["news"]["status"] != "success":
            raise AssertionError(f"Fetch news request failed: {data_phase}")
        if not isinstance(data_phase["news"]["news"], list):
            raise AssertionError(f"Fetch news request did not return readable data: {data_phase}")
        if final_report["recommendation"] != "data_fetched":
            raise AssertionError(f"Fetch news final report invalid: {final_report}")
        return

    if action == "fetch_fundamentals":
        if data_phase["fundamentals"]["status"] != "success":
            raise AssertionError(f"Fetch fundamentals request failed: {data_phase}")
        if "fundamentals" not in data_phase["fundamentals"]:
            raise AssertionError(f"Fetch fundamentals request did not return readable data: {data_phase}")
        if final_report["recommendation"] != "data_fetched":
            raise AssertionError(f"Fetch fundamentals final report invalid: {final_report}")
        return

    if action == "fetch_transformer_input":
        if data_phase["transformer_input"]["status"] != "success":
            raise AssertionError(f"Fetch transformer input request failed: {data_phase}")
        if "transformer_input" not in data_phase["transformer_input"]:
            raise AssertionError(f"Fetch transformer input request did not return readable data: {data_phase}")
        if final_report["recommendation"] != "data_fetched":
            raise AssertionError(f"Fetch transformer final report invalid: {final_report}")
        return

    if action == "analyze_sentiment":
        if analysis_phase["news_analysis"]["sentiment"]["count"] < 1:
            raise AssertionError(f"Sentiment-only request failed: {analysis_phase}")
        if final_report["recommendation"] != "analysis_complete":
            raise AssertionError(f"Sentiment-only final report invalid: {final_report}")
        return

    if action == "analyze_risk":
        if analysis_phase["risk_analysis"]["risk"]["risk_level"] not in {"low", "medium", "high", "unknown"}:
            raise AssertionError(f"Risk-only request failed: {analysis_phase}")
        if final_report["recommendation"] != "analysis_complete":
            raise AssertionError(f"Risk-only final report invalid: {final_report}")
        return

    if action == "analyze_transformer":
        transformer_signal = analysis_phase["transformer_analysis"]["transformer"]["signal"]
        if transformer_signal != "UNKNOWN" and "predicted daily return" not in transformer_signal:
            raise AssertionError(f"Transformer-only request failed: {analysis_phase}")
        if final_report["recommendation"] != "analysis_complete":
            raise AssertionError(f"Transformer-only final report invalid: {final_report}")
        return

    raise AssertionError(f"Unhandled action in validation: {action}")


def format_demo_report(index: int, request: str, result: dict):
    resolved = result["request_resolution"]
    final_report = result["final_report"]
    action = result["request_route"]["action"]

    lines = [
        f"Case {index}",
        f"Prompt: {request}",
        f"Action: {action}",
        (
            "Resolved: "
            f"{resolved['company_name']} -> {resolved['ticker']} ({resolved['sector']})"
        ),
        f"Decision: {final_report['recommendation'].upper()}",
        f"Golden rule: {result['flow']['golden_rule']}",
    ]

    if action == "full_analysis":
        sentiment = result["analysis_phase"]["news_analysis"]["sentiment"]
        risk = result["analysis_phase"]["risk_analysis"]["risk"]
        transformer = result["analysis_phase"]["transformer_analysis"]["transformer"]
        lines.append(
            "Signals: "
            f"sentiment_score={sentiment['score']}, "
            f"risk={risk['risk_level']}, "
            f"transformer={transformer['signal']} ({transformer['prediction']})"
        )
    elif action == "analyze_sentiment":
        sentiment = result["analysis_phase"]["news_analysis"]["sentiment"]
        lines.append(f"Signals: sentiment_score={sentiment['score']}")
    elif action == "analyze_risk":
        risk = result["analysis_phase"]["risk_analysis"]["risk"]
        lines.append(f"Signals: risk={risk['risk_level']}")
    elif action == "analyze_transformer":
        transformer = result["analysis_phase"]["transformer_analysis"]["transformer"]
        lines.append(
            f"Signals: transformer={transformer['signal']} ({transformer['prediction']})"
        )
    else:
        lines.append(f"Signals: {final_report['reasoning'][0]}")

    lines.append(f"Reasoning: {' | '.join(final_report['reasoning'])}")
    return "\n".join(lines)


def print_summary(results: list[dict]):
    recommendation_counts = {}
    sentiment_scores = []
    prediction_values = []

    for result in results:
        final_report = result["final_report"]
        recommendation = final_report["recommendation"]
        recommendation_counts[recommendation] = recommendation_counts.get(recommendation, 0) + 1

        if result["request_route"]["action"] in {"full_analysis", "analyze_sentiment"}:
            sentiment = result["analysis_phase"]["news_analysis"]["sentiment"]
            sentiment_scores.append(sentiment["score"])
        if result["request_route"]["action"] in {"full_analysis", "analyze_transformer"}:
            transformer = result["analysis_phase"]["transformer_analysis"]["transformer"]
            prediction_values.append(transformer["prediction"])

    print("=== SUMMARY ===")
    print(f"Total prompts: {len(results)}")
    if sentiment_scores:
        print(f"Average sentiment score: {round(mean(sentiment_scores), 4)}")
    if prediction_values:
        print(f"Average transformer prediction: {round(mean(prediction_values), 6)}")
    print("Recommendation counts:")
    for recommendation, count in sorted(recommendation_counts.items()):
        print(f"  - {recommendation}: {count}")
    print()


def run_live_smoke_test():
    print("Running live end-to-end stock analysis evaluation...\n")
    results = []

    for index, request in enumerate(LIVE_REQUESTS, start=1):
        if index > 1:
            time.sleep(13)
        result = run_stock_analysis(user_request=request)
        validate_result(result)
        results.append(result)
        print(format_demo_report(index, request, result))
        print()

    print_summary(results)
    print("LIVE SYSTEM EVALUATION PASSED")


if __name__ == "__main__":
    run_live_smoke_test()
