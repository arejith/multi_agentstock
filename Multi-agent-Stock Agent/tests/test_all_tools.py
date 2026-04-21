"""
Future flow if agents existed:
- Data Agent would call data_tools first to fetch external data and write to memory.
- Analysis Agent would then call analysis_tools to read stored data and run analysis.
- Supervisor Agent would orchestrate both phases without breaking the boundary.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.retrievers.fundamentals_retriever import FundamentalsRetriever
from memory.retrievers.news_retriever import NewsRetriever
from memory.retrievers.transformer_retriever import TransformerInputRetriever
from memory.stores.fundamentals_store import FundamentalsStore
from memory.stores.news_store import NewsVectorStore
from memory.stores.transformer_store import TransformerInputStore
from memory.writers.data_writer import DataWriter
from tools.tools_registry import get_tool_sets

from tests.support import TEST_SECTOR, TEST_TICKER, fake_download, install_fakes


def run_all_tools_test():
    print("Instantiating stores, writer, retrievers, and registry...")

    install_fakes()

    news_store = NewsVectorStore()
    fundamentals_store = FundamentalsStore()
    transformer_store = TransformerInputStore()
    writer = DataWriter(news_store, fundamentals_store, transformer_store)

    news_retriever = NewsRetriever(news_store)
    fundamentals_retriever = FundamentalsRetriever(fundamentals_store)
    transformer_retriever = TransformerInputRetriever(transformer_store)

    tool_sets = get_tool_sets(
        news_retriever,
        fundamentals_retriever,
        transformer_retriever,
        writer,
    )
    data_tools = {tool.name: tool for tool in tool_sets["data_tools"]}
    analysis_tools = {tool.name: tool for tool in tool_sets["analysis_tools"]}

    print("data_tools:", list(data_tools))
    print("analysis_tools:", list(analysis_tools))

    required_data_tools = {
        "fetch_and_store_news",
        "fetch_and_store_fundamentals",
        "fetch_and_store_transformer_input",
    }
    required_analysis_tools = {
        "get_news",
        "get_fundamentals",
        "get_transformer_input",
        "get_sentiment_analysis",
        "get_risk_analysis",
        "run_transformer_analysis",
    }

    missing_data = required_data_tools - set(data_tools)
    missing_analysis = required_analysis_tools - set(analysis_tools)
    if missing_data:
        raise AssertionError(f"Missing data tools: {sorted(missing_data)}")
    if missing_analysis:
        raise AssertionError(f"Missing analysis tools: {sorted(missing_analysis)}")

    print("\nRunning data tools...")
    news_output = data_tools["fetch_and_store_news"].invoke({"ticker": TEST_TICKER})
    fundamentals_output = data_tools["fetch_and_store_fundamentals"].invoke({"ticker": TEST_TICKER})
    transformer_store_output = data_tools["fetch_and_store_transformer_input"].invoke(
        {"ticker": TEST_TICKER, "sector": TEST_SECTOR}
    )
    prepared_transformer_input = transformer_retriever.get_latest(TEST_TICKER)

    print("fetch_and_store_news:", news_output)
    print("fetch_and_store_fundamentals:", fundamentals_output)
    print("fetch_and_store_transformer_input:", transformer_store_output)
    print(
        "stored_transformer_input:",
        {
            "ticker": prepared_transformer_input["ticker"],
            "sector": prepared_transformer_input["sector"],
            "replaced_column": prepared_transformer_input["replaced_column"],
            "sequence_length": prepared_transformer_input["sequence_length"],
        },
    )

    print("\nRunning analysis tools...")
    news_read_output = analysis_tools["get_news"].invoke(
        {"query": "strong growth", "ticker": TEST_TICKER}
    )
    sentiment_output = analysis_tools["get_sentiment_analysis"].invoke({"ticker": TEST_TICKER})
    risk_output = analysis_tools["get_risk_analysis"].invoke({"ticker": TEST_TICKER})
    transformer_input_output = analysis_tools["get_transformer_input"].invoke({"ticker": TEST_TICKER})
    transformer_output = analysis_tools["run_transformer_analysis"].invoke(
        {"prepared_input_json": json.dumps(transformer_input_output)}
    )
    fundamentals_read_output = analysis_tools["get_fundamentals"].invoke({"ticker": TEST_TICKER})

    print("get_news:", news_read_output)
    print("get_fundamentals:", fundamentals_read_output)
    print("get_transformer_input:", transformer_input_output)
    print("get_sentiment_analysis:", sentiment_output)
    print("get_risk_analysis:", risk_output)
    print("run_transformer_analysis:", transformer_output)

    if news_output["status"] != "stored":
        raise AssertionError(f"Unexpected news tool output: {news_output}")
    if fundamentals_output["status"] != "stored":
        raise AssertionError(f"Unexpected fundamentals tool output: {fundamentals_output}")
    if transformer_store_output["status"] != "stored":
        raise AssertionError(f"Unexpected transformer write output: {transformer_store_output}")
    if not news_read_output:
        raise AssertionError("News read tool returned no stored results")
    if transformer_input_output["ticker"] != TEST_TICKER:
        raise AssertionError(f"Unexpected transformer input output: {transformer_input_output}")
    expected_transformer_timestamp = fake_download("AAPL")["Close"].index[-1].isoformat()
    if transformer_input_output["source_timestamp"] != expected_transformer_timestamp:
        raise AssertionError(
            f"Transformer input should use latest Yahoo market timestamp: {transformer_input_output}"
        )
    if sentiment_output["ticker"] != TEST_TICKER or sentiment_output["count"] < 1:
        raise AssertionError(f"Unexpected sentiment output: {sentiment_output}")
    if risk_output["ticker"] != TEST_TICKER or risk_output["risk_level"] not in {"low", "medium", "high"}:
        raise AssertionError(f"Unexpected risk output: {risk_output}")
    if transformer_output["ticker"] != TEST_TICKER or "prediction" not in transformer_output:
        raise AssertionError(f"Unexpected transformer output: {transformer_output}")

    print("\nAll tool links validated successfully.")


if __name__ == "__main__":
    run_all_tools_test()
