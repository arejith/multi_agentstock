from langchain.tools import tool

from tools.data_sources.stock_data import build_transformer_input


def create_transformer_write_tool(writer):
    @tool
    def fetch_and_store_transformer_input(ticker: str, sector: str):
        """
        Build transformer-ready market input externally and store it in memory.
        """
        prepared_input = build_transformer_input(ticker, sector)
        writer.write_transformer_input(prepared_input)

        return {
            "status": "stored",
            "ticker": prepared_input["ticker"],
            "sector": prepared_input["sector"],
            "replaced_column": prepared_input["replaced_column"],
            "sequence_length": prepared_input["sequence_length"],
            "source_timestamp": prepared_input["source_timestamp"],
        }

    return fetch_and_store_transformer_input
