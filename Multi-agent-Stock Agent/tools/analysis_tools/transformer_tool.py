import json

from langchain.tools import tool

from models.Transformer.predict import TransformerPredictor


def create_transformer_tool():
    predictor = TransformerPredictor()

    @tool
    def run_transformer_analysis(prepared_input=None, prepared_input_json=None):
        """
        Run transformer inference on data prepared by the data layer.
        """
        if prepared_input is None and prepared_input_json is None:
            raise ValueError("Either prepared_input or prepared_input_json must be provided")

        if prepared_input is None:
            prepared_input = prepared_input_json

        if isinstance(prepared_input, str):
            prepared_input = json.loads(prepared_input)
        return predictor.predict_prepared(prepared_input)

    return run_transformer_analysis
