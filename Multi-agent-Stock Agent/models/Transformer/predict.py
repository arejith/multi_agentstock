from tools.data_sources.stock_data import FIXED_COMPANY_ORDER, prepared_input_to_tensor

from models.Transformer.inference import TransformerModel


class TransformerPredictor:
    def __init__(self, model_path=None):
        self.model = TransformerModel(model_path)

    def predict_prepared(self, prepared_input: dict):
        replaced_column = prepared_input["replaced_column"]
        if replaced_column not in FIXED_COMPANY_ORDER:
            raise ValueError(f"Unknown replaced column: {replaced_column}")

        prepared_tensor = prepared_input_to_tensor(prepared_input)
        last_step = self.model.predict_last_timestep(prepared_tensor)
        prediction_index = FIXED_COMPANY_ORDER.index(replaced_column)
        prediction = last_step[0, prediction_index].item()

        return {
            "ticker": prepared_input["ticker"],
            "sector": prepared_input["sector"],
            "prediction": round(prediction, 6),
            "signal": self._get_signal(prediction),
            "replaced_column": replaced_column,
            "columns": prepared_input["columns"],
            "source_timestamp": prepared_input["source_timestamp"],
        }

    def _get_signal(self, value):
        if value > 0.01:
            return "BUY"
        if value < -0.01:
            return "SELL"
        return "HOLD"
