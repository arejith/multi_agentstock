from pathlib import Path

import torch


DEFAULT_MODEL_PATH = Path("models/Transformer/timeseries_model_scripted.pt")


class TransformerModel:
    def __init__(self, model_path=None):
        self.device = torch.device("cpu")
        resolved_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.model = torch.jit.load(str(resolved_path), map_location=self.device)
        self.model.eval()

    def predict(self, prepared_tensor):
        prepared_tensor = prepared_tensor.to(self.device)
        with torch.no_grad():
            return self.model(prepared_tensor)

    def predict_last_timestep(self, prepared_tensor):
        output = self.predict(prepared_tensor)
        if output.ndim == 3:
            return output[:, -1, :]
        if output.ndim == 2:
            return output
        raise ValueError(f"Unexpected transformer output shape: {tuple(output.shape)}")
