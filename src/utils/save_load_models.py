import joblib

from ..types.saved_model import SavedModel

def save_model(model: SavedModel, model_name: str) -> None:
    joblib.dump(model, model_name)

def load_model(model_name: str) -> SavedModel:
    model = joblib.load(model_name)
    return model