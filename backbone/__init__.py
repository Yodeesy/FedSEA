# backbone/__init__.py
import os
import importlib


def get_all_models():
    """
    Scans the 'backbone' directory and returns a list of available model names.
    Ignores __init__.py and non-python files.
    """
    return [
        model.split('.')[0]
        for model in os.listdir('backbone')
        if not model.startswith('__') and model.endswith('.py')
    ]


# Dictionary to map model names to model classes
model_registry = {}

for model_name in get_all_models():
    try:
        mod = importlib.import_module('backbone.' + model_name)

        # Assumption: Class name matches the file name (e.g., GCN.py -> class GCN)
        if hasattr(mod, model_name):
            model_registry[model_name] = getattr(mod, model_name)
    except Exception as e:
        print(f"[Warning] Failed to import backbone model '{model_name}': {e}")


def get_model(model_name, input_dim, hidden_dim, out_dim, num_layers, dropout, **kwargs):
    """
    Factory method to instantiate a GNN backbone model.
    """
    if model_name not in model_registry:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(model_registry.keys())}")

    # Instantiate and return the model
    return model_registry[model_name](
        input_dim,
        hidden_dim,
        out_dim,
        num_layers,
        dropout,
        **kwargs
    )