'''Initialize the src package.

This module imports the model and processor from `src.model` and makes them
available as top‑level attributes so that other modules (e.g., `predict.py`) can
simply do:

    from src import processor, model

The import also runs the model loading logic once at package import time,
which is appropriate for a small demo project.
'''
from .model import get_model_and_processor

# Load the model and processor when the package is imported.
model, processor, device = get_model_and_processor()
