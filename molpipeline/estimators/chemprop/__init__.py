"""Initialize Chemprop module."""

import importlib.util

if importlib.util.find_spec("chemprop") is not None:
    from molpipeline.estimators.chemprop.models import (
        ChempropClassifier,
        ChempropModel,
        ChempropNeuralFP,
        ChempropRegressor,
    )

    __all__ = [
        "ChempropClassifier",
        "ChempropModel",
        "ChempropNeuralFP",
        "ChempropRegressor",
    ]
else:
    __all__ = []
