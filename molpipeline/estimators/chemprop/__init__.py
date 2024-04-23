"""Initialize Chemprop module."""

import pkgutil

installed_packages = {pkg.name for pkg in pkgutil.iter_modules()}
if "chemprop" in installed_packages:
    from molpipeline.estimators.chemprop.models import (  # noqa: F401
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
