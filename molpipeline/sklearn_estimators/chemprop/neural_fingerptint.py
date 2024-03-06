"""Wrap Chemprop in a sklearn like transformer returning the neural fingerprint as a numpy array."""

from molpipeline.sklearn_estimators.chemprop import ABCChemprop


class ChempropNeuralFP(ABCChemprop):
    """Wrap Chemprop in a sklearn like transformer returning the neural fingerprint as a numpy array."""
