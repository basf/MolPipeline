"""Functions for creating default chemprop models."""

from typing import Any

from molpipeline.estimators.chemprop import ChempropModel, ChempropNeuralFP
from molpipeline.estimators.chemprop.component_wrapper import (
    MPNN,
    BinaryClassificationFFN,
    BondMessagePassing,
    SumAggregation,
)


def get_binary_classification_mpnn() -> MPNN:
    """Get a Chemprop model for binary classification.

    Returns
    -------
    ChempropModel
        The Chemprop model.
    """
    binary_clf_ffn = BinaryClassificationFFN()
    aggregate = SumAggregation()
    bond_message_passing = BondMessagePassing()
    mpnn = MPNN(
        message_passing=bond_message_passing,
        agg=aggregate,
        predictor=binary_clf_ffn,
    )
    return mpnn


def get_neural_fp_encoder(
    init_kwargs: dict[str, Any] | None = None,
) -> ChempropNeuralFP:
    """Get the Chemprop model.

    Parameters
    ----------
    init_kwargs : dict[str, Any], optional
        Additional keyword arguments to pass to `ChempropNeuralFP` during initialization.

    Returns
    -------
    ChempropNeuralFP
        The Chemprop model.
    """
    mpnn = get_binary_classification_mpnn()
    init_kwargs = init_kwargs or {}
    chemprop_model = ChempropNeuralFP(
        model=mpnn, lightning_trainer__accelerator="cpu", **init_kwargs
    )
    return chemprop_model


def get_chemprop_model_binary_classification_mpnn() -> ChempropModel:
    """Get the Chemprop model.

    Returns
    -------
    ChempropModel
        The Chemprop model.
    """
    mpnn = get_binary_classification_mpnn()
    chemprop_model = ChempropModel(model=mpnn, lightning_trainer__accelerator="cpu")
    return chemprop_model
