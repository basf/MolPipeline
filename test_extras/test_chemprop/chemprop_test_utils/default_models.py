"""Functions for creating default chemprop models."""

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


def get_neural_fp_encoder() -> ChempropNeuralFP:
    """Get the Chemprop model.

    Returns
    -------
    ChempropNeuralFP
        The Chemprop model.
    """
    mpnn = get_binary_classification_mpnn()
    chemprop_model = ChempropNeuralFP(model=mpnn, lightning_trainer__accelerator="cpu")
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
