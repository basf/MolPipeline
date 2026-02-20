"""Functions for creating default chemprop models."""

from typing import Any

import numpy as np
from lightning import pytorch as pl

from molpipeline.any2mol import SmilesToMol
from molpipeline.error_handling import ErrorFilter, FilterReinserter
from molpipeline.estimators.chemprop.component_wrapper import (
    MPNN,
    BinaryClassificationFFN,
    BondMessagePassing,
    SumAggregation,
)
from molpipeline.estimators.chemprop.models import (
    ChempropClassifier,
    ChempropModel,
    ChempropMulticlassClassifier,
    ChempropRegressor,
)
from molpipeline.estimators.chemprop.neural_fingerprint import ChempropNeuralFP
from molpipeline.mol2any import MolToSmiles
from molpipeline.mol2any.mol2chemprop import MolToChemprop
from molpipeline.pipeline import Pipeline
from molpipeline.post_prediction import PostPredictionWrapper


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
    return MPNN(
        message_passing=bond_message_passing,
        agg=aggregate,
        predictor=binary_clf_ffn,
    )


def get_neural_fp_encoder(
    init_kwargs: dict[str, Any] | None = None,
) -> ChempropNeuralFP:
    """Get the Chemprop model.

    Parameters
    ----------
    init_kwargs : dict[str, Any], optional
        Additional keyword arguments to pass to ChempropNeuralFP during initialization.

    Returns
    -------
    ChempropNeuralFP
        The Chemprop model.

    """
    mpnn = get_binary_classification_mpnn()
    init_kwargs = init_kwargs or {}
    return ChempropNeuralFP(
        model=mpnn,
        lightning_trainer__accelerator="cpu",
        **init_kwargs,
    )


def get_chemprop_model_binary_classification_mpnn() -> ChempropModel:
    """Get the Chemprop model.

    Returns
    -------
    ChempropModel
        The Chemprop model.

    """
    mpnn = get_binary_classification_mpnn()
    return ChempropModel(model=mpnn, lightning_trainer__accelerator="cpu")


def get_smiles_checker_pipeline() -> Pipeline:
    """Get a pipeline that reads and writes the SMILES string.

    Invalid SMILES strings are replaced with NaN.

    Returns
    -------
    Pipeline
        The pipeline that reads and writes the SMILES string.

    """
    smiles2mol = SmilesToMol()
    mol2smiles = MolToSmiles()
    error_filter = ErrorFilter(filter_everything=True)
    filter_reinserter = FilterReinserter.from_error_filter(
        error_filter,
        fill_value=np.nan,
    )
    return Pipeline(
        [
            ("smiles2mol", smiles2mol),
            ("error_filter", error_filter),
            ("mol2smiles", mol2smiles),
            ("filter_reinserter", filter_reinserter),
        ],
    )


# pylint: disable=duplicate-code
def get_model() -> ChempropModel:
    """Get the Chemprop model.

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
    return ChempropModel(model=mpnn)


def get_model_pipeline() -> Pipeline:
    """Get the Chemprop model pipeline.

    Returns
    -------
    Pipeline
        The Chemprop model pipeline.

    """
    chemprop_model = get_model()
    mol2chemprop = MolToChemprop()
    smiles2mol = SmilesToMol()
    return Pipeline(
        steps=[
            ("smiles2mol", smiles2mol),
            ("mol2chemprop", mol2chemprop),
            ("model", chemprop_model),
        ],
    )


DEFAULT_TRAINER = pl.Trainer(
    accelerator="cpu",
    logger=False,
    enable_checkpointing=False,
    max_epochs=5,
    enable_model_summary=False,
    enable_progress_bar=False,
    val_check_interval=0.0,
)


def get_regression_pipeline(n_tasks: int = 1) -> Pipeline:
    """Get the Chemprop model pipeline for regression.

    Parameters
    ----------
    n_tasks : int
        The number of tasks for model initialization, i.e. number of target variables.

    Returns
    -------
    Pipeline
        The Chemprop model pipeline for regression.

    """
    smiles2mol = SmilesToMol()
    mol2chemprop = MolToChemprop()
    error_filter = ErrorFilter(filter_everything=True)
    filter_reinserter = FilterReinserter.from_error_filter(
        error_filter,
        fill_value=np.nan,
    )
    chemprop_model = ChempropRegressor(
        lightning_trainer=DEFAULT_TRAINER,
        n_tasks=n_tasks,
    )
    return Pipeline(
        steps=[
            ("smiles2mol", smiles2mol),
            ("mol2chemprop", mol2chemprop),
            ("error_filter", error_filter),
            ("model", chemprop_model),
            ("filter_reinserter", PostPredictionWrapper(filter_reinserter)),
        ],
    )


def get_classification_pipeline(
    chemprop_kwargs: dict[str, Any] | None = None,
) -> Pipeline:
    """Get the Chemprop model pipeline for classification.

    Parameters
    ----------
    chemprop_kwargs : dict[str, Any] | None, optional
        Additional keyword arguments to pass to the ChempropClassifier during
        initialization.

    Returns
    -------
    Pipeline
        The Chemprop model pipeline for classification.

    """
    chemprop_kwargs = chemprop_kwargs or {}
    smiles2mol = SmilesToMol()
    mol2chemprop = MolToChemprop()
    error_filter = ErrorFilter(filter_everything=True)
    filter_reinserter = FilterReinserter.from_error_filter(
        error_filter,
        fill_value=np.nan,
    )
    chemprop_model = ChempropClassifier(
        lightning_trainer=DEFAULT_TRAINER,
        **chemprop_kwargs,
    )
    return Pipeline(
        steps=[
            ("smiles2mol", smiles2mol),
            ("mol2chemprop", mol2chemprop),
            ("error_filter", error_filter),
            ("model", chemprop_model),
            ("filter_reinserter", PostPredictionWrapper(filter_reinserter)),
        ],
    )


def get_multiclass_classification_pipeline(n_classes: int) -> Pipeline:
    """Get the Chemprop model pipeline for multiclass classification.

    Parameters
    ----------
    n_classes : int
        The number of classes for model initialization.

    Returns
    -------
    Pipeline
        The Chemprop model pipeline for multiclass classification.

    """
    smiles2mol = SmilesToMol()
    mol2chemprop = MolToChemprop()
    error_filter = ErrorFilter(filter_everything=True)
    filter_reinserter = FilterReinserter.from_error_filter(
        error_filter,
        fill_value=np.nan,
    )
    chemprop_model = ChempropMulticlassClassifier(
        n_classes=n_classes,
        lightning_trainer=DEFAULT_TRAINER,
    )
    return Pipeline(
        steps=[
            ("smiles2mol", smiles2mol),
            ("mol2chemprop", mol2chemprop),
            ("error_filter", error_filter),
            ("model", chemprop_model),
            ("filter_reinserter", PostPredictionWrapper(filter_reinserter)),
        ],
    )
