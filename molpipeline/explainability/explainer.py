"""Explainer classes for explaining predictions."""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import numpy.typing as npt
import shap
from scipy.sparse import issparse, spmatrix

from molpipeline import Pipeline
from molpipeline.abstract_pipeline_elements.core import OptionalMol
from molpipeline.explainability.explanation import Explanation
from molpipeline.explainability.fingerprint_utils import fingerprint_shap_to_atomweights
from molpipeline.mol2any import MolToMorganFP
from molpipeline.utils.subpipeline import SubpipelineExtractor


# pylint: disable=C0103,W0613
def _to_dense(
    feature_matrix: npt.NDArray[Any] | spmatrix,
) -> npt.NDArray[Any]:
    """Mitigate feature incompatibility with SHAP objects.

    Parameters
    ----------
    feature_matrix : npt.NDArray[Any] | spmatrix
        The input features.

    Returns
    -------
    Any
        The input features in a compatible format.
    """
    if issparse(feature_matrix):
        return feature_matrix.todense()  # type: ignore[union-attr]
    return feature_matrix


# This function might also be put at a more central position in the lib.
def _get_predictions(
    pipeline: Pipeline, feature_matrix: npt.NDArray[Any] | spmatrix
) -> npt.NDArray[np.float_]:
    """Get the predictions of a model.

    Raises if no adequate method is found.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline containing the model.
    feature_matrix : Any
        The input data.

    Returns
    -------
    npt.NDArray[np.float_]
        The predictions.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(feature_matrix)
    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function(feature_matrix)
    if hasattr(pipeline, "predict"):
        return pipeline.predict(feature_matrix)
    raise ValueError("Could not determine the model output predictions")


def _convert_shap_feature_weights_to_atom_weights(
    feature_weights: npt.NDArray[np.float_],
    molecule: OptionalMol,
    featurization_element: MolToMorganFP,
    feature_vector: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """Convert SHAP feature weights to atom weights.

    Parameters
    ----------
    feature_weights : npt.NDArray[np.float_]
        The feature weights.
    molecule : OptionalMol
        The molecule.
    featurization_element : MolToMorganFP
        The featurization element.
    feature_vector : npt.NDArray[np.float_]
        The feature vector.

    Returns
    -------
    npt.NDArray[np.float_]
        The atom weights.
    """
    if feature_weights.ndim == 1:
        # regression case
        feature_weights_present_bits_only = feature_weights.copy()
    elif feature_weights.ndim == 2:
        # binary classification case. Take the weights for the positive class.
        feature_weights_present_bits_only = feature_weights[:, 1].copy()
    else:
        raise ValueError(
            "Unsupported number of dimensions for feature weights. Expected 1 or 2."
        )

    # reset shap values for bits that are not present in the molecule
    feature_weights_present_bits_only[feature_vector == 0] = 0

    atom_weights = np.array(
        fingerprint_shap_to_atomweights(
            molecule,
            featurization_element,
            feature_weights_present_bits_only,
        )
    )
    return atom_weights


# pylint: disable=R0903
class AbstractExplainer(abc.ABC):
    """Abstract class for explainer objects."""

    # pylint: disable=C0103,W0613
    @abc.abstractmethod
    def explain(self, X: Any, **kwargs: Any) -> list[Explanation]:
        """Explain the predictions for the input data.

        Parameters
        ----------
        X : Any
            The input data to explain.
        kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        list[Explanation]
            List of explanations corresponding to the input samples.
        """


# pylint: disable=R0903
class SHAPTreeExplainer(AbstractExplainer):
    """Class for SHAP's TreeExplainer wrapper."""

    def __init__(self, pipeline: Pipeline, **kwargs: Any) -> None:
        """Initialize the SHAPTreeExplainer.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline containing the model to explain.
        kwargs : Any
            Additional keyword arguments for SHAP's TreeExplainer.
        """
        self.pipeline = pipeline
        pipeline_extractor = SubpipelineExtractor(self.pipeline)

        # extract the fitted model
        model = pipeline_extractor.get_model_element()
        if model is None:
            raise ValueError("Could not determine the model to explain.")

        # set up the actual explainer
        self.explainer = shap.TreeExplainer(
            model,
            **kwargs,
        )

        # extract the molecule reader subpipeline
        self.molecule_reader_subpipeline = (
            pipeline_extractor.get_molecule_reader_subpipeline()
        )
        if self.molecule_reader_subpipeline is None:
            raise ValueError("Could not determine the molecule reader subpipeline.")

        # extract the featurization subpipeline
        self.featurization_subpipeline = (
            pipeline_extractor.get_featurization_subpipeline()
        )
        if self.featurization_subpipeline is None:
            raise ValueError("Could not determine the featurization subpipeline.")

        # extract fill values for checking error handling
        self.fill_values = pipeline_extractor.get_all_filter_reinserter_fill_values()
        self.fill_values_contain_nan = np.isnan(self.fill_values).any()

    def _prediction_is_valid(self, prediction: Any) -> bool:
        """Check if the prediction is valid using some heuristics.

        Can be used to catch inputs that failed the pipeline for some reason.

        Parameters
        ----------
        prediction : Any
            The prediction.
        Returns
        -------
        bool
            Whether the prediction is valid.
        """
        # if no prediction could be obtained (length is 0); the prediction guaranteed failed.
        if len(prediction) == 0:
            return False

        # if a value in the prediction is a fill-value, we - assume - the explanation has failed.
        if np.isin(prediction, self.fill_values).any():
            return False
        if self.fill_values_contain_nan and np.isnan(prediction).any():
            # the extra nan check is necessary because np.isin does not work with nan
            return False

        return True

    # pylint: disable=C0103,W0613
    def explain(self, X: Any, **kwargs: Any) -> list[Explanation]:
        """Explain the predictions for the input data.

        If the calculation of the SHAP values for an input sample fails, the explanation will be invalid.
        This can be checked with the Explanation.is_valid() method.

        Parameters
        ----------
        X : Any
            The input data to explain.
        kwargs : Any
            Additional keyword arguments for SHAP's TreeExplainer.shap_values.

        Returns
        -------
        list[Explanation]
            List of explanations corresponding to the input data.
        """
        featurization_element = self.featurization_subpipeline.steps[-1][1]  # type: ignore[union-attr]

        explanation_results = []
        for input_sample in X:

            input_sample = [input_sample]

            # get predictions
            prediction = _get_predictions(self.pipeline, input_sample)
            if not self._prediction_is_valid(prediction):
                # we use the prediction to check if the input is valid. If not, we cannot explain it.
                explanation_results.append(Explanation())
                continue

            if prediction.ndim > 1:
                prediction = prediction.squeeze()

            # get the molecule
            molecule = self.molecule_reader_subpipeline.transform(input_sample)[0]  # type: ignore[union-attr]

            # get feature vectors
            feature_vector = self.featurization_subpipeline.transform(input_sample)  # type: ignore[union-attr]
            feature_vector = _to_dense(feature_vector)
            feature_vector = np.asarray(feature_vector).squeeze()

            # Feature names should also be extracted from the Pipeline.
            # But first, we need to add the names to the pipelines.
            # Therefore, feature_names is just None currently.
            feature_names = None

            # compute the shap values for the features
            feature_weights = self.explainer.shap_values(feature_vector, **kwargs)
            feature_weights = np.asarray(feature_weights).squeeze()

            atom_weights = None
            bond_weights = None

            if isinstance(featurization_element, MolToMorganFP):
                # for Morgan fingerprint, we can map the shap values to atom weights
                atom_weights = _convert_shap_feature_weights_to_atom_weights(
                    feature_weights,
                    molecule,
                    featurization_element,
                    feature_vector,
                )

            explanation_results.append(
                Explanation(
                    feature_vector=feature_vector,
                    feature_names=feature_names,
                    molecule=molecule,
                    prediction=prediction,
                    feature_weights=feature_weights,
                    atom_weights=atom_weights,
                    bond_weights=bond_weights,
                )
            )

        return explanation_results
