"""Explainer classes for explaining predictions."""

from __future__ import annotations

import abc
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from scipy.sparse import issparse, spmatrix
from sklearn.base import BaseEstimator

try:
    from typing import override  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import override

from molpipeline import Pipeline
from molpipeline.abstract_pipeline_elements.core import OptionalMol
from molpipeline.experimental.explainability.explanation import (
    AtomExplanationMixin,
    BondExplanationMixin,
    FeatureExplanationMixin,
    FeatureInfoMixin,
    SHAPExplanationMixin,
    SHAPFeatureAndAtomExplanation,
    SHAPFeatureExplanation,
)
from molpipeline.experimental.explainability.fingerprint_utils import (
    fingerprint_shap_to_atomweights,
)
from molpipeline.mol2any import MolToMorganFP
from molpipeline.utils.subpipeline import SubpipelineExtractor, get_model_from_pipeline


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


def _get_prediction_function(
    pipeline: Pipeline | BaseEstimator,
) -> Callable[[npt.ArrayLike], npt.ArrayLike]:
    """Get the prediction function of a model.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline containing the model.

    Returns
    -------
    Any
        The prediction function.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba
    if hasattr(pipeline, "decision_function"):
        return pipeline.decision_function
    if hasattr(pipeline, "predict"):
        return pipeline.predict
    raise ValueError("Could not determine the model output predictions")


# This function might also be put at a more central position in the lib.
def _get_predictions(
    pipeline: Pipeline, feature_matrix: npt.NDArray[Any] | spmatrix
) -> npt.NDArray[np.float64]:
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
    npt.NDArray[np.float64]
        The predictions.
    """
    prediction_function = _get_prediction_function(pipeline)
    prediction = prediction_function(feature_matrix)
    return np.array(prediction)


def _convert_shap_feature_weights_to_atom_weights(
    feature_weights: npt.NDArray[np.float64],
    molecule: OptionalMol,
    featurization_element: MolToMorganFP,
    feature_vector: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert SHAP feature weights to atom weights.

    Parameters
    ----------
    feature_weights : npt.NDArray[np.float64]
        The feature weights.
    molecule : OptionalMol
        The molecule.
    featurization_element : MolToMorganFP
        The featurization element.
    feature_vector : npt.NDArray[np.float64]
        The feature vector.

    Returns
    -------
    npt.NDArray[np.float64]
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


class AbstractSHAPExplainer(abc.ABC):  # pylint: disable=too-few-public-methods
    """Abstract class for SHAP explainer objects."""

    @abc.abstractmethod
    def explain(
        self, X: Any, **kwargs: Any  # pylint: disable=invalid-name,unused-argument
    ) -> list[SHAPFeatureExplanation | SHAPFeatureAndAtomExplanation]:
        """Explain the predictions for the input data.

        Parameters
        ----------
        X : Any
            The input data to explain.
        kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        list[SHAPFeatureExplanation | SHAPFeatureAndAtomExplanation]
            List of explanations corresponding to the input samples.
        """


class SHAPExplainerAdapter(
    AbstractSHAPExplainer, abc.ABC
):  # pylint: disable=too-few-public-methods
    """Adapter for SHAP explainer wrappers for handling molecules and pipelines."""

    # used for dynamically defining the return type of the explain method
    return_element_type_: type[SHAPFeatureExplanation | SHAPFeatureAndAtomExplanation]

    def __init__(
        self,
        pipeline: Pipeline,
        explainer: shap.TreeExplainer | shap.KernelExplainer,
    ) -> None:
        """Initialize the SHAPTreeExplainer.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline containing the model to explain.
        explainer : shap.TreeExplainer | shap.KernelExplainer
            The shap explainer object.
        """
        self.pipeline = pipeline
        self.explainer = explainer

        pipeline_extractor = SubpipelineExtractor(self.pipeline)

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

        # determine type of returned explanation
        featurization_element = self.featurization_subpipeline.steps[-1][1]  # type: ignore[union-attr]
        if isinstance(featurization_element, MolToMorganFP):
            self.return_element_type_ = SHAPFeatureAndAtomExplanation
        else:
            self.return_element_type_ = SHAPFeatureExplanation

    @staticmethod
    def _prediction_is_valid(prediction: Any) -> bool:
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

        # use pandas.isna function to check for invalid predictions, e.g. None, np.nan,
        # pd.NA. Note that fill values like 0 will be considered as valid predictions.
        if pd.isna(prediction).any():
            return False

        return True

    @override
    def explain(
        self, X: Any, **kwargs: Any  # pylint: disable=invalid-name,unused-argument
    ) -> list[SHAPFeatureExplanation | SHAPFeatureAndAtomExplanation]:
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
        list[SHAPFeatureExplanation | SHAPFeatureAndAtomExplanation]
            List of explanations corresponding to the input data.
        """
        featurization_element = self.featurization_subpipeline.steps[-1][1]  # type: ignore[union-attr]

        explanation_results: list[
            SHAPFeatureExplanation | SHAPFeatureAndAtomExplanation
        ] = []
        for input_sample in X:

            input_sample = [input_sample]

            # get predictions
            prediction = _get_predictions(self.pipeline, input_sample)
            if not self._prediction_is_valid(prediction):
                # we use the prediction to check if the input is valid. If not, we cannot explain it.
                explanation_results.append(self.return_element_type_())
                continue

            if prediction.ndim > 1:
                prediction = prediction.squeeze()

            # get the molecule
            molecule = self.molecule_reader_subpipeline.transform(input_sample)[0]  # type: ignore[union-attr]

            # get feature vectors
            feature_vector = self.featurization_subpipeline.transform(input_sample)  # type: ignore[union-attr]
            feature_vector = _to_dense(feature_vector)
            feature_vector = np.asarray(feature_vector).squeeze()

            if feature_vector.size == 0:
                # if the feature vector is empty, we cannot explain the prediction.
                # This happens for failed instances in pipeline with fill values
                # that could be valid predictions, like 0.
                explanation_results.append(self.return_element_type_())
                continue

            # compute the shap values for the features
            feature_weights = self.explainer.shap_values(feature_vector, **kwargs)
            feature_weights = np.asarray(feature_weights).squeeze()

            atom_weights = None
            bond_weights = None

            if issubclass(
                self.return_element_type_, AtomExplanationMixin
            ) and isinstance(featurization_element, MolToMorganFP):
                # for Morgan fingerprint, we can map the shap values to atom weights
                atom_weights = _convert_shap_feature_weights_to_atom_weights(
                    feature_weights,
                    molecule,
                    featurization_element,
                    feature_vector,
                )

            # gather all input data for the explanation type to be returned
            explanation_data = {
                "molecule": molecule,
                "prediction": prediction,
            }
            if issubclass(self.return_element_type_, FeatureInfoMixin):
                explanation_data["feature_vector"] = feature_vector
                if not hasattr(featurization_element, "feature_names"):
                    raise ValueError(
                        "Featurization element does not have a get_feature_names method."
                    )
                explanation_data["feature_names"] = featurization_element.feature_names  # type: ignore[union-attr]

            if issubclass(self.return_element_type_, FeatureExplanationMixin):
                explanation_data["feature_weights"] = feature_weights
            if issubclass(self.return_element_type_, AtomExplanationMixin):
                explanation_data["atom_weights"] = atom_weights
            if issubclass(self.return_element_type_, BondExplanationMixin):
                explanation_data["bond_weights"] = bond_weights
            if issubclass(self.return_element_type_, SHAPExplanationMixin):
                explanation_data["expected_value"] = np.atleast_1d(
                    self.explainer.expected_value
                )

            explanation_results.append(self.return_element_type_(**explanation_data))

        return explanation_results


class SHAPTreeExplainer(SHAPExplainerAdapter):  # pylint: disable=too-few-public-methods
    """Wrapper for SHAP's TreeExplainer that can handle pipelines and molecules.

    Wraps SHAP's TreeExplainer to explain predictions of a pipeline containing a
    tree-based model.

    Note on failed instances:
    SHAPTreeExplainer will automatically handle fill values for failed instances and
    returns an invalid explanation for them. However, fill values that could be valid
    predictions, e.g. 0, are not necessarily detected. Set the fill value to np.nan or
    None if these failed instances should not be explained.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        **kwargs: Any,
    ) -> None:
        """Initialize the SHAPKernelExplainer.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline containing the model to explain.
        kwargs : Any
            Additional keyword arguments for SHAP's Explainer.
        """
        explainer = self._create_explainer(pipeline, **kwargs)
        super().__init__(pipeline, explainer)

    @staticmethod
    def _create_explainer(pipeline: Pipeline, **kwargs: Any) -> shap.TreeExplainer:
        """Create the TreeExplainer object from shap.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline containing the model to explain.
        kwargs : Any
            Additional keyword arguments for the explainer.

        Returns
        -------
        shap.TreeExplainer
            The explainer object.
        """
        model = get_model_from_pipeline(pipeline, raise_not_found=True)
        explainer = shap.TreeExplainer(
            model,
            **kwargs,
        )
        return explainer


class SHAPKernelExplainer(
    SHAPExplainerAdapter
):  # pylint: disable=too-few-public-methods
    """Wrapper for SHAP's KernelExplainer that can handle pipelines and molecules."""

    def __init__(
        self,
        pipeline: Pipeline,
        **kwargs: Any,
    ) -> None:
        """Initialize the SHAPKernelExplainer.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline containing the model to explain.
        kwargs : Any
            Additional keyword arguments for SHAP's Explainer.
        """
        explainer = self._create_explainer(pipeline, **kwargs)
        super().__init__(pipeline, explainer)

    @staticmethod
    def _create_explainer(pipeline: Pipeline, **kwargs: Any) -> shap.KernelExplainer:
        """Create the explainer object.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline containing the model to explain.
        kwargs : Any
            Additional keyword arguments for the explainer.

        Returns
        -------
        shap.KernelExplainer
            The explainer object.
        """
        model = get_model_from_pipeline(pipeline, raise_not_found=True)
        prediction_function = _get_prediction_function(model)
        explainer = shap.KernelExplainer(
            prediction_function,
            **kwargs,
        )
        return explainer
