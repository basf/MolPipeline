"""Explainer classes for explaining predictions."""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import numpy.typing as npt
import shap
from scipy.sparse import issparse, spmatrix

from molpipeline import Pipeline
from molpipeline.abstract_pipeline_elements.core import InvalidInstance, OptionalMol
from molpipeline.explainability.explanation import Explanation
from molpipeline.explainability.fingerprint_utils import fingerprint_shap_to_atomweights
from molpipeline.mol2any import MolToMorganFP
from molpipeline.utils.subpipeline import SubpipelineExtractor
from molpipeline.utils.value_checks import get_length


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
        return feature_matrix.todense()
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

        # create subpipelines for extracting intermediate results for explanations
        (
            self.molecule_reader_subpipeline,
            self.featurization_subpipeline,
            self.model_subpipeline,
        ) = self._extract_subpipelines(model, pipeline, pipeline_extractor)

        if len(self.featurization_subpipeline.steps) > 1:
            raise AssertionError(
                "The featurization subpipeline should only contain one element. Multiple elements are not supported."
            )

    def _extract_subpipelines(
        self, model: Any, pipeline: Pipeline, pipeline_extractor: SubpipelineExtractor
    ) -> tuple[Pipeline, Pipeline, Pipeline]:

        # first extract elements we need the output from
        featurization_element = pipeline_extractor.get_featurization_element()
        if featurization_element is None:
            raise ValueError("Could not determine the featurization element.")

        def get_index(element):
            for idx, step in enumerate(pipeline.steps):
                if id(step[1]) == id(element):
                    return idx
            return None

        featurization_element_idx = get_index(featurization_element)
        if featurization_element_idx is None:
            raise ValueError(
                "Could not determine the index of the featurization element."
            )
        model_element_idx = get_index(model)
        if model_element_idx is None:
            raise ValueError("Could not determine the index of the model element.")

        # reader subpipeline is from step 0 to one before the featurization element
        reader_subpipeline = self.pipeline[:featurization_element_idx]
        featurization_subpipeline = self.pipeline[
            featurization_element_idx:model_element_idx
        ]
        model_subpipeline = self.pipeline[model_element_idx:]
        return reader_subpipeline, featurization_subpipeline, model_subpipeline

    # def _extract_subpipelines(
    #     self, model: Any, pipeline: Pipeline, pipeline_extractor: SubpipelineExtractor
    # ) -> tuple[Pipeline, Pipeline, Pipeline]:
    #     """Extract the subpipelines from the pipeline extractor.
    #
    #     We extract 3 subpipeline. Each subpipeline is an interval of the original pipeline.
    #     1. The first subpipeline is for reading the input to a molecule. The resulting molecules are ready
    #     for featurization, .e.g. it went through standardization steps.
    #     2. The second subpipeline featurizes the molecules to a machine learning ready format.
    #     3. The third subpipeline executes the machine learning inference step, including post-processing.
    #
    #
    #     Parameters
    #     ----------
    #     model : Any
    #         The model element.
    #     pipeline : Pipeline
    #         The pipeline.
    #     pipeline_extractor : SubpipelineExtractor
    #         The pipeline extractor.
    #
    #     Returns
    #     -------
    #     tuple[Pipeline, Pipeline, Pipeline]
    #         The molecule reader, featurization, and prediction subpipelines.
    #     """
    #
    #     # The pipeline in split into subsequent intervals covering the whole pipeline.
    #     # The intervals are defined as:
    #     # 1. Molecule reading subpipeline: from the beginning to the position before the featurization element.
    #     # 2. Featurization subpipeline: from the featurization element to the position before the model element.
    #     # 3. Model subpipeline: from the position after the featurization element to the end of the pipeline.
    #     # This heuristic process needs only to find the featurization element and model element to infer
    #     # the subpipelines.
    #
    #     # first extract elements we need the output from
    #     featurization_element = pipeline_extractor.get_featurization_element()
    #     if featurization_element is None:
    #         raise ValueError("Could not determine the featurization element.")
    #
    #     # reader subpipeline is from step 0 to one before the featurization element
    #     reader_subpipeline = pipeline_extractor.get_subpipeline(
    #         pipeline.steps[0][1], featurization_element, second_offset=-1
    #     )
    #     if reader_subpipeline is None:
    #         raise ValueError("Could not determine the molecule reader subpipeline.")
    #
    #     # the featurization subpipeline is from the featurization element one element before the model element.
    #     featurization_subpipeline = pipeline_extractor.get_subpipeline(
    #         featurization_element, model, second_offset=-1
    #     )
    #     if featurization_subpipeline is None:
    #         raise ValueError("Could not determine the featurization subpipeline.")
    #
    #     # the model subpipeline is from the first element after the featurization element until the end of the pipeline
    #     model_subpipeline = pipeline_extractor.get_subpipeline(
    #         featurization_element, pipeline.steps[-1][1], first_offset=1
    #     )
    #     if model_subpipeline is None:
    #         raise ValueError("Could not determine the model subpipeline.")
    #
    #     return reader_subpipeline, featurization_subpipeline, model_subpipeline

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
        featurization_element = self.featurization_subpipeline.steps[0][1]  # type: ignore

        explanation_results = []
        for input_sample in X:

            input_sample = [input_sample]

            # get the molecule
            molecule_list = self.molecule_reader_subpipeline.transform(input_sample)  # type: ignore
            if len(molecule_list) == 0 or isinstance(molecule_list[0], InvalidInstance):
                explanation_results.append(Explanation())
                continue

            feature_vector = self.featurization_subpipeline.transform(molecule_list)  # type: ignore
            if get_length(feature_vector) == 0 or isinstance(
                feature_vector[0], InvalidInstance
            ):
                explanation_results.append(Explanation())
                continue

            # get predictions
            prediction = _get_predictions(self.model_subpipeline, feature_vector)
            if len(prediction) == 0 or isinstance(prediction[0], InvalidInstance):
                explanation_results.append(Explanation())
                continue

            # todo fill values?

            if prediction.ndim > 1:
                prediction = prediction.squeeze()

            # reshape feature vector for SHAP and output
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
                    molecule_list[0],
                    featurization_element,
                    feature_vector,
                )

            explanation_results.append(
                Explanation(
                    feature_vector=feature_vector,
                    feature_names=feature_names,
                    molecule=molecule_list[0],
                    prediction=prediction,
                    feature_weights=feature_weights,
                    atom_weights=atom_weights,
                    bond_weights=bond_weights,
                )
            )
        return explanation_results
