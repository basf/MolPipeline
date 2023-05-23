"""Class for combining molepieline and SKLearn Pipeline."""

from __future__ import annotations

import copy
from typing import Any, get_args, Iterable, Optional, overload

import numpy as np
import numpy.typing as npt
from sklearn.base import clone
import warnings

from molpipeline.pipeline import MolPipeline
from molpipeline.utils.none_handling import NoneCollector
from molpipeline.utils.molpipe_types import NoneHandlingOptions
from molpipeline.utils.json_operations import (
    sklearn_model_from_json,
    sklearn_model_to_json,
)


class PipelineModel:
    """Pipeline for combining MolPipeline and Sklearn functionalities."""

    _mol_pipeline: MolPipeline

    def __init__(
        self,
        mol_pipeline: MolPipeline,
        sklearn_model: Any,
        none_handling: NoneHandlingOptions = "raise",
        fill_value: Any = np.nan,
        n_jobs: int = 1,
        handle_nones: Optional[NoneHandlingOptions] = None,
    ) -> None:
        """Initialize the MLPipeline.

        Parameters
        ----------
        mol_pipeline: MolPipeline
            MolPipeline for preprocessing molecules.
        sklearn_model: Any
            Sklearn Model.
        none_handling: Literal["raise", "record_remove", "fill_dummy"]
            Parameter defining the handling of nones.
        fill_value: Any
            If none_handling == "fill_dummy": Mols which are None are substituted with fill_value in
            final output.
        n_jobs: int
            Number of cores used.
        handle_nones: Optional[NoneHandlingOptions]
            For backwards compatibility. If not None, this value is used for none_handling.
        """
        self._mol_pipeline = mol_pipeline
        self._mol_pipeline.none_collector = NoneCollector(fill_value)
        self._skl_model = sklearn_model
        self.n_jobs = n_jobs

        self.none_handling = none_handling
        if handle_nones is not None:
            warnings.warn("handle_nones is deprecated. Use none_handling instead.")
            self.none_handling = handle_nones

    @classmethod
    def from_json(cls, json_dict: dict[str, Any]) -> PipelineModel:
        """Create PipelineModel from json dict.

        Parameters
        ----------
        json_dict: dict[str, Any]
            Json dict containing the information of the PipelineModel.

        Returns
        -------
        PipelineModel
            PipelineModel created from json dict.
        """
        mol_pipeline = MolPipeline.from_json(json_dict["mol_pipeline"])
        skl_model = sklearn_model_from_json(json_dict["skl_model"])
        return cls(
            mol_pipeline, skl_model, json_dict["none_handling"], json_dict["fill_value"]
        )

    @property
    def mol_pipeline(self) -> MolPipeline:
        """Get the mol_pipeline."""
        return self._mol_pipeline

    @property
    def n_jobs(self) -> int:
        """Get number of cores used."""
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs: int) -> None:
        """Set number of cores used.

        Parameters
        ----------
        n_jobs: int
            Number of cores used.
        Returns
        -------
        None
        """
        self._n_jobs = n_jobs
        self._mol_pipeline.n_jobs = n_jobs
        if hasattr(self._skl_model, "n_jobs"):
            self._skl_model.set_params(n_jobs=n_jobs)

    @property
    def none_handling(self) -> NoneHandlingOptions:
        """Get none_handling."""
        return self._none_handling

    @none_handling.setter
    def none_handling(self, none_handling: NoneHandlingOptions) -> None:
        """Set none_handling.

        Parameters
        ----------
        none_handling: NoneHandlingOptions
            Parameter defining the handling of nones.
        Returns
        -------
        None
        """
        if none_handling not in get_args(NoneHandlingOptions):
            raise ValueError(
                f"none_handling must be one of {get_args(NoneHandlingOptions)}, but is {none_handling}."
            )
        self._none_handling = none_handling
        if none_handling == "raise":
            self._mol_pipeline.none_handling = "raise"
        elif none_handling == "record_remove" or none_handling == "fill_dummy":
            self._mol_pipeline.none_handling = "record_remove"
        else:
            raise NotImplementedError(
                f"This is a bug. {none_handling} must be included."
            )

    @property
    def none_indices(self) -> list[int]:
        """Get indices of molecules which are None."""
        return self._mol_pipeline.none_collector.none_indices

    @property
    def _none_collector(self) -> NoneCollector:
        """Get the none_collector."""
        warnings.warn("_none_collector is deprecated. Use none_collector instead.")
        return self._mol_pipeline.none_collector

    @property
    def none_collector(self) -> NoneCollector:
        """Get the none_collector."""
        return self._mol_pipeline.none_collector

    @property
    def ml_model(self) -> Any:
        """Get the ml_model."""
        return self._skl_model

    def _remove_nones(self, value_iterable: Iterable[Any]) -> npt.NDArray[Any]:
        value_array = np.array(list(value_iterable))
        if len(self.none_indices) > 0:
            value_array = np.delete(value_array, self.none_indices)
        return value_array

    @overload
    def _fit_transform_molpipeline(
        self, molecule_iterable: Iterable[Any], y_values: None
    ) -> tuple[Any, None]:
        ...

    @overload
    def _fit_transform_molpipeline(
        self, molecule_iterable: Iterable[Any], y_values: Iterable[Any]
    ) -> tuple[Any, npt.NDArray[np.float_]]:
        ...

    def _fit_transform_molpipeline(
        self, molecule_iterable: Iterable[Any], y_values: Optional[Iterable[Any]]
    ) -> tuple[Any, Optional[npt.NDArray[np.float_]]]:
        """Fit and transform the molpipeline.

        Parameters
        ----------
        molecule_iterable: Iterable[Any]
            Iterable of molecule representations (SMILES, MolBlocks RDKit Molecules, etc.).
                Input depends on the first element of the mol_pipeline.
        y_values: Optional[Iterable[Any]]
            Values expected as output, used for ML training. Not required for unsupervised learning.
        Returns
        -------
        tuple[Any, Optional[npt.NDArray[np.float_]]]
            Tuple of transformed input and raw output. X values which are None and corresponding y values are removed.
        """
        ml_input = self._mol_pipeline.fit_transform(molecule_iterable)
        if len(self.none_indices) > 0 and self.none_handling == "raise":
            raise ValueError(f"Encountered Nones during fit at {self.none_indices}")

        if y_values is not None:
            y_values = self._remove_nones(y_values)

        return ml_input, y_values

    def molpipeline_transform(
        self,
        molecule_iterable: Iterable[Any],
    ) -> Any:
        """Apply molpipeline tranformation to molecules.

        Parameters
        ----------
        molecule_iterable: Iterable[Any]
            Iterable of molecule representations (SMILES, MolBlocks RDKit Molecules, etc.).
            Input depends on the first element of the mol_pipeline.
        Returns
        -------
        Any
            Output of molpipeline transformation without applying the sklearn method.
        """
        ml_input = self._mol_pipeline.transform(molecule_iterable)
        if len(self.none_indices) > 0 and self.none_handling == "raise":
            raise ValueError(f"Encountered Nones during fit at {self.none_indices}")
        return ml_input

    def _finalize_output(
        self, output: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        if len(self.none_indices) > 0:
            if self.none_handling == "fill_dummy":
                output = self.none_collector.fill_with_dummy(output)
        return output

    def fit(
        self,
        molecule_iterable: Iterable[Any],
        y_values: Optional[Iterable[Any]] = None,
        **fitparams: dict[Any, Any],
    ) -> PipelineModel:
        """Fit MLPipeline according to input.

        Parameters
        ----------
        molecule_iterable: Iterable[Any]
            Iterable of molecules.
        y_values: Optional[Iterable[Any]]
            Values expected as output, used for ML training.
        fitparams:  dict[Any, Any]
            Parameter for SKLearn pipeline.

        Returns
        -------
        self
            Fitted PipelineModel.
        """
        # pylint: disable=E0633
        # bug: pylint does not recognize overload
        ml_input, y_values = self._fit_transform_molpipeline(
            molecule_iterable, y_values
        )
        self._skl_model.fit(ml_input, y_values, **fitparams)
        return self

    def fit_transform(
        self,
        molecule_iterable: Iterable[Any],
        y_values: Optional[Iterable[Any]] = None,
        **fitparams: dict[Any, Any],
    ) -> npt.NDArray[np.float_]:
        """Fit MLPipeline according to input and return transformed input.

        Parameters
        ----------
        molecule_iterable: Iterable[Any]
            Iterable of molecules.
        y_values: Iterable[Any]
            Values expected as output, used for ML training.
        fitparams:  dict[Any, Any]
            Parameter for SKLearn pipeline.

        Returns
        -------
        npt.NDArray[Any]
            Transformed input.
        """
        # pylint: disable=E0633
        # bug: pylint does not recognize overload
        ml_input, y_values = self._fit_transform_molpipeline(
            molecule_iterable, y_values
        )
        ml_output = self._skl_model.fit_transform(ml_input, y_values, **fitparams)
        final_output = self._finalize_output(ml_output)
        return final_output

    def fit_predict(
        self,
        molecule_iterable: Iterable[Any],
        y_values: Iterable[Any],
        **fitparams: dict[Any, Any],
    ) -> npt.NDArray[np.float_]:
        """Fit MLPipeline according to input and return predictions for input.

        Parameters
        ----------
        molecule_iterable: Iterable[Any]
            Iterable of molecules.
        y_values: Iterable[Any]
            Values expected as output, used for ML training.
        fitparams:  dict[Any, Any]
            Parameter for SKLearn pipeline.

        Returns
        -------
        npt.NDArray[Any]
            Prediction for input.
        """
        # pylint: disable=E0633
        # bug: pylint does not recognize overload
        ml_input, y_values = self._fit_transform_molpipeline(
            molecule_iterable, y_values
        )
        ml_output = self._skl_model.fit_predict(ml_input, y_values, **fitparams)
        final_output = self._finalize_output(ml_output)
        return final_output

    def to_json(self) -> dict[str, Any]:
        """Transform model parameters to json format.

        Returns
        -------
        dict[str, Any]
            Model parameters in json format.
        """
        return {
            "mol_pipeline": self._mol_pipeline.to_json(),
            "skl_model": sklearn_model_to_json(self._skl_model),
            "fill_value": copy.copy(self.none_collector.fill_value),
            "none_handling": copy.copy(self.none_handling),
        }

    def transform(
        self, molecule_iterable: Iterable[Any], **tranformparams: dict[Any, Any]
    ) -> npt.NDArray[Any]:
        """Transform the input.

        Parameters
        ----------
        molecule_iterable: Iterable[Any]
            Iterable of molecules.
        tranformparams:  dict[Any, Any]
            Parameter for SKLearn pipeline.

        Returns
        -------
        npt.NDArray[Any]
            Output of transformation.
        """
        ml_input = self.molpipeline_transform(molecule_iterable)
        ml_output = self._skl_model.transform(ml_input, **tranformparams)
        final_output = self._finalize_output(ml_output)
        return final_output

    def predict(
        self, molecule_iterable: Iterable[Any], **predictparams: dict[Any, Any]
    ) -> npt.NDArray[Any]:
        """Predict the input.

        Parameters
        ----------
        molecule_iterable: Iterable[Any]
            Iterable of molecules.
        predictparams: dict[Any, Any]
            Parameter for SKLearn pipeline

        Returns
        -------
        npt.NDArray[Any]
            Array of predicted values.
        """
        ml_input = self.molpipeline_transform(molecule_iterable)
        ml_output = self._skl_model.predict(ml_input, **predictparams)
        final_output = self._finalize_output(ml_output)
        return final_output

    def predict_proba(
        self, molecule_iterable: Iterable[Any], **predictparams: dict[Any, Any]
    ) -> npt.NDArray[Any]:
        """Predict the input.

        Parameters
        ----------
        molecule_iterable: Iterable[Any]
            Iterable of molecules.
        predictparams: dict[Any, Any]
            Parameter for SKLearn pipeline

        Returns
        -------
        npt.NDArray[Any]
            Array of predicted values.
        """
        if not hasattr(self._skl_model, "predict_proba"):
            raise AttributeError("Model does not support predict_proba!")
        ml_input = self.molpipeline_transform(molecule_iterable)
        ml_output = self._skl_model.predict_proba(ml_input, **predictparams)
        final_output = self._finalize_output(ml_output)
        return final_output

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep: bool
            If True, create a deep copy of the parmeters.
        Returns
        -------
        dict[str, Any]
            A dictionary of parameter names and corresponding values.
        """
        if deep:
            parameter_dict = {
                "mol_pipeline": self._mol_pipeline.copy(),
                "sklearn_model": clone(self._skl_model),
                "none_handling": str(self.none_handling),
                "fill_value": copy.copy(self.none_collector.fill_value),
            }
            return parameter_dict

        parameter_dict = {
            "mol_pipeline": self._mol_pipeline,
            "sklearn_model": self._skl_model,
            "none_handling": self.none_handling,
            "fill_value": self.none_collector.fill_value,
        }
        return parameter_dict

    def set_params(self, **params: dict[str, Any]) -> PipelineModel:
        """Set the parameters of this estimator.

        Implemented for compatibility with sklearn GridSearchCV.

        Parameters
        ----------
        params: dict[str, Any]
            Dictionary of model parameters.

        Returns
        -------
        PipelineModel
            PipelineModel with updated parameters.
        """
        params = dict(params)

        if "mol_pipeline" in params:
            mol_pipeline = params.pop("mol_pipeline")
            if not isinstance(mol_pipeline, MolPipeline):
                raise TypeError(f"Not a MoleculePipeline: {type(mol_pipeline)}")
            self._mol_pipeline = mol_pipeline

        if "sklearn_model" in params:
            skl_model = params.pop("sklearn_model")
            if not hasattr(skl_model, "set_params"):
                raise TypeError(
                    "Potentially not an SKLearn model, as it does not have the function set_params!"
                )

        if "none_handling" in params:
            value = params.pop("none_handling")
            if value not in get_args(NoneHandlingOptions):
                raise TypeError(f"Invalid selection for NoneHandlingOptions: {value}")
            self.none_handling = value  # type: ignore

        if "fill_value" in params:
            self.none_collector.fill_value = params.pop("fill_value")

        # All remaining parameters are passed to the sklearn model.
        self._skl_model.set_params(**params)
        return self
