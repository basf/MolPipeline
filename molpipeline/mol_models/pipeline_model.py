"""Class for combining molepieline and SKLearn Pipeline."""

from __future__ import annotations

import copy
from typing import Any, get_args, Iterable, Literal, Optional, overload

import numpy as np
import numpy.typing as npt
from sklearn.base import clone

from molpipeline.pipeline import MolPipeline
from molpipeline.utils.none_handling import NoneCollector

NoneHandlingOptions = Literal["raise", "record_remove", "fill_dummy"]


class PipelineModel:
    """Pipeline for combining MolPipeline and Sklearn functionalities."""

    _mol_pipeline: MolPipeline

    def __init__(
        self,
        mol_pipeline: MolPipeline,
        sklearn_model: Any,
        handle_nones: NoneHandlingOptions = "raise",
        fill_value: Any = np.nan,
    ) -> None:
        """Initialize the MLPipeline.

        Parameters
        ----------
        mol_pipeline: MolPipeline
            MolPipeline for preprocessing molecules.
        sklearn_model: Any
            Sklearn Model.
        handle_nones: Literal["raise", "record_remove", "fill_dummy"]
            Parameter defining the handling of nones.
        fill_value:
            If handle_nones == "fill_dummy": Mols which are None are substituted with fill_value in
            final output.
        """
        self.handle_nones = handle_nones

        self._mol_pipeline = mol_pipeline
        self._mol_pipeline.handle_nones = "record_remove"
        self._mol_pipeline.none_collector = NoneCollector(fill_value)

        self._skl_model = sklearn_model

    @property
    def none_indices(self) -> list[int]:
        """Get indices of molecules which are None."""
        return self._mol_pipeline.none_collector.none_indices

    @property
    def _none_collector(self) -> NoneCollector:
        """Get the none_collector."""
        return self._mol_pipeline.none_collector

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
        """Fit and transform the molpipeline."""
        ml_input = self._mol_pipeline.fit_transform(molecule_iterable)
        if len(self.none_indices) > 0 and self.handle_nones == "raise":
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
            Iterable of molecules.
        Returns
        -------
        Any
        """
        ml_input = self._mol_pipeline.transform(molecule_iterable)
        if len(self.none_indices) > 0 and self.handle_nones == "raise":
            raise ValueError(f"Encountered Nones during fit at {self.none_indices}")
        return ml_input

    def _finalize_output(
        self, output: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        if len(self.none_indices) > 0:
            if self.handle_nones == "fill_dummy":
                output = self._none_collector.fill_with_dummy(output)
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
        """
        # pylint: disable=E0633
        # bug: pylint does not recognize overload
        ml_input, y_values = self._fit_transform_molpipeline(
            molecule_iterable, y_values
        )
        ml_output = self._skl_model.fit_predict(ml_input, y_values, **fitparams)
        final_output = self._finalize_output(ml_output)
        return final_output

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
        """
        ml_input = self.molpipeline_transform(molecule_iterable)
        ml_output = self._skl_model.predict(ml_input, **predictparams)
        final_output = self._finalize_output(ml_output)
        return final_output

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        if deep:
            parameter_dict = {
                "mol_pipeline": self._mol_pipeline.copy(),
                "sklearn_model": clone(self._skl_model),
                "handle_nones": str(self.handle_nones),
                "fill_value": copy.copy(self._none_collector.fill_value),
            }
            """Get parameters for this estimator."""
            return parameter_dict
        parameter_dict = {
            "mol_pipeline": self._mol_pipeline,
            "sklearn_model": self._skl_model,
            "handle_nones": self.handle_nones,
            "fill_value": self._none_collector.fill_value,
        }
        return parameter_dict

    def set_params(self, **params: dict[str, Any]) -> PipelineModel:
        """Set the parameters of this estimator."""
        if "mol_pipeline" in params:
            mol_pipeline = params.pop("mol_pipeline")
            if not isinstance(mol_pipeline, MolPipeline):
                raise TypeError(f"Not a MoleculePipeline: {type(mol_pipeline)}")
            self._mol_pipeline = mol_pipeline
        if "handle_nones" in params:
            value = params.pop("handle_nones")
            if value not in get_args(NoneHandlingOptions):
                raise TypeError(f"Invalid selection for NoneHandlingOptions: {value}")
            self.handle_nones = value  # type: ignore
        if "fill_value" in params:
            self._none_collector.fill_value = params.pop("fill_value")
        if "sklearn_model" in params:
            skl_model = params.pop("sklearn_model")
            if not hasattr(skl_model, "set_params"):
                raise TypeError(
                    "Potentially not an SKLearn model, as it does not have the function set_params!"
                )
            if params:
                skl_model.set_params(**params)
            self._skl_model = skl_model
        return self
