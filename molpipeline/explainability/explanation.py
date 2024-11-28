"""Module for explanation class."""

from __future__ import annotations

import abc
import dataclasses

import numpy as np
import numpy.typing as npt

from molpipeline.abstract_pipeline_elements.core import RDKitMol


@dataclasses.dataclass(kw_only=True)
class _AbstractMoleculeExplanation(abc.ABC):
    """Abstract class representing an explanation for a prediction for a molecule."""

    molecule: RDKitMol | None = None
    prediction: npt.NDArray[np.float64] | None = None


@dataclasses.dataclass(kw_only=True)
class FeatureInfoMixin:
    """Mixin providing additional information about the features used in the explanation."""

    feature_vector: npt.NDArray[np.float64] | None = None
    feature_names: list[str] | None = None


@dataclasses.dataclass(kw_only=True)
class FeatureExplanationMixin:
    """Explanation based on feature importance scores, e.g. Shapley Values."""

    # explanation scores for individual features
    feature_weights: npt.NDArray[np.float64] | None = None


@dataclasses.dataclass(kw_only=True)
class AtomExplanationMixin:
    """Atom score based explanation."""

    # explanation scores for individual atoms
    atom_weights: npt.NDArray[np.float64] | None = None


@dataclasses.dataclass(kw_only=True)
class BondExplanationMixin:
    """Bond score based explanation."""

    # explanation scores for individual bonds
    bond_weights: npt.NDArray[np.float64] | None = None


@dataclasses.dataclass(kw_only=True)
class SHAPExplanationMixin:
    """Mixin providing additional information only present in SHAP explanations."""

    expected_value: npt.NDArray[np.float64] | None = None


@dataclasses.dataclass(kw_only=True)
class SHAPFeatureExplanation(
    FeatureInfoMixin,
    FeatureExplanationMixin,
    SHAPExplanationMixin,
    _AbstractMoleculeExplanation,  # base-class should be the last element https://www.ianlewis.org/en/mixins-and-python
):
    """Explanation using feature importance scores from SHAP."""

    def is_valid(self) -> bool:
        """Check if the explanation is valid.

        Returns
        -------
        bool
            True if the explanation is valid, False otherwise.
        """
        return all(
            [
                self.feature_vector is not None,
                self.feature_names is not None,
                self.molecule is not None,
                self.prediction is not None,
                self.feature_weights is not None,
            ]
        )


@dataclasses.dataclass(kw_only=True)
class SHAPFeatureAndAtomExplanation(
    FeatureInfoMixin,
    FeatureExplanationMixin,
    SHAPExplanationMixin,
    AtomExplanationMixin,
    _AbstractMoleculeExplanation,
):
    """Explanation using feature and atom importance scores from SHAP."""

    def is_valid(self) -> bool:
        """Check if the explanation is valid.

        Returns
        -------
        bool
            True if the explanation is valid, False otherwise.
        """
        return all(
            [
                self.feature_vector is not None,
                self.feature_names is not None,
                self.molecule is not None,
                self.prediction is not None,
                self.feature_weights is not None,
                self.atom_weights is not None,
            ]
        )
