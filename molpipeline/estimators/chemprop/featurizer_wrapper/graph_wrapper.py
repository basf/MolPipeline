"""Wrapper for Chemprop GraphFeaturizer."""

from dataclasses import InitVar
from typing import Any

try:
    from typing import Self  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Self

from chemprop.featurizers.molgraph import (
    SimpleMoleculeMolGraphFeaturizer as _SimpleMoleculeMolGraphFeaturizer,
)


class SimpleMoleculeMolGraphFeaturizer(_SimpleMoleculeMolGraphFeaturizer):
    """Wrapper for Chemprop SimpleMoleculeMolGraphFeaturizer."""

    extra_atom_fdim: InitVar[int]
    extra_bond_fdim: InitVar[int]

    def get_params(
        self, deep: bool = True  # pylint: disable=unused-argument
    ) -> dict[str, InitVar[int]]:
        """Get parameters for the featurizer.

        Parameters
        ----------
        deep: bool, optional (default=True)
            Used for compatibility with scikit-learn.

        Returns
        -------
        dict[str, int]
            Parameters of the featurizer.
        """
        return {}

    def set_params(self, **parameters: Any) -> Self:  # pylint: disable=unused-argument
        """Set the parameters of the featurizer.

        Parameters
        ----------
        parameters: Any
            Parameters to set. Only used for compatibility with scikit-learn.

        Returns
        -------
        Self
            This featurizer with the parameters set.
        """
        return self
