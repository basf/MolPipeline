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
        return {
            "extra_atom_fdim": self.extra_atom_fdim,
            "extra_bond_fdim": self.extra_bond_fdim,
        }

    def set_params(self, **parameters: Any) -> Self:
        """Set the parameters of the featurizer.

        Parameters
        ----------
        parameters: Any
            Parameters to set.

        Returns
        -------
        Self
            This featurizer with the parameters set.
        """
        if "extra_atom_fdim" in parameters:
            self.atom_fdim -= self.extra_atom_fdim
            self.extra_atom_fdim = parameters["extra_atom_fdim"]
            self.atom_fdim += self.extra_atom_fdim
        if "extra_bond_fdim" in parameters:
            self.bond_fdim -= self.extra_bond_fdim
            self.extra_bond_fdim = parameters["extra_bond_fdim"]
            self.bond_fdim += self.extra_bond_fdim
        return self
