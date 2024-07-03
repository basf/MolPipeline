"""Classes for encoding molecules as phys-chem vector."""

# pylint: disable=too-many-arguments

from __future__ import annotations

import warnings
from typing import Any, Iterable, Optional

import numpy as np
import numpy.typing as npt

try:
    from chemprop.data import MoleculeDatapoint, MoleculeDataset
    from chemprop.featurizers.base import GraphFeaturizer, VectorFeaturizer

    from molpipeline.estimators.chemprop.featurizer_wrapper.graph_wrapper import (
        SimpleMoleculeMolGraphFeaturizer,
    )
except ImportError:
    warnings.warn(
        "chemprop not installed. MolToChemprop will not work.",
        ImportWarning,
    )


from molpipeline.abstract_pipeline_elements.core import MolToAnyPipelineElement
from molpipeline.utils.molpipeline_types import RDKitMol


class MolToChemprop(MolToAnyPipelineElement):
    """PipelineElement for creating a graph representation used as input for Chemprop models.

    Each molecule is transformed to a MoleculeDatapoint object, which are then assembled into a MoleculeDataset.
    The MoleculeDataset can be used as input for Chemprop models.[1]

    References
    ----------
    [1] https://github.com/chemprop/chemprop/
    """

    graph_featurizer: GraphFeaturizer[RDKitMol] | None
    mol_featurizer: VectorFeaturizer[RDKitMol] | None

    def __init__(
        self,
        graph_featurizer: GraphFeaturizer[RDKitMol] | None = None,
        mol_featurizer: VectorFeaturizer[RDKitMol] | None = None,
        name: str = "Mol2Chemprop",
        n_jobs: int = 1,
        uuid: Optional[str] = None,
    ) -> None:
        """Initialize MolToChemprop.

        Parameters
        ----------
        graph_featurizer: GraphFeaturizer[RDKitMol] | None, optional (default=None)
            Defines how the graph is featurized. Defaults to None.
        mol_featurizer: MoleculeFeaturizer | None, optional (default=None)
            In contrast to graph_featurizer, features from the mol_featurizer are not used during the message passing.
            These features are concatenated to the neural fingerprints before the feedforward layers.
        name: str, optional (default="Mol2Chemprop")
            Name of the pipeline element. Defaults to "Mol2Chemprop".
        n_jobs: int
            Number of parallel jobs to use. Defaults to 1.
        uuid: str | None, optional (default=None)
            UUID of the pipeline element.
        """
        self.graph_featurizer = graph_featurizer or SimpleMoleculeMolGraphFeaturizer()
        self.mol_featurizer = mol_featurizer

        super().__init__(
            name=name,
            n_jobs=n_jobs,
            uuid=uuid,
        )

    def pretransform_single(self, value: RDKitMol) -> MoleculeDatapoint:
        """Transform a single molecule to a ChemProp MoleculeDataPoint.

        Parameters
        ----------
        value: RDKitMol
            RDKit molecule to transform.

        Returns
        -------
        MoleculeDatapoint
            Molecular representation used as input for ChemProp. None if transformation failed.
        """
        mol_features: npt.NDArray[np.float64] | None = None
        if self.mol_featurizer is not None:
            mol_features = np.array(self.mol_featurizer(value), dtype=np.float64)
        return MoleculeDatapoint(mol=value, x_d=mol_features)

    def assemble_output(
        self, value_list: Iterable[MoleculeDatapoint]
    ) -> MoleculeDataset:
        """Assemble the output from the parallelized pretransform_single.

        Parameters
        ----------
        value_list: Iterable
            List of transformed values.

        Returns
        -------
        Any
            Assembled output.
        """
        return MoleculeDataset(data=list(value_list), featurizer=self.graph_featurizer)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this pipeline element.

        Parameters
        ----------
        deep: bool, optional (default=True)
            If True, will return the parameters for this pipeline element and its subobjects.

        Returns
        -------
        dict
            Parameters of this pipeline element.
        """
        params = super().get_params(deep=deep)
        params["graph_featurizer"] = self.graph_featurizer
        params["mol_featurizer"] = self.mol_featurizer

        if deep:
            if hasattr(self.graph_featurizer, "get_params"):
                graph_featurizer_params = self.graph_featurizer.get_params(deep=deep)  # type: ignore
                for key, value in graph_featurizer_params.items():
                    params[f"graph_featurizer__{key}"] = value
            if hasattr(self.mol_featurizer, "get_params"):
                mol_featurizer_params = self.mol_featurizer.get_params(deep=deep)  # type: ignore
                for key, value in mol_featurizer_params.items():
                    params[f"mol_featurizer__{key}"] = value
        return params

    def set_params(self, **parameters: Any) -> MolToChemprop:
        """Set the parameters of this pipeline element.

        Parameters
        ----------
        **parameters: Any
            Parameters to set.

        Returns
        -------
        MolToChemprop
            This pipeline element with the parameters set.
        """
        param_copy = dict(parameters)
        graph_featurizer = param_copy.pop("graph_featurizer", None)
        mol_featurizer = param_copy.pop("mol_featurizer", None)
        if graph_featurizer is not None:
            self.graph_featurizer = graph_featurizer
        if mol_featurizer is not None:
            self.mol_featurizer = mol_featurizer
        graph_featurizer_params = {}
        mol_featurizer_params = {}
        for key in list(param_copy.keys()):
            if "__" not in key:
                continue
            component_name, _, param_name = key.partition("__")
            if component_name == "graph_featurizer":
                graph_featurizer_params[param_name] = param_copy.pop(key)
            elif component_name == "mol_featurizer":
                mol_featurizer_params[param_name] = param_copy.pop(key)
        if hasattr(self.graph_featurizer, "set_params"):
            self.graph_featurizer.set_params(**graph_featurizer_params)  # type: ignore
        if hasattr(self.mol_featurizer, "set_params"):
            self.mol_featurizer.set_params(**mol_featurizer_params)  # type: ignore

        super().set_params(**param_copy)
        return self
