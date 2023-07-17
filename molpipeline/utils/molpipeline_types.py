"""Definition of types used in molpipeline."""
from __future__ import annotations
from typing import (
    Any,
    Literal,
    Optional,
    Protocol
)

import numpy.typing as npt


from rdkit.Chem import Mol as RDKitMol  # type: ignore[import]

OptionalMol = Optional[RDKitMol]

NoneHandlingOptions = Literal["raise", "record_remove", "fill_dummy"]


class _AnyPredictor(Protocol):

    def fit_predict(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> npt.NDArray[Any]:
        ...

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> None:
        ...


class _AnyTransformer(Protocol):

    def fit_transform(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> npt.NDArray[Any]:
        ...

    def fit(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> None:
        ...
