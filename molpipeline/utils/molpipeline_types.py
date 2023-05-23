"""Definition of types used in molpipeline."""

from typing import Literal, Optional

from rdkit.Chem import Mol as RDKitMol  # type: ignore[import]

OptionalMol = Optional[RDKitMol]

NoneHandlingOptions = Literal["raise", "record_remove", "fill_dummy"]
