"""Initialize pipeline_elements package, which includes all rdkit wrappings to sklearn."""
# pylint: disable=no-name-in-module
from rdkit.Chem import PropertyPickleOptions, SetDefaultPickleProperties

# Keep all properties when pickling. Otherwise, we will lose properties set on RDKitMol when passed to
# multiprocessing subprocesses.
SetDefaultPickleProperties(PropertyPickleOptions.AllProps)
