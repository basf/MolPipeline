"""Init."""

from rdkit.Chem import SetDefaultPickleProperties, PropertyPickleOptions

# Keep all properties when pickling. Otherwise we will lose properties set on RDKitMol when passed to
# multiprocessing subprocesses.
SetDefaultPickleProperties(PropertyPickleOptions.AllProps)
