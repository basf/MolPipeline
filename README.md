# MolPipeline
MolPipeline is a Python package providing RDKit functionality in a Scikit-learn like fashion.

## Background

The open-source package [scikit-learn](https://scikit-learn.org/) provides a large variety of machine
learning algorithms and data processing tools, among which is the `Pipeline` class, allowing users to
prepend custom data processing steps to the machine learning model.
`MolPipeline` extends this concept to the field of chemoinformatics by
wrapping default functionalities of [RDKit](https://www.rdkit.org/), such as reading and writing SMILES strings
or calculating molecular descriptors from a molecule-object.

A notable difference to the `Pipeline` class of scikit-learn is that the Pipline from `MolPipeline` allows for 
instances to fail during processing without interrupting the whole pipeline.
Such behaviour is useful when processing large datasets, where some SMILES strings might not encode valid molecules
or some descriptors might not be calculable for certain molecules.


## Publications

TODO

## Installation
```commandline
pip install molpipeline
```

## Usage

See the [notebooks](notebooks) folder for basic and advanced examples of how to use Molpipeline.

A basic example of how to use MolPipeline to create a fingerprint-based model is shown below (see also the [notebook](notebooks/01_getting_started_with_molpipeline.ipynb)): 
```python
from molpipeline import Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.mol2any import MolToMorganFP
from molpipeline.mol2mol import (
    ElementFilter,
    SaltRemover,
)

from sklearn.ensemble import RandomForestRegressor

# set up pipeline
pipeline = Pipeline([
      ("auto2mol", AutoToMol()),                                     # reading molecules
      ("element_filter", ElementFilter()),                           # standardization
      ("salt_remover", SaltRemover()),                               # standardization
      ("morgan2_2048", MolToMorganFP(n_bits=2048, radius=2)),        # fingerprints and featurization
      ("RandomForestRegressor", RandomForestRegressor())             # machine learning model
    ],
    n_jobs=4)

# fit the pipeline
pipeline.fit(X=["CCCCCC", "c1ccccc1"], y=[0.2, 0.4])
# make predictions from SMILES strings
pipeline.predict(["CCC"])
# output: array([0.29])
```

Molpipeline also provides custom estimators for standard cheminformatics tasks that can be integrated into pipelines,
like clustering for scaffold splits (see also the [notebook](notebooks/02_scaffold_split_with_custom_estimators.ipynb)):

```python
from molpipeline.estimators import MurckoScaffoldClustering

scaffold_smiles = [
    "Nc1ccccc1",
    "Cc1cc(Oc2nccc(CCC)c2)ccc1",
    "c1ccccc1",
]
linear_smiles = ["CC", "CCC", "CCCN"]

# run the scaffold clustering
scaffold_clustering = MurckoScaffoldClustering(
    make_generic=False, linear_molecules_strategy="own_cluster", n_jobs=16
)
scaffold_clustering.fit_predict(scaffold_smiles + linear_smiles)
# output: array([1., 0., 1., 2., 2., 2.])
```


## License

TODO