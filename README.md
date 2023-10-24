# MolPipeline
This package provides rdkit functionality in a scikit-learn like fashion.

The open-source package [scikit-learn](https://scikit-learn.org/) provides a large variety of machine
learning algorithms and data processing tools, among which is the `Pipeline` class, allowing users to
prepend custom data processing steps to the machine learning model.
The herein introduced package `MolPipeline` extends this concept to the field of chemoinformatics by
wrapping default functionalities of [RDKit](https://www.rdkit.org/), such as reading and writing SMILES strings
or calculating molecular descriptors from a molecule-object.

A notable difference to the `Pipeline` class of scikit-learn is that the Pipline from `MolPipeline` allowes for 
instances to fail during processing without interrupting the whole pipeline.
Such behaviour is useful when processing large datasets, where some SMILES strings might not encode valid molecules
or some descriptors might not be calculable for certain molecules.


## Installation
```commandline
pip install -e git+https:\\github.com\cheminformaticsbasf\molpipeline.git
```
### Structure for paper
- Introduction
- Implementation
  - Wrapped Functionallities
  - Code structure
  - Parallelization
  - Error handling
- Results
  - SMILES Standardization
  - Group based splits
    - Random
    - Scaffold
    - Clustering
  - Exemplary use of ML Model with MolPipeline
