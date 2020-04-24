# pyNSGP
This Python 3 code is an implementation of multi-objective genetic programming using NSGA-II for symbolic regression.

## Dependencies
Numpy & scikit-learn.

## Installation
Run `pip install --user .` from within the folder.

## Example 
pyNSGP can be run as a scikit-learn regression estimator. See `test.py` for an example. 
The first objective is the mean-squared-error (rescaled by `100/var(y)`), the second is solution size. If `use_interpretability_model=True` is used, then the second objective is implemented by predicting human-interpretability according to the linear model found in the paper referenced below.

## Reference
If you use this code, please support our research by citing the related [paper](https://arxiv.org/abs/2004.11170):
> M. Virgolin, A. De Lorenzo, E. Medvet, F. Randone. "Learning a Formula of Interpretability to Learn Interpretable Formulas". arXiv preprint arXiv:2004.11170 (2020)

For the other part of the code related to this work, see [this repository](https://github.com/MaLeLabTs/GPFormulasInterpretability). 
