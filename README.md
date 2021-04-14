# pyNSGP
This Python 3 code is an implementation of multi-objective genetic programming using NSGA-II for symbolic regression.

![example use of pyNSGP for symbolic regression](https://github.com/marcovirgolin/pyNSGP/blob/master/example_pic.jpeg?raw=true)

Note: a C++ re-implementation is available in the [GP-GOMEA repo](https://github.com/marcovirgolin/GP-GOMEA).


## Dependencies
Numpy & scikit-learn.

## Installation
Run `pip install --user .` from within the folder.

## Example 
pyNSGP can be run as a scikit-learn regression estimator. See `test.py` for an example. 
The first objective is the mean-squared-error, the second is solution size. If `use_interpretability_model=True` is used, then the second objective is implemented by predicting human-interpretability according to the linear model found in the paper referenced below.

## Reference
If you use this code, please support our research by citing the related [paper](https://doi.org/10.1007/978-3-030-58115-2_6):
> M. Virgolin, A. De Lorenzo, E. Medvet, F. Randone. "Learning a Formula of Interpretability to Learn Interpretable Formulas". Parallel Problem Solving from Nature XVI (2020).

Our preprint is available on [arXiv](https://arxiv.org/abs/2004.11170), and we also made a [video](https://www.youtube.com/watch?v=V2lmbStyMGE&ab_channel=MarcoVirgolin).

For the other part of the code related to this work, see [this repository](https://github.com/MaLeLabTs/GPFormulasInterpretability). 
