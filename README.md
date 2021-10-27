# pyNSGP
This Python 3 code is an implementation of multi-objective genetic programming using NSGA-II for symbolic regression.

![example use of pyNSGP for symbolic regression](https://github.com/marcovirgolin/pyNSGP/blob/master/example_pic.jpeg?raw=true)

Note: a C++ re-implementation is available in the [GP-GOMEA repo](https://github.com/marcovirgolin/GP-GOMEA).


### Note
Added penalization of duplicates (from ref. [2]) to better preserve diversity (left img: penalization ON, right img: penalization OFF):
![example run with penalization ON and OFF](https://github.com/marcovirgolin/pyNSGP/blob/master/penalize_duplicates.png?raw=true)

## Dependencies
Numpy & scikit-learn.

## Installation
Run `pip install --user .` from within the folder.

## Example 
pyNSGP can be run as a scikit-learn regression estimator. See `test.py` for an example. 
The first objective is the mean-squared-error, the second is solution size. If `use_interpretability_model=True` is used, then the second objective is implemented by predicting human-interpretability according to the linear model found in the paper referenced below.

## Reference
If you use this code, please support our research by citing the related paper(s) that applies:
> [1] M. Virgolin, A. De Lorenzo, E. Medvet, F. Randone. "Learning a Formula of Interpretability to Learn Interpretable Formulas". [Parallel Problem Solving from Nature XVI (2020)](https://doi.org/10.1007/978-3-030-58115-2_6). [arXiv preprint](https://arxiv.org/abs/2004.11170). [Video](https://www.youtube.com/watch?v=V2lmbStyMGE&ab_channel=MarcoVirgolin).

> [2] M. Virgolin, A. De Lorenzo, F. Randone, E. Medvet, M. Wahde. "Model Learning with Personalized Interpretability Estimation (ML-PIE)". To appear in EC+DM Workshop, Genetic and Evolutionary Computation Conference (2021). [arXiv preprint](https://arxiv.org/abs/2104.06060).


For the other part of the code used in [1], see [this repository](https://github.com/MaLeLabTs/GPFormulasInterpretability). 
