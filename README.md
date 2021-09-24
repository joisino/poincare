# Poincare: Recommending Publication Venues via Treatment Effect Estimation

This project aims to recommend publication venues to scientific papers.

Paper: https://arxiv.org/abs/2010.09157

## How to Use

### Requirements

Python 3.7

* numpy >= 1.16.4
* scipy >= 1.5.4
* scikit-learn >= 0.22.1

Install them by `pip install -r requirements.txt`.

### Dataset

`dblp-dataset.pickle` is the preprocessed dataset. This pickled object is a list, where each element represents a paper and is a tuple with the format `(title, feature, venue, citation)`.

* `title` is a string
* `feature` is a 3812-dimensional numpy array that represents a bag of fields of study. The names of fields are listed in `fields.txt`. The i-th line in `fields.txt` corresponds to the i-th dimension.
* `venue` is an integer in `{0, 1, 2, 3, 4}`. `0`: AAAI, `1`: IJCAI, `2`: KDD, `3`: NeurIPS, `4`: ICML.
* `citation` is an integer.

### Run

```
$ python main.py
```

Note that it may take a few hours to a few days. If it takes too long time, reduce the number of iterations and/or the set of hyperparameters.

### Results

|Method|AAAI|IJCAI|KDD|NeurIPS|ICML|Total|
|----|----|----|----|----|----|----|
|Linear Regression|0.000 ± 0.058|-0.081 ± 0.046|0.109 ± 0.097|0.117 ± 0.060|-0.030 ± 0.066|-0.028 ± 0.026|
|Random Forrest|0.015 ± 0.064|-0.096 ± 0.045|0.069 ± 0.150|0.120 ± 0.084|0.048 ± 0.059|-0.094 ± 0.029|
|Support Vector Machine|0.004 ± 0.070|-0.081 ± 0.050|0.073 ± 0.105|0.160 ± 0.049|0.058 ± 0.074|0.025 ± 0.031|
|Multi Layer Perceptron|0.006 ± 0.046|-0.063 ± 0.049|0.087 ± 0.107|0.143 ± 0.040|0.017 ± 0.068|0.014 ± 0.024|
|Poincare|0.179 ± 0.050|0.252 ± 0.078|0.159 ± 0.085|0.406 ± 0.041|0.457 ± 0.071|0.388 ± 0.026|
|Poincare-UW|0.183 ± 0.049|0.245 ± 0.071|0.164 ± 0.073|0.405 ± 0.040|0.460 ± 0.065|0.389 ± 0.024|
|Poincare-S|0.033 ± 0.068|0.076 ± 0.077|0.110 ± 0.108|0.343 ± 0.063|0.457 ± 0.071|0.192 ± 0.035|

## Citation

```
@article{sato2020poincare,
  author    = {Ryoma Sato and Makoto Yamada and Hisashi Kashima},
  title     = {Poincare: Recommending Publication Venues via Treatment Effect Estimation},
  journal   = {arXiv},
  year      = {2020},
}
```
