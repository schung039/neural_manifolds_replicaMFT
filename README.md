# Replica Mean Field Theory Analysis of Object Manifolds

Analysis tool for measuring manifold classification capacity, manifold radius, and manifold dimension.  Implements the technique described in *Classification and Geometry of General Perceptual Manifolds, (2018) Physical Review X.* and refined in *Separability and Geometry of Object Manifolds in Deep Neural Networks, (2019) BioRxiv *

## Install

First install required dependencies with
```
pip install -r requirements.txt
```

Then install the package via
```
pip install -e .
```

### Constructing data for analysis
Sample **P** classes from the dataset, and **M** examples from each class.  Calculate the activations at the desired layer for each example, and package them into an array of shape **(N, M)** where **N** is the dimensionality of the layer.  In the current implementation, **M** should be less than **N**.  Assemble each of these into an iterable (ex: a list like `X = [(N, M1), (N, M2),..., (N,MP)]`)

### Mean-field theoretic manifold analysis
After specifying the margin `kappa` and the number of t-vectors to sample `n_t` (a good default is 200), compute the average mean-field theoretic manifold capacity, manifold radius, and and dimensionality.

First, import the required packages.  `manifold_analysis_correlation` contains the mean-field theoretic analysis including center correlations.
```python
import numpy as np
from mftma.manifold_analysis_correlation import manifold_analysis_corr
```
Set up the data to analyze. Data should be a sequence of numpy arrays of shape (num_features, num_samples). Here, we show an example with random data with N=5000 features, M=50 samples per manifold, and P=100 manifolds.
```python 
X = [np.random.randn(5000, 50) for i in range(100)] # Replace this with data to analyze
```
Analyze the data.  In this example, the analysis is done at zero margin (`kappa=0`) with `n_t=200` samples.
```python
kappa = 0
n_t = 200

capacity_all, radius_all, dimension_all, center_correlation, K = manifold_analysis_corr(X, kappa, n_t)
```
To compute the average manifold capacity, the average should be done inversely. For manifold radius and dimension, the usual average is appropriate.
```python
avg_capacity = 1/np.mean(1/capacity_all)
avg_radius = np.mean(radius_all)
avg_dimensionality = np.mean(dimension_all)
```
For this random data example, the result is (up to small differences due to `n_t=200` samples)
```python
avg_capacity = 0.04
avg_radius = 1.48
avg_dimension = 36.17
center_correlation = 0.00
K = 1
```

### Manifold datasets
Some of the manifold datasets used in *Untangling in Invariant Speech Recognition* are available for download.

The LibriSpeech word manifolds dataset is avalaible here:
https://www.dropbox.com/sh/rh0wrsw88e77azd/AABG_YjDitkiYzfx6K45StXMa?dl=0

The LibriSpeech speaker manifolds dataset is available here:
https://www.dropbox.com/sh/wej6hq24c70irwl/AAAoQ6f6Sa5xOCSYnRDt9w4Ga?dl=0
