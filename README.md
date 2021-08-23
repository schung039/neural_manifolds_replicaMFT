# Replica Mean Field Theory Analysis of Object Manifolds

Analysis tool for measuring manifold classification capacity, manifold radius, and manifold dimension, implemented for the results in *[Untangling in Invariant Speech Recognition, (2019) NeurIPS](https://arxiv.org/abs/2003.01787)*.  The code implements the technique first described in *Classification and Geometry of General Perceptual Manifolds, (2018) Physical Review X.* and refined in *Separability and Geometry of Object Manifolds in Deep Neural Networks, (2019) BioRxiv; (2020) Nature Communications*. 

If you find this code useful for your research, please cite [our paper](https://arxiv.org/abs/2003.01787):  
```
@inproceedings{stephenson2019untangling,
  title={Untangling in Invariant Speech Recognition},
  author={Stephenson, Cory and Feather, Jenelle and Padhy, Suchismita and Elibol, Oguz and Tang, Hanlin and McDermott, Josh and Chung, SueYeon},
  booktitle={Advances in Neural Information Processing Systems},
  pages={14368--14378},
  year={2019}
}
```

## Install

First install required dependencies with
```
pip install -r requirements.txt
```

Then install the package via
```
pip install -e .
```
## Usage
The following contains usage instructions for constructing data and feeding it to the analysis tool. An example analysis of a deep neural network implemented in PyTorch along with some higher level tools can be found in this [example notebook](examples/MFTMA_VGG16_example.ipynb).

### Constructing data for analysis
Sample **P** classes from the dataset, and **M** examples from each class.  Calculate the activations at the desired layer for each example, and package them into an array of shape **(N, M)** where **N** is the dimensionality of the layer.  In the current implementation, **M** should be less than **N**.  Assemble each of these into an iterable (ex: a list like `X = [(N, M1), (N, M2),..., (N,MP)]`)

## Mean-field theoretic manifold analysis
After specifying the margin `kappa` and the number of t-vectors to sample `n_t` (a good default is 200), compute the average mean-field theoretic manifold capacity, manifold radius, and and dimensionality.

First, import the required packages.  `manifold_analysis_correlation` contains the mean-field theoretic analysis including center correlations.
```python
import numpy as np
from mftma.manifold_analysis_correlation import manifold_analysis_corr
```
Set up the data to analyze. Data should be a sequence of numpy arrays of shape (num_features, num_samples). Here, we show an example with random data with N=5000 features, M=50 samples per manifold, and P=100 manifolds.
```python 
np.random.seed(0)
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
avg_dimension = np.mean(dimension_all)
```
For this random data example, the result is (up to small differences due to `n_t=200` samples)
```python
avg_capacity = 0.04
avg_radius = 1.47
avg_dimension = 36.17
center_correlation = 0.01
K = 1
```

## Total data dimension analysis
In addition to the manifold geometry metrics, the total dimensionality of the data can be computed. We use two different measures of total dimension. The first computes the number of dimensions required to explain a given percentage (default 90%) of the total variance. The second computes the participation ratio of the data. Using the random data from above, this looks like
```python
from mftma.alldata_dimension_analysis import alldata_dimension_analysis

percentage = 0.90
D_participation_ratio, D_explained_variance, D_feature = alldata_dimension_analysis(X, perc=percentage)
```
For the random data, this results in the large values of
```python
D_participation_ratio = 2499
D_explained_variance = 2548
D_feature = 5000 # By construction of the dataset and returned simply for convenience
```
## Manifold datasets
Some of the manifold datasets used in *Untangling in Invariant Speech Recognition* are available for download.

The LibriSpeech word manifolds dataset is avalaible here:
https://www.dropbox.com/sh/rh0wrsw88e77azd/AABG_YjDitkiYzfx6K45StXMa?dl=0

The LibriSpeech speaker manifolds dataset is available here:
https://www.dropbox.com/sh/wej6hq24c70irwl/AAAoQ6f6Sa5xOCSYnRDt9w4Ga?dl=0

## Support 
Email cory.stephenson@intel.com or sueyeonchung@gmail.com with any questions.
