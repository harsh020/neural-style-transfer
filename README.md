# neural_style_transfer
Package to perform Neural Style Transfer. A reimplemetation of Neural Style Transfer
by tensorflow.org


# How to install
* Clone the repository using `git clone https://github.com/harsh020/neural-style-transfer.git`.
* Create a virtualenv, if don't want to install globally.
* `pip install .`.


# How to use
```
from neural_style_transfer as nst

gen = nst.Generator('path/to/style/image', 'path/to/content/image')
tensor = gen.fit()
img = gen.transform(tensor)
```
