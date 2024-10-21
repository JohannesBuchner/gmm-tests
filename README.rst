Numerical tests of Gaussian Mixture models
-------------------------------------------

This repo contains automated fuzzing tests for two libraries,
scikit-learn and pypmc, which define Gaussian Mixture Models.

The finding is scikit-learn's log-probabilities are 
inaccurate, reported as a bug at
https://github.com/scikit-learn/scikit-learn/issues/29989

In contrast, pypmc seems to work fine with relative and absolute
deviations below 1e-4.

Usage
------

    $ pytest

You need hypothesis and pytest installed.


**Resolved!**
see bug report above.

