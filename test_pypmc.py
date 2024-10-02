import numpy as np
from numpy import array
from scipy.stats import multivariate_normal
from numpy.testing import assert_allclose
from hypothesis import given, strategies as st, example, settings
from hypothesis.extra.numpy import arrays
from pypmc.density.mixture import create_gaussian_mixture

def valid_QR(vectors):
    q, r = np.linalg.qr(vectors)
    return q.shape == vectors.shape and np.all(np.abs(np.diag(r)) > 1e-3) and np.all(np.abs(np.diag(r)) < 1000)

def make_covariance_matrix_via_QR(normalisations, vectors):
    q, r = np.linalg.qr(vectors)
    orthogonal_vectors = q @ np.diag(np.diag(r))
    cov = orthogonal_vectors @ np.diag(normalisations) @ orthogonal_vectors.T
    return cov

def valid_covariance_matrix(A, min_std=1e-6):
    if not np.isfinite(A).all():
        return False
    if (np.diag(A) <= min_std).any():
        return False

    try:
        np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return False

    try:
        multivariate_normal(mean=np.zeros(len(A)), cov=A)
    except ValueError:
        return False

    return True

# Strategy to generate arbitrary dimensionality mean and covariance
@st.composite
def mean_and_cov(draw):
    dim = draw(st.integers(min_value=1, max_value=10))  # Arbitrary dimensionality
    mu = draw(arrays(np.float64, (dim,), elements=st.floats(-10, 10)))  # Mean vector
    eigval = draw(arrays(np.float64, (dim,), elements=st.floats(1e-6, 10)))
    vectors = draw(arrays(np.float64, (dim,dim), elements=st.floats(-10, 10)).filter(valid_QR))
    cov = make_covariance_matrix_via_QR(eigval, vectors)
    x = draw(arrays(np.float64, (2, dim), elements=st.floats(-1e6, 1e6)))
    return dim, mu, cov, x


@settings(max_examples=1000)
@given(mean_and_cov())
@example(
    mean_cov=(2, array([0., 0.]), array([[1., 0.],
            [0., 1.]]), array([[0., 0.],
            [0., 0.]])),
).via('discovered failure')
def test_single(mean_cov):
    # a askcarl with one component must behave the same as a single gaussian
    ndim, mu, cov, x = mean_cov
    if not valid_covariance_matrix(cov):
        return

    truth_logp = multivariate_normal(mu, cov).logpdf(x)
    target_mixture = create_gaussian_mixture([mu], [cov], [1.0])
    pypmc_logp = np.array([target_mixture.evaluate(xi) for xi in x])

    assert_allclose(truth_logp, pypmc_logp, atol=1e-2, rtol=1e-2)

