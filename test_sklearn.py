import numpy as np
from numpy import array
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from numpy.testing import assert_allclose
from hypothesis import given, strategies as st, example, settings
from hypothesis.extra.numpy import arrays
import sklearn.mixture
from  sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob

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
    #if not np.std(A) > min_std:
    #    return False
    if (np.diag(A) <= min_std).any():
        return False

    try:
        np.linalg.inv(np.linalg.inv(A))
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


@given(mean_and_cov())
@example(
    mean_cov=(2, array([0., 0.]), array([[1., 0.],
            [0., 1.]]), array([[0., 0.],
            [0., 0.]])),
).via('discovered failure')
def test_single(mean_cov):
    # a askcarl with one component must behave the same as a single gaussian
    ndim, mu, cov, x = mean_cov
    ncomponents = 1
    if not valid_covariance_matrix(cov):
        return

    rv_truth = multivariate_normal(mu, cov)
    covs = [cov]
    means = [mu]
    weights = [1.0]

    truth_logp = rv_truth.logpdf(x)
    truth_p = rv_truth.pdf(x)

    precisions = [np.linalg.inv(cov) for cov in covs]
    # compare results of GMM to sklearn
    skgmm = sklearn.mixture.GaussianMixture(
        n_components=ncomponents, weights_init=weights,
        means_init=means, precisions_init=precisions)
    skgmm._initialize(np.zeros((1, 1)), None)
    skgmm._set_parameters((weights, np.array(means), covs, skgmm.precisions_cholesky_))
    assert_allclose(skgmm.weights_, weights)
    assert_allclose(skgmm.means_, means)
    assert_allclose(skgmm.covariances_, covs)
    # compare results of GMM to pypmc
    print(x, skgmm.means_, skgmm.precisions_cholesky_)
    sk_logp = _estimate_log_gaussian_prob(x, skgmm.means_, skgmm.precisions_cholesky_, 'full').flatten()
    print(sk_logp)
    assert sk_logp.shape == (len(x),), (sk_logp.shape, len(x))
    print(skgmm.weights_, truth_logp, truth_p)
    assert_allclose(truth_logp, sk_logp, atol=1e-2, rtol=1e-2)
    sk_logp = skgmm.score(x)
    sk_p = skgmm.predict_proba(x)
    assert_allclose(truth_logp, sk_logp, atol=1e-2, rtol=1e-2)
    _, sk_logp2 = skgmm._estimate_log_prob_resp(x)
    assert_allclose(sk_logp2, sk_logp)
    assert_allclose(truth_p, sk_p, atol=1e-300, rtol=1e-4)

