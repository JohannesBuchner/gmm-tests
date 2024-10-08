import numpy as np
from numpy import array
from scipy.stats import multivariate_normal
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


@settings(max_examples=1000)
@given(mean_and_cov())
@example(
    mean_cov=(2, array([0., 0.]), array([[1., 0.],
            [0., 1.]]), array([[0., 0.],
            [0., 0.]])),
).via('discovered failure')
@example(
    mean_cov=(1, array([0.]), array([[1.]]), array([[1.],
            [0.]])),
).via('discovered failure')
@example(
    mean_cov=(6,
     array([0., 0., 0., 0., 0., 0.]),
     array([[ 1.00000107e+00, -5.71254024e-08,  9.99999696e-01,
              9.99999583e-01,  9.99999675e-01,  9.99999979e-01],
            [-5.71254024e-08,  1.54534373e-06,  1.67651586e-08,
             -2.54656267e-07, -4.52842673e-09,  2.99544937e-07],
            [ 9.99999696e-01,  1.67651586e-08,  1.00000095e+00,
              9.99999657e-01,  9.99999707e-01,  9.99999994e-01],
            [ 9.99999583e-01, -2.54656267e-07,  9.99999657e-01,
              1.00000119e+00,  9.99999635e-01,  9.99999940e-01],
            [ 9.99999675e-01, -4.52842673e-09,  9.99999707e-01,
              9.99999635e-01,  1.00000099e+00,  9.99999990e-01],
            [ 9.99999979e-01,  2.99544937e-07,  9.99999994e-01,
              9.99999940e-01,  9.99999990e-01,  1.00000010e+00]]),
     array([[0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.]])),
).via('discovered failure')
@example(
    mean_cov=(6,
     array([0., 0., 0., 0., 0., 0.]),
     array([[ 2.28000099, -4.12000012,  2.27999939, -1.59999953,  2.27999988,
              2.27999986],
            [-4.12000012, 21.48000008, -4.12000007,  6.39999967, -4.11999995,
             -4.11999995],
            [ 2.27999939, -4.12000007,  2.28000087, -1.59999973,  2.27999991,
              2.2799999 ],
            [-1.59999953,  6.39999967, -1.59999973,  2.0000013 , -1.60000021,
             -1.6000002 ],
            [ 2.27999988, -4.11999995,  2.27999991, -1.60000021,  2.28000032,
              2.27999984],
            [ 2.27999986, -4.11999995,  2.2799999 , -1.6000002 ,  2.27999984,
              2.28000035]]),
     array([[0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.]])),
).via('discovered failure')
@example(
    mean_cov=(7,
     array([0., 0., 0., 0., 0., 0., 0.]),
     array([[156.25637361, -31.26969455,  12.40439959, -31.24564618,
              -31.22989933, -31.24225165, -31.2688804 ],
             [-31.26969455,   6.33006684,  -2.20457805,   6.2198964 ,
                6.16861271,   6.20884123,   6.34227887],
             [ 12.40439959,  -2.20457805,   2.43400622,  -2.56530348,
               -2.80150635,  -2.61622149,  -2.21679021],
             [-31.24564618,   6.2198964 ,  -2.56530348,   7.16206397,
                5.80788728,   5.8481158 ,   6.20768424],
             [-31.22989933,   6.16861271,  -2.80150635,   5.80788728,
                7.34003101,   5.75696927,   6.15640055],
             [-31.24225165,   6.20884123,  -2.61622149,   5.8481158 ,
                5.75696927,   7.23169779,   6.19662906],
             [-31.2688804 ,   6.34227887,  -2.21679021,   6.20768424,
                6.15640055,   6.19662906,   6.36588918]]),
     array([[0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0.]])),
).via('discovered failure')
@example(
    mean_cov=(1, array([0.]), array([[1.]]), array([[0.], [0.]])),
).via('discovered failure')
@example(
    mean_cov=(8,
     array([0., 0., 0., 0., 0., 0., 0., 0.]),
     array([[ 1.53365164e-06, -4.56057424e-07, -2.47157040e-07,
              -1.88188005e-07,  1.47416388e-07,  1.47416391e-07,
               4.49153299e-07,  1.47416391e-07],
             [-4.56057424e-07,  1.76562629e+00,  9.99999625e-01,
               9.99999679e-01, -3.26562499e+00,  1.98437501e+00,
               2.53124938e+00,  1.98437501e+00],
             [-2.47157040e-07,  9.99999625e-01,  1.00000099e+00,
               9.99999580e-01,  9.99999915e-01,  9.99999915e-01,
               1.00000006e+00,  9.99999915e-01],
             [-1.88188005e-07,  9.99999679e-01,  9.99999580e-01,
               1.00000091e+00,  9.99999927e-01,  9.99999927e-01,
               1.00000005e+00,  9.99999927e-01],
             [ 1.47416388e-07, -3.26562499e+00,  9.99999915e-01,
               9.99999927e-01,  2.47656250e+01, -4.48437497e+00,
              -7.53124995e+00, -4.48437497e+00],
             [ 1.47416391e-07,  1.98437501e+00,  9.99999915e-01,
               9.99999927e-01, -4.48437497e+00,  2.26562529e+00,
               2.96875004e+00,  2.26562479e+00],
             [ 4.49153299e-07,  2.53124938e+00,  1.00000006e+00,
               1.00000005e+00, -7.53124995e+00,  2.96875004e+00,
               4.06250038e+00,  2.96875004e+00],
             [ 1.47416391e-07,  1.98437501e+00,  9.99999915e-01,
               9.99999927e-01, -4.48437497e+00,  2.26562479e+00,
               2.96875004e+00,  2.26562529e+00]]),
     array([[0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0.]])),
).via('discovered failure')
@example(
    mean_cov=(7,
        array([0., 0., 0., 0., 0., 0., 0.]),
        array([[ 7.52684752, -4.9545883 , -0.31361835,  0.52269766,  0.52269929,
             6.92515709, -9.70648636],
           [-4.9545883 ,  3.26139136,  0.20644122, -0.3440688 , -0.34406688,
            -4.55852195,  6.38935566],
           [-0.31361835,  0.20644122,  0.01308347, -0.02177316, -0.0217791 ,
            -0.28854817,  0.40443697],
           [ 0.52269766, -0.3440688 , -0.02177316,  0.03630094,  0.0362984 ,
             0.48091352, -0.67406172],
           [ 0.52269929, -0.34406688, -0.0217791 ,  0.0362984 ,  0.03630217,
             0.48091543, -0.67405981],
           [ 6.92515709, -4.55852195, -0.28854817,  0.48091352,  0.48091543,
             6.37156541, -8.9305567 ],
           [-9.70648636,  6.38935566,  0.40443697, -0.67406172, -0.67405981,
            -8.9305567 , 12.51732134]]),
       array([[0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0.]]))
).via('discovered failure')
@example(
    mean_cov=(3,
        array([0., 0., 0.]),
        array([[32.00000007, 35.99999993,  8.00000001],
               [35.99999993, 40.50000006,  8.99999998],
               [ 8.00000001,  8.99999998,  2.00000008]]),
        array([[0., 0., 0.], [0., 0., 0.]]))
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
    assert x.shape == (2, ndim), (x.shape, 2, ndim)

    truth_logp = rv_truth.logpdf(x)
    assert truth_logp.shape == (len(x),)
    truth_p = rv_truth.pdf(x)

    precisions = [np.linalg.inv(cov) for cov in covs]

    # initialize GMM:
    skgmm = sklearn.mixture.GaussianMixture(
        n_components=ncomponents, weights_init=weights,
        means_init=means, precisions_init=precisions)
    skgmm._initialize(np.zeros((1, 1)), None)
    skgmm._set_parameters((weights, np.array(means), covs, skgmm.precisions_cholesky_))
    assert_allclose(skgmm.weights_, weights)
    assert_allclose(skgmm.means_, means)
    assert_allclose(skgmm.covariances_, covs)
    print('input:', x, skgmm.weights_, skgmm.means_, 'covs:', skgmm.covariances_, skgmm.precisions_cholesky_)
    print('expectation:', truth_p, 'log:', truth_logp)

    # compute probability:
    sk_logp = _estimate_log_gaussian_prob(x, skgmm.means_, skgmm.precisions_cholesky_, 'full').flatten()
    print('actual (_estimate_log_gaussian_prob):', sk_logp)
    assert sk_logp.shape == (len(x),), (sk_logp.shape, len(x))
    assert_allclose(truth_logp, sk_logp, atol=1e-4, rtol=1e-4)
    sk_logp1, sk_logp2 = skgmm._estimate_log_prob_resp(x)
    print('actual (_estimate_log_prob_resp):', sk_logp1, sk_logp2)
    assert_allclose(truth_logp, sk_logp1, atol=1e-4, rtol=1e-4)
    #assert_allclose(truth_logp, sk_logp2)

    sk_logp = skgmm.score(x)
    sk_p = skgmm.predict_proba(x)
    print('actual:', sk_p, 'log', sk_logp)

    # TODO: enable these two tests as well
    # assert_allclose(truth_logp, sk_logp, atol=1e-2, rtol=1e-2)  # -> sklearn-GMM gives score=0
    # assert_allclose(truth_p, sk_p, atol=1e-300, rtol=1e-4)  # -> sklearn-GMM gives p=1

