import numpy as np

# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html


def batchnorm_forward(x, gamma, beta, eps):
    D, N = x.shape

    # step1: calculate mean
    mu = 1./N * np.sum(x, axis=1, keepdims=True)

    # step2: subtract mean vector of every trainings example
    xmu = x - mu

    # step3: following the lower branch - calculation denominator
    sq = xmu**2

    # step4: calculate variance
    var = 1./N * np.sum(sq, axis=1, keepdims=True)

    # step5: add eps for numerical stability, then sqrt
    sqrtvar = np.sqrt(var + eps)

    # step6: invert sqrtwar
    ivar = 1./sqrtvar

    # step7: execute normalization
    xhat = xmu * ivar

    # step8: Nor the two transformation steps
    gammax = gamma * xhat

    # step9
    out = gammax + beta

    # store intermediate
    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

    return out, cache


def batchnorm_backward(dout, cache):
    # unfold the variables stored in cache
    xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache

    # get the dimensions of the input/output
    D, N = dout.shape

    # step9
    dbeta = np.sum(dout, axis=1, keepdims=True)
    dgammax = dout  # not necessary, but more understandable

    # step8
    dgamma = np.sum(dgammax*xhat, axis=1, keepdims=True)
    dxhat = dgammax * gamma

    # step7
    divar = np.sum(dxhat*xmu, axis=1, keepdims=True)
    dxmu1 = dxhat * ivar

    # step6
    dsqrtvar = -1. /(sqrtvar**2) * divar

    # step5
    dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

    # step4
    dsq = 1. /N * np.ones((D, N)) * dvar

    # step3
    dxmu2 = 2 * xmu * dsq

    # step2
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dx1, axis=1, keepdims=True)

    # step1
    dx2 = 1. /N * np.ones((D, N)) * dmu

    #step0
    dx = dx1 + dx2

    return dx, dgamma, dbeta
