# import pytest
import numpy as np
import torch
from torch.autograd import Variable

from utils import *

import sys
sys.path.append('..')
from optim.lpsgd import LPSGD
from examples import regression

np.random.seed(0xdeadbeef)

def regression_LPSGD(x, y, w, lr, K=1, n_features=None, n_classes=1, bits=None, scale=0.0001):
    model = regression.utils.build_model(n_features, n_classes, initial_value=w)
    x = torch.from_numpy(x).float()
    # Linear regression
    if n_classes == 1:
        y = torch.from_numpy(y).float().view(-1,1)
        loss = torch.nn.MSELoss()
    else: # Multiclass logistic
        y = torch.from_numpy(y).long()
        loss = torch.nn.CrossEntropyLoss()

    synth_dataset = regression.utils.SynthDataset(x, y)
    train_loader = torch.utils.data.DataLoader(synth_dataset)
    lpsgd_opt = LPSGD(model.parameters(), scale_factor=0.000000001, lr=lr, bits=bits)

    for k in range(K):
        for i, (data, target) in enumerate(train_loader):
            lpsgd_opt.step()
            w = np.asarray([p.data.numpy() for p in
                list(model.parameters())]).reshape(n_classes, n_features)
    return w

def regression_SGD(x, y, w, lr, K, n_features=None, n_classes=1):
    model = regression.utils.build_model(n_features, n_classes, initial_value=w)
    x = torch.from_numpy(x).float()
    # Linear regression
    if n_classes == 1:
        y = torch.from_numpy(y).float().view(-1,1)
        loss = torch.nn.MSELoss()
    else: # Multiclass logistic
        y = torch.from_numpy(y).long()
        loss = torch.nn.CrossEntropyLoss()

    synth_dataset = regression.utils.SynthDataset(x, y)
    train_loader = torch.utils.data.DataLoader(synth_dataset)
    sgd_opt = torch.optim.SGD(model.parameters(), lr=lr)

    for k in range(K):
        for i, (data, target) in enumerate(train_loader):
            sgd_opt.step()
            w = np.asarray([p.data.numpy() for p in
                list(model.parameters())]).reshape(n_classes, n_features)
    return w


def test_logistic_regression_lpsgd_fp_mode():
    n_samples = 100
    n_features = 10
    n_class = 3
    lr = 0.1
    K = 10
    b = None
    x = np.random.rand(n_samples, n_features)
    y = np.random.uniform(0,1, size=(n_samples,))
    w = np.random.uniform(0, 0.1, (n_class, n_features))
    w_sgd = regression_SGD(x, y, w, lr, K=K, n_features=n_features, n_classes=n_class)
    w_lpsgd = regression_LPSGD(x, y, w, lr, K=K, n_features=n_features, n_classes=n_class, bits=b)
    np.testing.assert_allclose(w_sgd, w_lpsgd, rtol=1e-4)
    print("logistic regression lpsgd fp model test done")


def test_logistic_regression_lpsgd_lp_mode1():
    n_samples = 100
    n_features = 10
    n_class = 3
    lr = 0.1
    K = 10
    b = 32
    scale = 0.0000001
    x = np.random.rand(n_samples, n_features)
    y = np.random.uniform(0,1, size=(n_samples,))
    w = np.random.uniform(0, 0.1, (n_class, n_features))
    w_sgd = regression_SGD(x, y, w, lr, K=K, n_features=n_features, n_classes=n_class)
    w_lpsgd = regression_LPSGD(x, y, w, lr, K=K, n_features=n_features, n_classes=n_class, bits=b, scale=scale)
    np.testing.assert_allclose(w_sgd, w_lpsgd, rtol=1e-4)
    print("logistic regression lpsgd lp model test 1 done")


def test_logistic_regression_lpsgd_lp_mode2():
    n_samples = 100
    n_features = 10
    n_class = 3
    lr = 0.1
    K = 10
    b = 16
    scale = 0.01
    x = np.random.rand(n_samples, n_features)
    y = np.random.uniform(0,1, size=(n_samples,))
    w = np.random.uniform(0, 0.1, (n_class, n_features))
    w_sgd = regression_SGD(x, y, w, lr, K=K, n_features=n_features, n_classes=n_class)
    w_lpsgd = regression_LPSGD(x, y, w, lr, K=K, n_features=n_features, n_classes=n_class, bits=b, scale=scale)
    assert np.sum( (w_sgd - w_lpsgd)**2) / np.sum(w_sgd**2) > 1e-4
    # np.testing.assert_allclose(w_sgd, w_lpsgd, rtol=1e-4)
    print("logistic regression lpsgd lp model test 2 done")


if __name__ == "__main__":
    test_logistic_regression_lpsgd_lp_mode1()
    test_logistic_regression_lpsgd_lp_mode2()
    # test_logistic_regression_lpsgd_fp_mode()
