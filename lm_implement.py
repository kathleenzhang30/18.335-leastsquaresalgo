import numpy as np
from scipy.optimize import least_squares
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

def numerical_jacobian(f, p, eps=1e-6):
    p = np.asarray(p, dtype=float)
    r0 = np.asarray(f(p))
    m = r0.size
    n = p.size
    J = np.zeros((m, n))

    for j in range(n):
        dp = np.zeros_like(p)
        h = eps * max(1.0, abs(p[j]))
        dp[j] = h
        r1 = np.asarray(f(p + dp))
        J[:, j] = (r1 - r0) / h

    return J

def levenberg_marquardt(model, xdata, ydata, p0):
    p = p0.astype(float)

    def residual_vector(params):
        return ydata - model(params, xdata)

    r = residual_vector(p)
    cost = 0.5 * r @ r

    lamb = 1e-3  
    nu = 2        

    for _ in range(100):
        J = numerical_jacobian(residual_vector, p)
        g = J.T @ r
        H = J.T @ J

        delta = np.linalg.solve(H + lamb * np.eye(len(p)), -g)

        p_trial = p + delta
        r_trial = residual_vector(p_trial)
        cost_trial = 0.5 * r_trial @ r_trial

        act = cost - cost_trial
        pred = -0.5 * delta @ (g + lamb * delta)

        rho = act / pred if pred > 0 else 0

        if rho > 0:
            # Accept step
            p = p_trial
            r = r_trial
            cost = cost_trial

            lamb = lamb * max(1/3, 1 - (2*rho - 1)**3)
            nu = 2
        else:
            lamb = lamb * nu
            nu = 2 * nu

        if np.linalg.norm(delta) < 1e-8:
            break

    return p, cost


def gradient_descent(model, xdata, ydata, p0, lr=1e-3, epochs=500):
    x = torch.tensor(xdata, dtype=torch.float32)
    y = torch.tensor(ydata, dtype=torch.float32)
    params = torch.tensor(p0, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.SGD([params], lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        pred = torch.tensor(model(params, xdata), dtype=torch.float32)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        optimizer.step()

    return params.detach().numpy()

def stochastic_gradient_descent(model, xdata, ydata, p0, lr=1e-3, epochs=20, batch_size=16):
    x = torch.tensor(xdata, dtype=torch.float32)
    y = torch.tensor(ydata, dtype=torch.float32)
    params = torch.tensor(p0, dtype=torch.float32, requires_grad=True)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.SGD([params], lr=lr)

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = torch.tensor(model(params, xb.numpy()), dtype=torch.float32)
            loss = torch.mean((pred - yb) ** 2)
            loss.backward()
            optimizer.step()

    return params.detach().numpy()

def gauss_newton(model, xdata, ydata, p0):
    def residuals(params):
        return model(params, xdata) - ydata

    result = least_squares(
        residuals,
        p0,
        method='trf',        
        jac='2-point',     
    )
    return result.x, result




