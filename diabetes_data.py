import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_raw, y_raw = load_diabetes(return_X_y=True)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X_raw)
y = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

n_samples, n_features = X.shape

X = np.column_stack([X, np.ones(len(X))])
D = X.shape[1]        
H = 10                
P = H * D + H + H + 1  

p0_bad = np.random.randn(P) * 3.0

# nonlinear neural network model

def unpack_params(p):
    W1 = p[:H*D].reshape(H, D)
    b1 = p[H*D:H*D+H]
    start = H*D+H
    W2 = p[start:start+H]
    b2 = p[start+H]
    return W1, b1, W2, b2

def nn_model(p, X):
    W1, b1, W2, b2 = unpack_params(p)
    hidden = np.tanh(X @ W1.T + b1)
    return hidden @ W2 + b2

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


p_gn, hist_gn = gauss_newton(nn_model, X, y, p0_bad)
p_lm, hist_lm = levenberg_marquardt(nn_model, X, y, p0_bad)
p_gd, hist_gd = gradient_descent(nn_model, X, y, p0_bad)
p_sgd, hist_sgd = stochastic_gradient_descent(nn_model, X, y, p0_bad)

def r2(y_true, y_pred):
    ssr = np.sum((y_pred - y_true)**2)
    sst = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ssr/sst

for name, p in [
    ("LM", p_lm),
    ("GN", p_gn),
    ("GD", p_gd),
    ("SGD", p_sgd),
]:
    y_pred = nn_model(p, X)
    m = mse(y, y_pred)
    r = r2(y, y_pred)
    print(f"{name:<6} | MSE={m:.6f} | R2={r:.4f}")


plt.figure(figsize=(10,6))
plt.plot(hist_lm, label="LM", linewidth=2)
plt.plot(hist_gn, label="GN", linewidth=2)
plt.plot(hist_gd, label="GD", linewidth=2)
plt.plot(hist_sgd, label="SGD", linewidth=2)
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Nonlinear Regression on Diabetes Dataset")
plt.legend()
plt.grid(True)
plt.show()

def compute_grad_norm(model, jac, p, X, y):
    r = model(p, X) - y
    J = jac(p, X)
    grad = J.T @ r / len(X)
    return np.linalg.norm(grad)


grad_norm_gn = [ ]
grad_norm_lm = [ ]
grad_norm_gd = [ ]
grad_norm_sgd = [ ]

def grad_norm_trace(history_params):
    return [compute_grad_norm(nn_model, nn_jacobian, p, X, y) for p in history_params]


param_trace_gn = []
param_trace_lm = []
param_trace_gd = []
param_trace_sgd = []


p_gn, hist_gn, trace_gn = gauss_newton(nn_model, X, y, p0_bad)
p_lm, hist_lm, trace_lm, lamb_trace_lm = levenberg_marquardt(nn_model, X, y, p0_bad)
p_gd, hist_gd, trace_gd = gradient_descent(nn_model, X, y, p0_bad)
p_sgd, hist_sgd, trace_sgd = stochastic_gradient_descent(nn_model, X, y, p0_bad)

grad_norm_gn  = [compute_grad_norm(nn_model, numerical_jacobian, p, X, y) for p in trace_gn]
grad_norm_lm  = [compute_grad_norm(nn_model, numerical_jacobian, p, X, y) for p in trace_lm]
grad_norm_gd  = [compute_grad_norm(nn_model, numerical_jacobian, p, X, y) for p in trace_gd]
grad_norm_sgd = [compute_grad_norm(nn_model, numerical_jacobian, p, X, y) for p in trace_sgd]

#gradient norm convergence graph 

plt.figure(figsize=(10,6))
plt.plot(grad_norm_lm,  label="LM", linewidth=2)
plt.plot(grad_norm_gn,  label="GN", linewidth=2)
plt.plot(grad_norm_gd,  label="GD", linewidth=2)
plt.plot(grad_norm_sgd, label="SGD", linewidth=2)
plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm Convergence")
plt.legend()
plt.grid(True)
plt.show()


y_pred_lm  = nn_model(p_lm, X)
y_pred_gn  = nn_model(p_gn, X)
y_pred_gd  = nn_model(p_gd, X)
y_pred_sgd = nn_model(p_sgd, X)


#actual vs predicted graph 

plt.figure(figsize=(10,6))
plt.plot(y, label="Actual", linewidth=2)
plt.plot(y_pred_lm,  label="LM", linewidth=2)
plt.plot(y_pred_gn,  label="GN", linewidth=2)
plt.plot(y_pred_gd,  label="GD", linewidth=2)
plt.plot(y_pred_sgd, label="SGD", linewidth=2)
plt.title("Predicted vs Actual (Full Range)")
plt.legend()
plt.grid(True)
plt.show()

# Zoomed region
a, b = 150, 200
plt.figure(figsize=(10,6))
plt.plot(y[a:b], label="Actual", linewidth=2)
plt.plot(y_pred_lm[a:b],  label="LM")
plt.plot(y_pred_gn[a:b],  label="GN")
plt.plot(y_pred_gd[a:b],  label="GD")
plt.plot(y_pred_sgd[a:b], label="SGD")
plt.title("Predicted vs Actual (Zoomed)")
plt.legend()
plt.grid(True)
plt.show()

