from datetime import datetime
from meteostat import Stations, Daily
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
import torch
from torch.utils.data import DataLoader, TensorDataset

start = datetime(2013, 3, 1)
end = datetime(2023, 3, 1)

data = Daily('72509', start, end)
data = data.fetch()
data=data.reset_index().iloc[:,[0,1,2,3,4,6,7,9]]

data.to_csv('boston_weather_data.csv',index=False)

csv_path = 'boston_weather_data.csv'   
df = pd.read_csv(csv_path)

df['time_parsed'] = pd.to_datetime(df['time'])
df = df.sort_values('time_parsed').reset_index(drop=True)

t0 = df['time_parsed'].iloc[0]
df['t_days'] = (df['time_parsed'] - t0).dt.total_seconds() / (24*3600.0)
t = df['t_days'].values
y = df['tavg'].values

t_mean = t.mean()
t_std = t.std() if t.std() > 0 else 1.0
t_scaled = (t - t_mean) / t_std  # unitless time
# We'll fit model using t_scaled but for interpreting B parameter we can convert later.

#  y = A * sin(omega * t + phi) + B * t + C
#  parameters: p = [A, omega, phi, B, C]

def model_np(p, t_in):
    A, omega, phi, B, C = p
    return A * np.sin(omega * t_in + phi) + B * t_in + C

def residuals_np(p, t_in, y_true):
    return model_np(p, t_in) - y_true

def mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def print_stats(name, p, t_in, y_true):
    y_pred = model_np(p, t_in)
    print(f"{name:10s} | MSE={mse(y_true,y_pred):10.4f} | R2={r2(y_true,y_pred):8.4f} | params={np.round(p,6)}")

# approximate the initilization
approx_amp = 0.5 * (np.nanmax(y) - np.nanmin(y))
omega0 = 2*np.pi * (t_std / 365.0)  
p0 = np.array([approx_amp, omega0, 0.0, 0.0, np.nanmean(y)], dtype=float)
print("initial p0:", p0)

t_fit = t_scaled
y_fit = y

# GN and LM with tracking

def gauss_newton_manual(p0, t_in, y_in, max_iter=100, tol=1e-8):
    p = p0.copy()
    history = []
    for k in range(max_iter):
        r = model_np(p, t_in) - y_in       # residuals (n,)
        J = numerical_jacobian(p, t_in)           # n x 5
        JTJ = J.T @ J
        JTr = J.T @ r
        # Solve JTJ * dp = -JTr  (standard GN)
        # add tiny damping for numerical stability
        dp = np.linalg.solve(JTJ + 1e-10*np.eye(JTJ.shape[0]), -JTr)
        p_new = p + dp
        loss = 0.5 * np.mean(r**2)
        history.append(loss)
        if np.linalg.norm(dp) < tol:
            p = p_new
            break
        p = p_new
    return p, history

def levenberg_marquardt_manual(p0, t_in, y_in, max_iter=200, tau=1e-3, nu=2.0, tol=1e-8):
    p = p0.copy()
    J = jacobian_np(p, t_in)
    A_mat = J.T @ J
    mu = tau * np.max(np.diag(A_mat))
    history = []
    for k in range(max_iter):
        r = model_np(p, t_in) - y_in
        J = jacobian_np(p, t_in)
        A_mat = J.T @ J
        g = J.T @ r
        loss = 0.5 * np.mean(r**2)
        history.append(loss)
        if np.linalg.norm(g, ord=np.inf) < tol:
            break
        try:
            dp = np.linalg.solve(A_mat + mu * np.eye(A_mat.shape[0]), -g)
        except np.linalg.LinAlgError:
            # fallback small identity
            dp = np.linalg.lstsq(A_mat + mu * np.eye(A_mat.shape[0]), -g, rcond=None)[0]
        p_try = p + dp
        r_try = model_np(p_try, t_in) - y_in
        loss_try = 0.5 * np.mean(r_try**2)
        rho = (loss - loss_try) / (0.5 * dp @ (mu * dp - g))
        if rho > 0:
            # accept
            p = p_try
            mu = mu * max(1/3, 1 - (2*rho - 1)**3)
            nu = 2.0
        else:
            mu = mu * nu
            nu = 2 * nu
        if np.linalg.norm(dp) < tol:
            break
    return p, history

# GD and SGD with tracking 
def gradient_descent_torch(p0, t_in, y_in, lr=1e-2, epochs=200):
    # t_in, y_in are numpy arrays
    x = torch.tensor(t_in, dtype=torch.float32).unsqueeze(1)  # shape (n,1)
    y_t = torch.tensor(y_in, dtype=torch.float32).unsqueeze(1)
    params = torch.tensor(p0, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.SGD([params], lr=lr)
    history = []
    for ep in range(epochs):
        opt.zero_grad()
        A = params[0]; omega = params[1]; phi = params[2]; B = params[3]; C = params[4]
        arg = omega * x + phi
        pred = A * torch.sin(arg) + B * x + C
        loss = torch.mean((pred - y_t)**2) * 0.5
        history.append(loss.item())
        loss.backward()
        opt.step()
    return params.detach().numpy(), history

def sgd_torch(p0, t_in, y_in, lr=1e-2, epochs=500, batch_size=32):
    x_all = torch.tensor(t_in, dtype=torch.float32).unsqueeze(1)
    y_all = torch.tensor(y_in, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(x_all, y_all)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    params = torch.tensor(p0, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.SGD([params], lr=lr)
    history = []
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            A = params[0]; omega = params[1]; phi = params[2]; B = params[3]; C = params[4]
            arg = omega * xb + phi
            pred = A * torch.sin(arg) + B * xb + C
            loss = torch.mean((pred - yb)**2) * 0.5
            loss.backward()
            opt.step()
        with torch.no_grad():
            A = params[0]; omega = params[1]; phi = params[2]; B = params[3]; C = params[4]
            arg_all = omega * x_all + phi
            pred_all = A * torch.sin(arg_all) + B * x_all + C
            loss_epoch = 0.5 * torch.mean((pred_all - y_all)**2).item()
            history.append(loss_epoch)
    return params.detach().numpy(), history

p0_fit = p0.copy()

p_gn, hist_gn = gauss_newton_manual(p0_fit, t_fit, y_fit, max_iter=50)
print_stats("Gauss-Newton", p_gn, t_fit, y_fit)

p_lm, hist_lm = levenberg_marquardt_manual(p0_fit, t_fit, y_fit, max_iter=500)
print_stats("Levenberg-Marquardt", p_lm, t_fit, y_fit)

p_gd, hist_gd = gradient_descent_torch(p0_fit, t_fit, y_fit, lr=1e-2, epochs=50)
print_stats("Gradient-Descent", p_gd, t_fit, y_fit)

p_sgd, hist_sgd = sgd_torch(p0_fit, t_fit, y_fit, lr=1e-2, epochs=50, batch_size=32)
print_stats("SGD", p_sgd, t_fit, y_fit)


# Convergence of loss

plt.figure(figsize=(10,5))
plt.plot(hist_gn, label='Gauss-Newton', linewidth=2)
plt.plot(hist_lm, label='Levenberg-Marquardt', linewidth=2)
plt.plot(hist_gd, label='Gradient-Descent', linewidth=1)
plt.plot(hist_sgd, label='SGD', linewidth=1)
plt.yscale('log')
plt.xlabel('Iteration / Epoch')
plt.ylabel('0.5 * MSE Loss (log scale)')
plt.title('Convergence Histories (loss)')
plt.legend()
plt.grid(True)
plt.show()

# Actual vs Predicted 

t_grid_days = np.linspace(t.min(), t.max(), 1000)
t_grid_scaled = (t_grid_days - t_mean) / t_std

y_gn = model_np(p_gn, t_grid_scaled)
y_lm = model_np(p_lm, t_grid_scaled)
y_gd = model_np(p_gd, t_grid_scaled)
y_sgd = model_np(p_sgd, t_grid_scaled)

plt.figure(figsize=(14,6))
plt.scatter(t, y, s=6, color='gray', alpha=0.5, label='Actual tavg')
plt.plot(t_grid_days, y_gn, label='Gauss-Newton', linewidth=2, linestyle='dotted', markevery=40, markersize=8)
plt.plot(t_grid_days, y_lm, linestyle='dotted',label='Levenberg-Marquardt', linewidth=2)

plt.plot(t_grid_days, y_gd, label='Gradient-Descent', linewidth=1)
plt.plot(t_grid_days, y_sgd, label='SGD', linewidth=1)
plt.xlabel('Days since ' + t0.strftime('%Y-%m-%d'))
plt.ylabel('Average temperature (°C)')
plt.title('Actual vs Predicted (different solvers)')
plt.legend()
plt.show()


