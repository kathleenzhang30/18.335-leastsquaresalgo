np.random.seed(42)
torch.manual_seed(42)

def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else float('nan')

os.makedirs("results", exist_ok=True)

# all the different types of models 

def linear_model_np(params, x):
    a, b = params
    return a * x + b

def linear_model_torch(params, x):
    # params: torch tensor shape (2,)
    a = params[0]
    b = params[1]
    return a * x + b

def nonlinear1_model_np(params, x):
    # y = A * x * exp(B * x)
    A, B = params
    return A * x * np.exp(B * x)

def nonlinear1_model_torch(params, x):
    A = params[0]
    B = params[1]
    return A * x * torch.exp(B * x)

def nonlinear2_model_np(params, x):
    a, b = params
    return a * np.exp(b * x)

def nonlinear2_model_torch(params, x):
    a = params[0]
    b = params[1]
    return a * torch.exp(b * x)

def highdim_model_np(params, X):
    return X @ params

def highdim_model_torch(params, X):
    return X @ params

def damped_sine_model_np(params, x):
    A, w, alpha = params
    return A * np.sin(w * x) * np.exp(-alpha * x)

def damped_sine_model_torch(params, x):
    A, w, alpha = params
    return A * torch.sin(w * x) * torch.exp(-alpha * x)


#SciPy wrapper methods 

def lm_fit(model_np, xdata, ydata, p0):
    params, res = levenberg_marquardt(model_np, xdata, ydata, p0)
    return params, res

def gn_fit(model_np, xdata, ydata, p0):
    def residuals(p):
        return (model_np(p, xdata) - ydata).ravel()
    res = least_squares(residuals, p0, method='trf', jac='2-point')
    return res.x, res

#PyTorch optimization wrappers 

def gradient_descent_torch(model_torch, xdata, ydata, p0, lr=1e-2, epochs=1000, verbose=False):
    x = torch.tensor(xdata, dtype=torch.float32)
    y = torch.tensor(ydata, dtype=torch.float32)

    params = torch.nn.Parameter(torch.tensor(p0, dtype=torch.float32))
    optimizer = torch.optim.SGD([params], lr=lr)

    last_loss = None
    for e in range(epochs):
        optimizer.zero_grad()
        y_pred = model_torch(params, x)
        loss = torch.mean((y_pred - y) ** 2)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu().numpy())
        if verbose and (e % max(1, epochs // 10) == 0):
            print(f"[GD] epoch {e:4d}/{epochs} loss={last_loss:.6f}")

    return params.detach().cpu().numpy(), last_loss

def sgd_torch(model_torch, xdata, ydata, p0, lr=1e-2, epochs=200, batch_size=32, verbose=False):
    x = torch.tensor(xdata, dtype=torch.float32)
    y = torch.tensor(ydata, dtype=torch.float32)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    params = torch.nn.Parameter(torch.tensor(p0, dtype=torch.float32))
    optimizer = torch.optim.SGD([params], lr=lr)

    last_loss = None
    for e in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            y_pred = model_torch(params, xb)
            loss = torch.mean((y_pred - yb) ** 2)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().numpy())
            n_batches += 1
        last_loss = epoch_loss / max(1, n_batches)
        if verbose and (e % max(1, epochs // 10) == 0):
            print(f"[SGD] epoch {e:4d}/{epochs} avg_loss={last_loss:.6f}")

    return params.detach().cpu().numpy(), last_loss

# Data Generation

def generate_datasets():
    # 1) Noisy linear
    x_linear = np.linspace(0, 5, 200)
    true_a, true_b = 2.5, -1.0
    y_linear = true_a * x_linear + true_b + np.random.normal(0, 0.5, x_linear.shape)

    # 2) Nonlinear function 1: y = 5 x exp(-3 x) + noise
    x_nl1 = np.linspace(0, 2, 200)
    y_nl1 = 5.0 * x_nl1 * np.exp(-3.0 * x_nl1) + np.random.normal(0, 0.2, x_nl1.shape)

    # 3) Nonlinear function 2: y = a exp(b x) + noise (true a=2, b=1)
    x_nl2 = np.linspace(0, 2, 200)
    y_nl2 = 2.0 * np.exp(1.0 * x_nl2) + np.random.normal(0, 0.2, x_nl2.shape)

    # 4) High-dimensional regression
    # Hard high-dimensional regression
    n_samples, n_features = 300, 20

    # Correlated features
    C = 0.85
    base = np.random.randn(n_samples, 1)
    X_hd = base @ np.ones((1, n_features)) + np.random.randn(n_samples, n_features) * (1 - C)

    true_hd = np.random.randn(n_features)

    # Strong + heteroscedastic noise
    noise = np.random.randn(n_samples) * (2 + 0.3*np.linalg.norm(X_hd, axis=1))

    y_hd = X_hd @ true_hd + noise

    # 6) Oscillatory (Damped Sine)
    x_osc = np.linspace(0, 6*np.pi, 400)
    true_A, true_B, true_w = 1.5, -0.7, 0.6
    y_osc = (
        true_A * np.sin(true_w * x_osc)
        + true_B * np.cos(2 * true_w * x_osc)
        + np.random.normal(0, 0.25, size=x_osc.shape)
    )


    datasets = {
        "Linear": {"x": x_linear, "y": y_linear, "p0": np.array([1.0, 1.0]), "model_np": linear_model_np, "model_torch": linear_model_torch},
        "Nonlinear1": {"x": x_nl1, "y": y_nl1, "p0": np.array([2.0, -1.0]), "model_np": nonlinear1_model_np, "model_torch": nonlinear1_model_torch},
        "Nonlinear2": {"x": x_nl2, "y": y_nl2, "p0": np.array([1.0, 0.5]), "model_np": nonlinear2_model_np, "model_torch": nonlinear2_model_torch},
        "HighDim": {"x": X_hd, "y": y_hd, "p0": np.zeros(X_hd.shape[1]), "model_np": highdim_model_np, "model_torch": highdim_model_torch},
        "Oscillatory": {"x": x_osc, "y": y_osc, "p0": np.array([-1, 0.3, 1.5]), "model_np": damped_sine_model_np, "model_torch": damped_sine_model_torch},
    }

    return datasets


def fit_and_evaluate_all(datasets):
    summary = {}

    for name, d in datasets.items():
        x = d["x"]
        y = d["y"]
        p0 = d["p0"].astype(float)
        model_np = d["model_np"]
        model_torch = d["model_torch"]

        print(f"\n=== Dataset: {name} ===")
        res_entry = {}

        try:
            p_lm, res_lm = lm_fit(model_np, x, y, p0)
            y_pred_lm = model_np(p_lm, x)
            res_entry["LM"] = {"params": p_lm, "mse": mse(y, y_pred_lm), "r2": r2_score(y, y_pred_lm)}
            print(f"LM params: {p_lm}  MSE={res_entry['LM']['mse']:.6f}  R2={res_entry['LM']['r2']:.6f}")
        except Exception as e:
            print("LM failed:", e)
            res_entry["LM"] = {"error": str(e)}

        try:
            p_gn, res_gn = gn_fit(model_np, x, y, p0)
            y_pred_gn = model_np(p_gn, x)
            res_entry["GN"] = {"params": p_gn, "mse": mse(y, y_pred_gn), "r2": r2_score(y, y_pred_gn)}
            print(f"Gauss-Newton params: {p_gn}  MSE={res_entry['GN']['mse']:.6f}  R2={res_entry['GN']['r2']:.6f}")
        except Exception as e:
            print("GN failed:", e)
            res_entry["GN"] = {"error": str(e)}

        try:
            if name == "HighDim":
                lr_gd, epochs_gd = 1e-2, 2000
            elif name.startswith("Nonlinear"):
                lr_gd, epochs_gd = 1e-2, 1500
            else:
                lr_gd, epochs_gd = 5e-3, 2000

            p_gd, loss_gd = gradient_descent_torch(model_torch, x, y, p0, lr=lr_gd, epochs=epochs_gd, verbose=False)
            y_pred_gd = model_np(p_gd, x)
            res_entry["GD"] = {"params": p_gd, "mse": mse(y, y_pred_gd), "r2": r2_score(y, y_pred_gd), "last_loss": loss_gd}
            print(f"GD params: {p_gd}  MSE={res_entry['GD']['mse']:.6f}  R2={res_entry['GD']['r2']:.6f}")
        except Exception as e:
            print("GD failed:", e)
            res_entry["GD"] = {"error": str(e)}

        try:
            if name == "HighDim":
                lr_sgd, epochs_sgd, batch = 1e-2, 400, 32
            elif name.startswith("Nonlinear"):
                lr_sgd, epochs_sgd, batch = 5e-3, 300, 32
            else:
                lr_sgd, epochs_sgd, batch = 5e-3, 400, 32

            p_sgd, loss_sgd = sgd_torch(model_torch, x, y, p0, lr=lr_sgd, epochs=epochs_sgd, batch_size=batch, verbose=False)
            y_pred_sgd = model_np(p_sgd, x)
            res_entry["SGD"] = {"params": p_sgd, "mse": mse(y, y_pred_sgd), "r2": r2_score(y, y_pred_sgd), "last_loss": loss_sgd}
            print(f"SGD params: {p_sgd}  MSE={res_entry['SGD']['mse']:.6f}  R2={res_entry['SGD']['r2']:.6f}")
        except Exception as e:
            print("SGD failed:", e)
            res_entry["SGD"] = {"error": str(e)}

        summary[name] = res_entry

        # Plotting
        if name in ("Linear", "Nonlinear1", "Nonlinear2", "Oscillatory"):
            xs = x
            plt.figure(figsize=(8,5))
            plt.scatter(xs, y, s=12, alpha=0.6, label="data")
            # Plot dense curves for each method (where available)
            x_dense = np.linspace(np.min(xs), np.max(xs), 400)
            linestyles = {"LM": "-", "GN": "--", "GD": ":", "SGD": "-."}

            for method in ["LM", "GN", "GD", "SGD"]:
                info = res_entry.get(method)
                if info and "params" in info:
                    y_dense = model_np(info["params"], x_dense)
                    plt.plot(
                        x_dense,
                        y_dense,
                        label=f"{method} (MSE {info['mse']:.3f})",
                        linewidth=2,
                        linestyle=linestyles.get(method, "-")
                    )

            plt.title(f"{name} fit comparison")
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.tight_layout()
            plt.savefig(f"results/{name}_fit.png", dpi=200)
            plt.close()

        else:
            # High-dimensional: scatter true vs predicted for each method
            plt.figure(figsize=(12,4))
            methods = ["LM", "GN", "GD", "SGD"]
            n = len(methods)
            for i, method in enumerate(methods, start=1):
                plt.subplot(1, n, i)
                info = res_entry.get(method, {})
                if info and "params" in info:
                    y_pred = model_np(info["params"], x)
                    plt.scatter(y, y_pred, s=8, alpha=0.6)
                    mn, mx = np.min(np.concatenate([y, y_pred])), np.max(np.concatenate([y, y_pred]))
                    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
                    plt.xlabel("true y")
                    plt.ylabel("pred y")
                    plt.title(f"{method}\nMSE={info['mse']:.3f}, R2={info['r2']:.3f}")
                else:
                    plt.text(0.1, 0.5, f"{method} failed", transform=plt.gca().transAxes)
                    plt.axis("off")
            plt.suptitle("High-dimensional predicted vs true")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig("results/HighDim_pred_vs_true.png", dpi=200)
            plt.close()

    return summary


if __name__ == "__main__":
    datasets = generate_datasets()
    summary = fit_and_evaluate_all(datasets)

    print("\n\n=== Summary Table ===")
    for ds_name, info in summary.items():
        print(f"\nDataset: {ds_name}")
        for method in ["LM","GN","GD","SGD"]:
            m = info.get(method, {})
            if "params" in m:
                print(f"  {method:4s} | MSE={m['mse']:.6f} | R2={m['r2']:.6f} | params={np.array2string(m['params'], precision=4, max_line_width=120)}")
            else:
                print(f"  {method:4s} | error: {m.get('error','unknown')}")

    print("\nPlots saved to ./results/ (Linear_fit.png, Nonlinear1_fit.png, Nonlinear2_fit.png, Oscillatory_fit.png, HighDim_pred_vs_true.png)")
