"""
Monte Carlo limb-darkening simulation
Purpose: simulate photon random-walk in optical depth and record final scattering angle mu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_tau_step(rng):
    return -np.log(rng.random())

def emit_photon(tau_max, rng):
    tau = tau_max
    mu = rng.random()  # outward hemisphere [0,1]
    delta_tau = get_tau_step(rng)
    tau_new = tau - delta_tau * mu
    return tau_new, mu

def scatter_photon(tau, rng):
    mu = 2.0 * rng.random() - 1.0
    delta_tau = get_tau_step(rng)
    tau_new = tau - delta_tau * mu
    return tau_new, mu

def simulate_one_photon(tau_max, rng, max_scatter=10000):
    tau, mu = emit_photon(tau_max, rng)
    n_scatter = 0
    if tau < 0:
        return mu, n_scatter
    while True:
        n_scatter += 1
        tau, mu = scatter_photon(tau, rng)
        if n_scatter >= max_scatter:
            return None
        if tau < 0:
            return mu, n_scatter
        if tau >= tau_max:
            return None

def collect_escaped_mu(n_escaped, tau_max=10.0, seed=8675309, verbose=True):
    rng = np.random.default_rng(seed)
    mu_list = []
    attempts = 0
    report_every = max(1, n_escaped // 10)
    while len(mu_list) < n_escaped:
        attempts += 1
        res = simulate_one_photon(tau_max, rng)
        if res is None:
            continue
        mu_last, _ = res
        if mu_last > 0:
            mu_list.append(mu_last)
        if verbose and (len(mu_list) % report_every == 0):
            print(f"Collected {len(mu_list)}/{n_escaped} escaped photons (attempts {attempts}) for tau_max={tau_max}")
    return np.array(mu_list)

def compute_I_over_I1(mu_array, n_bins=20):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    counts, edges = np.histogram(mu_array, bins=bins)
    centers = 0.5*(edges[:-1] + edges[1:])
    with np.errstate(divide='ignore', invalid='ignore'):
        I_proxy = counts / centers
    if not np.isfinite(I_proxy[0]):
        I_proxy[0] = I_proxy[1]
    if I_proxy[-1] == 0:
        I_norm = I_proxy / (I_proxy.max() if I_proxy.max() > 0 else 1.0)
    else:
        I_norm = I_proxy / I_proxy[-1]
    sigma_counts = np.sqrt(counts)
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_I = sigma_counts / centers
    if not np.isfinite(sigma_I[0]):
        sigma_I[0] = sigma_I[1]
    if I_proxy[-1] == 0:
        sigma_I_norm = sigma_I / (I_proxy.max() if I_proxy.max() > 0 else 1.0)
    else:
        sigma_I_norm = sigma_I / I_proxy[-1]
    return centers, counts, I_norm, sigma_I_norm

def linear_limb(mu, a):
    return 1.0 - a*(1.0 - mu)

def fit_linear_limb(centers, I_norm, sigma=None):
    mask = centers > 0
    x = centers[mask]
    y = I_norm[mask]
    if sigma is None:
        popt, pcov = curve_fit(linear_limb, x, y, p0=[0.5])
    else:
        popt, pcov = curve_fit(linear_limb, x, y, p0=[0.5], sigma=sigma[mask], absolute_sigma=True)
    a = popt[0]
    a_err = np.sqrt(np.diag(pcov))[0]
    return a, a_err

def plot_histogram(centers, counts, tau_max):
    plt.figure(figsize=(6,4))
    width = (centers[1]-centers[0]) * 0.95
    plt.bar(centers, counts, width=width, color='C0', edgecolor='k', alpha=0.8)
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$N(\mu)$')
    plt.title(f'N(mu) histogram, tau_max={tau_max}')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

def plot_I_and_fit(centers, I_norm, sigma_I, a_fit, a_err, tau_max):
    plt.figure(figsize=(6,4))
    mu_plot = np.linspace(0.01, 1.0, 300)
    plt.errorbar(centers, I_norm, yerr=sigma_I, fmt='o', color='C1', capsize=3, label='MC binned $I(\\mu)/I_1$')
    plt.plot(mu_plot, linear_limb(mu_plot, a_fit), '-', color='C2', lw=2,
             label=f'Linear fit: $1 - a(1-\\mu)$, a={a_fit:.4f} ± {a_err:.4f}')
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$I(\mu)/I_1$')
    plt.title(f'I(mu)/I1 and linear fit, tau_max={tau_max}')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters
    N_PHOTONS = 100000
    TAU_THICK = 10.0
    TAU_THIN = 1e-4
    N_BINS = 20

    # Thick case
    mu_thick = collect_escaped_mu(N_PHOTONS, tau_max=TAU_THICK, seed=23455, verbose=True)
    centers_t, counts_t, I_norm_t, sigma_I_t = compute_I_over_I1(mu_thick, n_bins=N_BINS)
    a_t, aerr_t = fit_linear_limb(centers_t, I_norm_t, sigma=sigma_I_t)
    print(f"\nThick case (tau_max={TAU_THICK}): a = {a_t:.5f} ± {aerr_t:.5f}")

    # Plot 1: Thick N(mu)
    plot_histogram(centers_t, counts_t, TAU_THICK)

    # Plot 2: Thick I(mu)/I1 + fit
    plot_I_and_fit(centers_t, I_norm_t, sigma_I_t, a_t, aerr_t, TAU_THICK)

    # Thin case
    mu_thin = collect_escaped_mu(N_PHOTONS, tau_max=TAU_THIN, seed=92384, verbose=True)
    centers_th, counts_th, I_norm_th, sigma_I_th = compute_I_over_I1(mu_thin, n_bins=N_BINS)
    a_th, aerr_th = fit_linear_limb(centers_th, I_norm_th, sigma=sigma_I_th)
    print(f"\nThin case (tau_max={TAU_THIN}): a = {a_th:.5f} ± {aerr_th:.5f}")

    # Plot 3: Thin N(mu)
    plot_histogram(centers_th, counts_th, TAU_THIN)

    # Plot 4: Thin I(mu)/I1 + fit
    plot_I_and_fit(centers_th, I_norm_th, sigma_I_th, a_th, aerr_th, TAU_THIN)
