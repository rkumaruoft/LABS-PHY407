"""
Monte Carlo limb-darkening simulation (part b)
Simulate 1e5 escaped photons for tau_max=10, compute I(mu)/I1, fit linear limb-darkening.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def get_tau_step(rng):
    return -np.log(rng.random())

def emit_photon(tau_max, rng):
    """Emit photon from core at tau_max with outward mu in [0,1]. Take initial free path."""
    tau = tau_max
    mu = rng.random()  # outward hemisphere [0,1]
    delta_tau = get_tau_step(rng)
    tau_new = tau - delta_tau * mu
    return tau_new, mu

def scatter_photon(tau, rng):
    """Isotropic scattering: new mu uniform in [-1,1], then take step."""
    mu = 2.0 * rng.random() - 1.0
    delta_tau = get_tau_step(rng)
    tau_new = tau - delta_tau * mu
    return tau_new, mu

def simulate_one_photon(tau_max, rng, max_scatter=10000):
    """Return (mu_last, n_scatter) if photon escapes; return None if reabsorbed or max_scatter reached."""
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

def collect_escaped_mu(n_escaped, tau_max=10.0, seed=None, verbose=True):
    rng = np.random.default_rng(seed)
    mu_list = []
    attempts = 0
    while len(mu_list) < n_escaped:
        attempts += 1
        res = simulate_one_photon(tau_max, rng)
        if res is None:
            continue
        mu_last, _ = res
        # Only outward final directions are physically relevant for intensity vs mu (mu>0)
        # If mu_last < 0 (escaped but pointing inward) it still left the atmosphere; keep sign.
        # For histogram of emergent intensity vs mu we consider mu in (0,1].
        if mu_last > 0:
            mu_list.append(mu_last)
        # If mu_last <= 0, photon left but heading inward relative to local normal; rare but ignore for N(mu) vs mu>0
        if verbose and (len(mu_list) % (n_escaped//10) == 0):
            print(f"Collected {len(mu_list)}/{n_escaped} escaped photons (attempts {attempts})")
    return np.array(mu_list)

def compute_I_over_I1(mu_array, n_bins=20):
    """Compute N(mu) histogram, then I(mu) proportional to N(mu)/mu_center, normalized so I(1)=1."""
    bins = np.linspace(0.0, 1.0, n_bins+1)
    counts, edges = np.histogram(mu_array, bins=bins)
    # Avoid the zero-width first bin center at mu=0; use bin centers
    centers = 0.5*(edges[:-1] + edges[1:])
    # For bins with center mu=0 (first bin), avoid division by zero: skip or set to nan
    with np.errstate(divide='ignore', invalid='ignore'):
        I_proxy = counts / centers
    # Replace infinities/nans (first bin) by using the value from the next bin (simple fix)
    if np.isnan(I_proxy[0]) or np.isinf(I_proxy[0]):
        I_proxy[0] = I_proxy[1]
    # Normalize so that I(mu=1) = 1. Use the last bin center (closest to mu=1)
    I_norm = I_proxy / I_proxy[-1]
    return centers, counts, I_norm

def linear_limb(mu, a):
    """Linear limb-darkening law: I(mu)/I1 = 1 - a*(1-mu)."""
    return 1.0 - a*(1.0 - mu)

def fit_linear_limb(centers, I_norm):
    # Fit only bins with centers > 0 (exclude mu=0)
    mask = centers > 0
    x = centers[mask]
    y = I_norm[mask]
    popt, pcov = curve_fit(linear_limb, x, y, p0=[0.5])
    a = popt[0]
    a_err = np.sqrt(np.diag(pcov))[0]
    return a, a_err

if __name__ == "__main__":
    # Parameters
    N_PHOTONS = 100000   # number of escaped photons to collect
    TAU_MAX = 10.0
    SEED = 12345

    # Collect final mu for escaped photons
    mu_final = collect_escaped_mu(N_PHOTONS, tau_max=TAU_MAX, seed=SEED, verbose=True)

    # Compute histogram and I(mu)/I1
    n_bins = 20
    centers, counts, I_norm = compute_I_over_I1(mu_final, n_bins=n_bins)

    # Fit linear limb-darkening law
    a_fit, a_err = fit_linear_limb(centers, I_norm)
    print(f"\nFitted linear limb-darkening coefficient: a = {a_fit:.4f} ± {a_err:.4f}")

    # Plot results
    mu_plot = np.linspace(0.01, 1.0, 200)  # avoid exactly 0 for plotting analytic curve
    I_fit_curve = linear_limb(mu_plot, a_fit)

    plt.figure(figsize=(8,5))
    plt.errorbar(centers, I_norm, yerr=np.sqrt(counts)/np.maximum(centers, 1e-6)/counts.max()*I_norm.max(),
                 fmt='o', label='Monte Carlo binned $I(\\mu)/I_1$', capsize=3)
    plt.plot(mu_plot, I_fit_curve, '-', color='C1', lw=2, label=f'Linear fit: $1 - a(1-\\mu)$, a={a_fit:.4f}')
    plt.xlabel(r'$\mu$ (cosine of emergent angle)')
    plt.ylabel(r'$I(\mu)/I_1$ (normalized)')
    plt.title('Limb-darkening: Monte Carlo estimate and linear fit')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
