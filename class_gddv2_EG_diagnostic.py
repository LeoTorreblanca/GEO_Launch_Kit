from classy import Class
import numpy as np

OMEGA_B = 0.05

MODELS = {
    "LCDM": {
        "h": 0.6719527376635757,
        "Omega_m": 0.31171194726388524,
        "logA": 3.033232875988148,
        "xi": 1.0,
    },
    "GDDv2_min": {
        "h": 0.6714049044158887,
        "Omega_m": 0.31424565868154797,
        "logA": 3.0440543063084116,
        "xi": 0.7739826180784967,
    }
}


def run_class(params):
    h = params["h"]
    Om = params["Omega_m"]
    xi = params["xi"]
    A_s = np.exp(params["logA"]) / 1e10

    omega_b = OMEGA_B * h**2
    omega_cdm = (Om - OMEGA_B) * h**2

    cosmo = Class()
    cosmo.set({
        "h": h,
        "omega_b": omega_b,
        "omega_cdm": omega_cdm,
        "A_s": A_s,
        "n_s": 0.965,
        "tau_reio": 0.054,
        "output": "mPk",
        "P_k_max_1/Mpc": 2.0,
        "z_max_pk": 2.5,
        "xi_gdd": xi,
        "mu_gdd": 1.0,
        "gdd_mode": 0
    })
    cosmo.compute()
    return cosmo


def mu_eff(Omega_m, xi):
    Omega_d = Omega_m - OMEGA_B
    return (OMEGA_B + xi * Omega_d) / Omega_m


def EG(Omega_m, fz, Sigma=1.0):
    return Omega_m * Sigma / fz


z_vals = [0.1, 0.3, 0.5, 0.7, 1.0]

print("\n===================================")
print("GDDv2-min: diagnostico E_G")
print("===================================")

outputs = {}

for name, p in MODELS.items():
    cosmo = run_class(p)

    Om = p["Omega_m"]
    xi = p["xi"]
    mu = mu_eff(Om, xi)

    print("\nModelo:", name)
    print("Omega_m =", Om)
    print("xi      =", xi)
    print("mu_eff  =", mu)
    print("sigma8  =", cosmo.sigma8())

    rows = []

    print("\nz      f(z)      D(z)      fs8       E_G(Sigma=1)")
    for z in z_vals:
        f = cosmo.scale_independent_growth_factor_f(z)
        D = cosmo.scale_independent_growth_factor(z)
        fs8 = f * cosmo.sigma8() * D
        eg = EG(Om, f, Sigma=1.0)

        rows.append((z, f, D, fs8, eg))

        print(f"{z:4.2f}   {f:8.5f}  {D:8.5f}  {fs8:8.5f}  {eg:8.5f}")

    outputs[name] = rows

    cosmo.struct_cleanup()
    cosmo.empty()


print("\n===================================")
print("RATIOS GDD / LCDM")
print("===================================")

lcdm = outputs["LCDM"]
gdd = outputs["GDDv2_min"]

print("z      f_ratio   fs8_ratio   EG_ratio")
for a, b in zip(lcdm, gdd):
    z = a[0]
    f_ratio = b[1] / a[1]
    fs8_ratio = b[3] / a[3]
    eg_ratio = b[4] / a[4]

    print(f"{z:4.2f}   {f_ratio:8.5f}   {fs8_ratio:8.5f}   {eg_ratio:8.5f}")

print("\nLectura:")
print("- Si fs8_ratio < 1, GDD reduce crecimiento respecto a LCDM.")
print("- Si EG_ratio > 1 con Sigma=1, GDD predice una señal lensing/growth distinta.")
print("- Si datos reales de E_G favorecen LCDM, GDD queda tensionada.")
print("- Si datos reales admiten EG mayor, GDD gana plausibilidad.")
