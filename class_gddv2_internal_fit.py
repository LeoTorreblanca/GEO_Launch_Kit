from classy import Class
import numpy as np
from scipy.optimize import differential_evolution

# =========================
# Datos fσ8
# =========================

FS8_DATA = [
    (0.02, 0.360, 0.040), (0.067, 0.423, 0.055),
    (0.10, 0.370, 0.130), (0.15, 0.490, 0.050),
    (0.17, 0.510, 0.060), (0.22, 0.420, 0.070),
    (0.25, 0.351, 0.058), (0.32, 0.384, 0.095),
    (0.37, 0.460, 0.038), (0.38, 0.430, 0.054),
    (0.44, 0.413, 0.080), (0.51, 0.452, 0.057),
    (0.57, 0.444, 0.038), (0.60, 0.390, 0.063),
    (0.61, 0.457, 0.052), (0.73, 0.437, 0.072),
    (0.80, 0.470, 0.080), (0.86, 0.400, 0.110),
    (1.40, 0.482, 0.116), (1.52, 0.426, 0.077),
    (1.944, 0.364, 0.106)
]

z_data = np.array([x[0] for x in FS8_DATA])
fs8_obs = np.array([x[1] for x in FS8_DATA])
fs8_err = np.array([x[2] for x in FS8_DATA])

# =========================
# Priors
# =========================

OMEGA_B = 0.05

S8_OBS, S8_ERR = 0.776, 0.0325
RD_OBS, RD_ERR = 147.1, 0.3
OMBH2_OBS, OMBH2_ERR = 0.0224, 0.0001
OMMH2_OBS, OMMH2_ERR = 0.143, 0.002

# Prior CMB sobre amplitud primordial
LOGA_OBS = 3.044
LOGA_ERR = 0.014


# =========================
# CLASS
# =========================

def run_class(h, omega_m, logA, xi_gdd):
    A_s = np.exp(logA) / 1e10

    omega_b = OMEGA_B * h**2
    omega_cdm = (omega_m - OMEGA_B) * h**2

    if omega_cdm <= 0:
        raise ValueError("omega_cdm <= 0")

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
        "xi_gdd": xi_gdd
    })
    cosmo.compute()

    return cosmo, A_s


def fs8_class(cosmo, z):
    sigma8_0 = cosmo.sigma8()
    D = cosmo.scale_independent_growth_factor(z)
    f = cosmo.scale_independent_growth_factor_f(z)
    return f * sigma8_0 * D


def chi2_fs8(cosmo):
    pred = np.array([fs8_class(cosmo, z) for z in z_data])
    return float(np.sum(((fs8_obs - pred) / fs8_err) ** 2))


def S8_val(cosmo, omega_m):
    return float(cosmo.sigma8() * np.sqrt(omega_m / 0.3))


def chi2_s8(cosmo, omega_m):
    return float(((S8_val(cosmo, omega_m) - S8_OBS) / S8_ERR) ** 2)


def chi2_rd(cosmo):
    try:
        rd = float(cosmo.rs_drag())
    except Exception:
        rd = RD_OBS

    return float(((rd - RD_OBS) / RD_ERR) ** 2), rd


def chi2_ombh2(h):
    return float(((OMEGA_B * h**2 - OMBH2_OBS) / OMBH2_ERR) ** 2)


def chi2_ommh2(h, omega_m):
    return float(((omega_m * h**2 - OMMH2_OBS) / OMMH2_ERR) ** 2)


def chi2_As(logA):
    return float(((logA - LOGA_OBS) / LOGA_ERR) ** 2)


# =========================
# Fit
# =========================

def fit_model(model):
    if model == "LCDM":
        bounds = [
            (0.62, 0.75),   # h
            (0.22, 0.42),   # Omega_m
            (2.8, 3.3),     # log(1e10 A_s)
        ]
    elif model == "GDDv2":
        bounds = [
            (0.62, 0.75),   # h
            (0.22, 0.42),   # Omega_m
            (2.8, 3.3),     # log(1e10 A_s)
            (0.0, 1.0),     # xi_gdd
        ]
    else:
        raise ValueError(model)

    def obj(x):
        if model == "LCDM":
            h, omega_m, logA = x
            xi_gdd = 1.0
        else:
            h, omega_m, logA, xi_gdd = x

        if omega_m <= OMEGA_B:
            return 1e30

        try:
            cosmo, _ = run_class(h, omega_m, logA, xi_gdd)
            chi_rd_val, _ = chi2_rd(cosmo)

            val = (
                chi2_fs8(cosmo)
                + chi2_s8(cosmo, omega_m)
                + chi_rd_val
                + chi2_ombh2(h)
                + chi2_ommh2(h, omega_m)
                + chi2_As(logA)
            )

            cosmo.struct_cleanup()
            cosmo.empty()

            return float(val)

        except Exception:
            return 1e30

    res = differential_evolution(
        obj,
        bounds,
        seed=123,
        polish=True,
        tol=1e-4,
        maxiter=45,
        popsize=8
    )

    if model == "LCDM":
        h, omega_m, logA = res.x
        xi_gdd = 1.0
    else:
        h, omega_m, logA, xi_gdd = res.x

    cosmo, A_s = run_class(h, omega_m, logA, xi_gdd)

    chi_fs8_val = chi2_fs8(cosmo)
    chi_s8_val = chi2_s8(cosmo, omega_m)
    chi_rd_val, rd = chi2_rd(cosmo)
    chi_ob_val = chi2_ombh2(h)
    chi_om_val = chi2_ommh2(h, omega_m)
    chi_As_val = chi2_As(logA)

    chi_total = (
        chi_fs8_val
        + chi_s8_val
        + chi_rd_val
        + chi_ob_val
        + chi_om_val
        + chi_As_val
    )

    sigma8 = float(cosmo.sigma8())
    S8 = S8_val(cosmo, omega_m)

    if model == "LCDM":
        mu_eff = 1.0
    else:
        mu_eff = (OMEGA_B + xi_gdd * (omega_m - OMEGA_B)) / omega_m

    k = 3 if model == "LCDM" else 4
    n = len(FS8_DATA) + 5

    AIC = chi_total + 2 * k
    BIC = chi_total + k * np.log(n)

    result = {
        "model": model,
        "h": float(h),
        "H0": float(100 * h),
        "Omega_m": float(omega_m),
        "Omega_b": float(OMEGA_B),
        "Omega_d": float(omega_m - OMEGA_B),
        "logA": float(logA),
        "A_s": float(A_s),
        "sigma8": sigma8,
        "S8": S8,
        "rd": float(rd),
        "xi": float(xi_gdd),
        "mu_eff": float(mu_eff),
        "Omega_growth": float(mu_eff * omega_m),
        "chi_total": float(chi_total),
        "chi_fs8": float(chi_fs8_val),
        "chi_S8": float(chi_s8_val),
        "chi_rd": float(chi_rd_val),
        "chi_ombh2": float(chi_ob_val),
        "chi_ommh2": float(chi_om_val),
        "chi_As": float(chi_As_val),
        "AIC": float(AIC),
        "BIC": float(BIC),
    }

    cosmo.struct_cleanup()
    cosmo.empty()

    return result


# =========================
# Main
# =========================

print("\n==============================")
print("CLASS INTERNO GDDv2 — AJUSTE As + xi + PRIOR As")
print("==============================")

lcdm = fit_model("LCDM")
gdd = fit_model("GDDv2")

for r in [lcdm, gdd]:
    print("\nModelo:", r["model"])
    print("H0           =", r["H0"])
    print("h            =", r["h"])
    print("Omega_m      =", r["Omega_m"])
    print("Omega_b      =", r["Omega_b"])
    print("Omega_d      =", r["Omega_d"])
    print("log(1e10 As) =", r["logA"])
    print("A_s          =", r["A_s"])
    print("sigma8       =", r["sigma8"])
    print("S8           =", r["S8"])
    print("rd           =", r["rd"])
    print("xi           =", r["xi"])
    print("mu_eff       =", r["mu_eff"])
    print("Omega_growth =", r["Omega_growth"])
    print("chi_total    =", r["chi_total"])
    print("chi_fs8      =", r["chi_fs8"])
    print("chi_S8       =", r["chi_S8"])
    print("chi_rd       =", r["chi_rd"])
    print("chi_ombh2    =", r["chi_ombh2"])
    print("chi_ommh2    =", r["chi_ommh2"])
    print("chi_As       =", r["chi_As"])
    print("AIC          =", r["AIC"])
    print("BIC          =", r["BIC"])

print("\n==============================")
print("COMPARACIÓN")
print("==============================")
print("Delta chi2 GDD - LCDM =", gdd["chi_total"] - lcdm["chi_total"])
print("Delta AIC  GDD - LCDM =", gdd["AIC"] - lcdm["AIC"])
print("Delta BIC  GDD - LCDM =", gdd["BIC"] - lcdm["BIC"])
