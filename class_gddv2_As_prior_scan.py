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

LOGA_OBS = 3.044

AS_PRIOR_CASES = [
    ("As_fuerte", 0.014),
    ("As_medio", 0.030),
    ("As_debil", 0.060),
    ("Sin_As", None),
]


# =========================
# CLASS
# =========================

def run_class(h, omega_m, logA, xi_gdd, mu_gdd, gdd_mode):
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
        "xi_gdd": xi_gdd,
        "mu_gdd": mu_gdd,
        "gdd_mode": gdd_mode
    })
    cosmo.compute()
    return cosmo, A_s


def fs8_class(cosmo, z):
    return (
        cosmo.scale_independent_growth_factor_f(z)
        * cosmo.sigma8()
        * cosmo.scale_independent_growth_factor(z)
    )


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


def chi2_As(logA, logA_err):
    if logA_err is None:
        return 0.0
    return float(((logA - LOGA_OBS) / logA_err) ** 2)


def xi_equiv_from_mu(mu_eff, omega_m):
    od = omega_m - OMEGA_B
    if od <= 0:
        return np.nan
    return (mu_eff * omega_m - OMEGA_B) / od


# =========================
# Fit
# =========================

def fit_model(model, logA_err):
    if model == "LCDM":
        bounds = [(0.62, 0.75), (0.22, 0.42), (2.6, 3.4)]
        k = 3

    elif model == "GDDv2":
        bounds = [(0.62, 0.75), (0.22, 0.42), (2.6, 3.4), (0.0, 1.0)]
        k = 4

    elif model == "MU_LIBRE":
        bounds = [(0.62, 0.75), (0.22, 0.42), (2.6, 3.4), (0.0, 1.5)]
        k = 4

    else:
        raise ValueError(model)

    def unpack(x):
        if model == "LCDM":
            h, omega_m, logA = x
            xi_gdd = 1.0
            mu_gdd = 1.0
            gdd_mode = 0
        elif model == "GDDv2":
            h, omega_m, logA, xi_gdd = x
            mu_gdd = 1.0
            gdd_mode = 0
        else:
            h, omega_m, logA, mu_gdd = x
            xi_gdd = 1.0
            gdd_mode = 1

        return h, omega_m, logA, xi_gdd, mu_gdd, gdd_mode

    def obj(x):
        h, omega_m, logA, xi_gdd, mu_gdd, gdd_mode = unpack(x)

        if omega_m <= OMEGA_B:
            return 1e30

        try:
            cosmo, _ = run_class(h, omega_m, logA, xi_gdd, mu_gdd, gdd_mode)
            chi_rd_val, _ = chi2_rd(cosmo)

            total = (
                chi2_fs8(cosmo)
                + chi2_s8(cosmo, omega_m)
                + chi_rd_val
                + chi2_ombh2(h)
                + chi2_ommh2(h, omega_m)
                + chi2_As(logA, logA_err)
            )

            cosmo.struct_cleanup()
            cosmo.empty()

            return float(total)

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

    h, omega_m, logA, xi_gdd, mu_gdd, gdd_mode = unpack(res.x)
    cosmo, A_s = run_class(h, omega_m, logA, xi_gdd, mu_gdd, gdd_mode)

    chi_fs8_val = chi2_fs8(cosmo)
    chi_s8_val = chi2_s8(cosmo, omega_m)
    chi_rd_val, rd = chi2_rd(cosmo)
    chi_ob_val = chi2_ombh2(h)
    chi_om_val = chi2_ommh2(h, omega_m)
    chi_As_val = chi2_As(logA, logA_err)

    chi_total = (
        chi_fs8_val
        + chi_s8_val
        + chi_rd_val
        + chi_ob_val
        + chi_om_val
        + chi_As_val
    )

    sigma8 = float(cosmo.sigma8())
    s8 = S8_val(cosmo, omega_m)

    if model == "LCDM":
        mu_eff = 1.0
        xi_equiv = 1.0
    elif model == "GDDv2":
        mu_eff = (OMEGA_B + xi_gdd * (omega_m - OMEGA_B)) / omega_m
        xi_equiv = xi_gdd
    else:
        mu_eff = mu_gdd
        xi_equiv = xi_equiv_from_mu(mu_eff, omega_m)

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
        "S8": s8,
        "rd": float(rd),
        "xi": float(xi_gdd),
        "mu_gdd": float(mu_gdd),
        "mu_eff": float(mu_eff),
        "xi_equiv": float(xi_equiv),
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
# Main scan
# =========================

print("\n================================================")
print("CLASS INTERNO: ROBUSTEZ GDDv2 vs MU_LIBRE")
print("Scan del prior sobre log(1e10 As)")
print("================================================")

summary = []

for case_name, logA_err in AS_PRIOR_CASES:
    print("\n\n################################################")
    print("CASO:", case_name, "| LOGA_ERR =", logA_err)
    print("################################################")

    lcdm = fit_model("LCDM", logA_err)
    gdd = fit_model("GDDv2", logA_err)
    mu = fit_model("MU_LIBRE", logA_err)

    results = [lcdm, gdd, mu]

    for r in results:
        print("\nModelo:", r["model"])
        print("H0           =", r["H0"])
        print("Omega_m      =", r["Omega_m"])
        print("logA         =", r["logA"])
        print("sigma8       =", r["sigma8"])
        print("S8           =", r["S8"])
        print("rd           =", r["rd"])
        print("xi           =", r["xi"])
        print("mu_gdd       =", r["mu_gdd"])
        print("mu_eff       =", r["mu_eff"])
        print("xi_equiv     =", r["xi_equiv"])
        print("Omega_growth =", r["Omega_growth"])
        print("chi_total    =", r["chi_total"])
        print("chi_fs8      =", r["chi_fs8"])
        print("chi_S8       =", r["chi_S8"])
        print("chi_As       =", r["chi_As"])
        print("AIC          =", r["AIC"])
        print("BIC          =", r["BIC"])

    best = min(results, key=lambda x: x["BIC"])

    print("\n==============================")
    print("RESUMEN CASO:", case_name)
    print("==============================")
    print("Mejor BIC =", best["model"])
    print("Delta BIC GDD - LCDM =", gdd["BIC"] - lcdm["BIC"])
    print("Delta BIC MU  - LCDM =", mu["BIC"] - lcdm["BIC"])
    print("Delta BIC MU  - GDD  =", mu["BIC"] - gdd["BIC"])
    print("GDD xi =", gdd["xi"])
    print("GDD mu_eff =", gdd["mu_eff"])
    print("MU xi_equiv =", mu["xi_equiv"])
    print("MU físico GDD? =", 0.0 <= mu["xi_equiv"] <= 1.0)

    summary.append({
        "case": case_name,
        "LOGA_ERR": logA_err,
        "best": best["model"],
        "Delta_BIC_GDD_LCDM": gdd["BIC"] - lcdm["BIC"],
        "Delta_BIC_MU_LCDM": mu["BIC"] - lcdm["BIC"],
        "Delta_BIC_MU_GDD": mu["BIC"] - gdd["BIC"],
        "GDD_xi": gdd["xi"],
        "GDD_mu_eff": gdd["mu_eff"],
        "GDD_S8": gdd["S8"],
        "GDD_logA": gdd["logA"],
        "MU_mu_eff": mu["mu_eff"],
        "MU_xi_equiv": mu["xi_equiv"],
        "MU_physical_GDD": 0.0 <= mu["xi_equiv"] <= 1.0,
    })


print("\n\n================================================")
print("TABLA FINAL ROBUSTEZ")
print("================================================")

for s in summary:
    print("\nCASO:", s["case"])
    print("LOGA_ERR             =", s["LOGA_ERR"])
    print("Mejor BIC            =", s["best"])
    print("Delta BIC GDD-LCDM   =", s["Delta_BIC_GDD_LCDM"])
    print("Delta BIC MU-LCDM    =", s["Delta_BIC_MU_LCDM"])
    print("Delta BIC MU-GDD     =", s["Delta_BIC_MU_GDD"])
    print("GDD xi               =", s["GDD_xi"])
    print("GDD mu_eff           =", s["GDD_mu_eff"])
    print("GDD S8               =", s["GDD_S8"])
    print("GDD logA             =", s["GDD_logA"])
    print("MU mu_eff            =", s["MU_mu_eff"])
    print("MU xi_equiv          =", s["MU_xi_equiv"])
    print("MU físico como GDD?  =", s["MU_physical_GDD"])

print("\nLectura:")
print("- Si GDD gana o empata establemente salvo cuando As queda totalmente libre, hay señal robusta condicionada al anclaje CMB.")
print("- Si al debilitar As GDD pierde siempre, la hipótesis depende demasiado del prior de amplitud.")
print("- Si MU_LIBRE empata con GDD y xi_equiv es físico, MU no destrona a GDD: solo describe la misma supresión.")
