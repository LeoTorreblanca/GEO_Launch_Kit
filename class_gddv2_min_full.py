from classy import Class
import numpy as np
import os, re, glob
from scipy.optimize import differential_evolution
from scipy.linalg import cho_factor, cho_solve

# ==================================================
# RUTAS
# ==================================================

GDD_ROOT = "/mnt/c/Users/leone/Desktop/GDD"
BAO_ROOT = os.path.join(GDD_ROOT, "bao_data-master")
SN_FILE = os.path.join(GDD_ROOT, "Pantheon+SH0ES.dat")
SN_COV_FILE = os.path.join(GDD_ROOT, "Pantheon+SH0ES_STAT+SYS.cov")

# ==================================================
# CONSTANTES / PRIORS
# ==================================================

OMEGA_B = 0.05
c = 299792.458

S8_OBS, S8_ERR = 0.776, 0.0325
RD_OBS, RD_ERR = 147.1, 0.3
OMBH2_OBS, OMBH2_ERR = 0.0224, 0.0001
OMMH2_OBS, OMMH2_ERR = 0.143, 0.002
LOGA_OBS, LOGA_ERR = 3.044, 0.014

# ==================================================
# fσ8 DATA
# ==================================================

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

z_fs8 = np.array([x[0] for x in FS8_DATA])
fs8_obs = np.array([x[1] for x in FS8_DATA])
fs8_err = np.array([x[2] for x in FS8_DATA])

# ==================================================
# PANTHEON
# ==================================================

def load_pantheon():
    data = np.genfromtxt(SN_FILE, names=True, dtype=None, encoding=None)

    names = data.dtype.names

    z_col = "zHD" if "zHD" in names else ("zcmb" if "zcmb" in names else names[1])

    if "m_b_corr" in names:
        mag_col = "m_b_corr"
    elif "MU_SH0ES" in names:
        mag_col = "MU_SH0ES"
    elif "MU" in names:
        mag_col = "MU"
    else:
        raise ValueError("No encuentro columna de magnitud/MU en Pantheon")

    z = np.array(data[z_col], dtype=float)
    m = np.array(data[mag_col], dtype=float)

    raw = np.loadtxt(SN_COV_FILE)
    if raw.ndim == 1:
        if int(raw[0]) == len(z):
            cov = raw[1:].reshape(len(z), len(z))
        else:
            cov = raw.reshape(len(z), len(z))
    else:
        cov = raw

    cho = cho_factor(cov, lower=True, check_finite=False)
    ones = np.ones(len(z))

    print("Pantheon cargado:", len(z))

    return z, m, cho, ones


z_sn, m_sn, cov_sn_cho, ones_sn = load_pantheon()


def chi2_sn(cosmo):
    dl = np.array([cosmo.luminosity_distance(float(z)) for z in z_sn])
    mu_th = 5.0 * np.log10(dl) + 25.0

    diff = m_sn - mu_th

    # Marginalización analítica de offset absoluto M
    Cinv_d = cho_solve(cov_sn_cho, diff, check_finite=False)
    Cinv_1 = cho_solve(cov_sn_cho, ones_sn, check_finite=False)

    a = float(diff @ Cinv_d)
    b = float(diff @ Cinv_1)
    cval = float(ones_sn @ Cinv_1)

    return a - b*b/cval

# ==================================================
# BAO DESI
# ==================================================

def read_numbers(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            nums = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", line)
            if nums:
                rows.append([float(x) for x in nums])

    if not rows:
        raise ValueError("sin numeros")

    lens = [len(r) for r in rows]
    if len(set(lens)) == 1:
        arr = np.array(rows)
        if arr.shape[1] == 1:
            return arr[:, 0]
        return arr

    return np.array([x for r in rows for x in r])


def infer_z(path, mean_raw=None):
    name = os.path.basename(path).lower()
    arr = np.asarray(mean_raw) if mean_raw is not None else None

    if arr is not None and arr.ndim == 1 and len(arr) == 2:
        if 0.01 < arr[0] < 4:
            return float(arr[0])

    if arr is not None and arr.ndim == 2 and arr.shape[1] >= 2:
        z_col = arr[:, 0]
        if np.all((z_col > 0.01) & (z_col < 4)):
            return float(np.mean(z_col))

    m = re.search(r"z([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)", name)
    if m:
        return 0.5 * (float(m.group(1)) + float(m.group(2)))

    if "bgs" in name:
        return 0.295
    if "lrg+elg" in name:
        return 0.934
    if "elg" in name:
        return 1.321
    if "qso" in name:
        return 1.484
    if "lya" in name:
        return 2.33

    return None


def parse_bao_pair(mean_path, cov_path):
    mean_raw = read_numbers(mean_path)
    cov_raw = read_numbers(cov_path)

    name = os.path.basename(mean_path).lower()
    z = infer_z(mean_path, mean_raw)

    mean_arr = np.asarray(mean_raw)
    cov_arr = np.asarray(cov_raw)

    if mean_arr.ndim == 1 and len(mean_arr) == 2 and cov_arr.ndim == 1 and len(cov_arr) == 1:
        return {
            "z": z,
            "obs": ["DV"],
            "mean": np.array([mean_arr[1]]),
            "cov": np.array([[cov_arr[0]]]),
            "name": os.path.basename(mean_path)
        }

    if mean_arr.ndim == 2 and mean_arr.shape[1] >= 2:
        mean_vec = mean_arr[:, -1]
    else:
        mean_vec = mean_arr.flatten()

    if cov_arr.ndim == 1:
        if len(cov_arr) == len(mean_vec):
            cov = np.diag(cov_arr)
        elif len(cov_arr) == len(mean_vec)**2:
            cov = cov_arr.reshape(len(mean_vec), len(mean_vec))
        else:
            raise ValueError("cov incompatible")
    else:
        cov = cov_arr

    if "lya" in name:
        obs = ["DH", "DM"]
    elif len(mean_vec) == 2:
        obs = ["DM", "DH"]
    elif len(mean_vec) == 1:
        obs = ["DV"]
    else:
        raise ValueError("obs no interpretable")

    return {
        "z": z,
        "obs": obs,
        "mean": mean_vec,
        "cov": cov,
        "name": os.path.basename(mean_path)
    }


def load_bao(token):
    files = glob.glob(os.path.join(BAO_ROOT, "**", "*"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]

    means = [f for f in files if f.lower().endswith("_mean.txt") or f.lower().endswith("_mean")]
    covs = [f for f in files if f.lower().endswith("_cov.txt") or f.lower().endswith("_cov")]

    entries = []

    for mf in means:
        low = mf.lower()
        if token not in low:
            continue
        if "all_gccomb" in low:
            continue

        base_m = re.sub(r"_mean(\.txt)?$", "", mf, flags=re.I)

        for cf in covs:
            base_c = re.sub(r"_cov(\.txt)?$", "", cf, flags=re.I)
            if os.path.normcase(base_m) == os.path.normcase(base_c):
                try:
                    entries.append(parse_bao_pair(mf, cf))
                except Exception as e:
                    print("SKIP BAO:", os.path.basename(mf), e)
                break

    return entries


BAO = load_bao("desi_gaussian_bao")
print("BAO DR2 puntos:", sum(len(e["mean"]) for e in BAO))


def bao_predict(cosmo, z, obs, rd):
    DM = (1.0 + z) * cosmo.angular_distance(z)
    DH = 1.0 / cosmo.Hubble(z)
    DV = (z * DM * DM * DH) ** (1.0 / 3.0)

    if obs == "DM":
        return DM / rd
    if obs == "DH":
        return DH / rd
    if obs == "DV":
        return DV / rd

    raise ValueError(obs)


def chi2_bao(cosmo):
    try:
        rd = float(cosmo.rs_drag())
    except Exception:
        rd = RD_OBS

    total = 0.0
    for e in BAO:
        pred = np.array([bao_predict(cosmo, e["z"], ob, rd) for ob in e["obs"]])
        diff = e["mean"] - pred
        inv = np.linalg.pinv(e["cov"])
        total += float(diff.T @ inv @ diff)

    return total

# ==================================================
# CLASS + LIKELIHOODS
# ==================================================

def run_class(h, omega_m, logA, xi_gdd, model):
    A_s = np.exp(logA) / 1e10
    omega_b = OMEGA_B * h**2
    omega_cdm = (omega_m - OMEGA_B) * h**2

    if omega_cdm <= 0:
        raise ValueError("omega_cdm <= 0")

    if model == "LCDM":
        xi_gdd = 1.0

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
        "mu_gdd": 1.0,
        "gdd_mode": 0
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
    pred = np.array([fs8_class(cosmo, z) for z in z_fs8])
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


# ==================================================
# FIT
# ==================================================

def fit_model(model):
    if model == "LCDM":
        bounds = [
            (0.62, 0.75),
            (0.22, 0.42),
            (2.8, 3.3)
        ]
        k = 3
    elif model == "GDDv2_min":
        bounds = [
            (0.62, 0.75),
            (0.22, 0.42),
            (2.8, 3.3),
            (0.0, 1.0)
        ]
        k = 4
    else:
        raise ValueError(model)

    def unpack(x):
        if model == "LCDM":
            h, omega_m, logA = x
            xi = 1.0
        else:
            h, omega_m, logA, xi = x
        return h, omega_m, logA, xi

    def obj(x):
        h, omega_m, logA, xi = unpack(x)

        if omega_m <= OMEGA_B:
            return 1e30

        try:
            cosmo, _ = run_class(h, omega_m, logA, xi, model)
            chi_rd_val, _ = chi2_rd(cosmo)

            val = (
                chi2_bao(cosmo)
                + chi2_fs8(cosmo)
                + chi2_sn(cosmo)
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
        maxiter=35,
        popsize=7
    )

    h, omega_m, logA, xi = unpack(res.x)
    cosmo, A_s = run_class(h, omega_m, logA, xi, model)

    chi_bao_val = chi2_bao(cosmo)
    chi_fs8_val = chi2_fs8(cosmo)
    chi_sn_val = chi2_sn(cosmo)
    chi_s8_val = chi2_s8(cosmo, omega_m)
    chi_rd_val, rd = chi2_rd(cosmo)
    chi_ob_val = chi2_ombh2(h)
    chi_om_val = chi2_ommh2(h, omega_m)
    chi_As_val = chi2_As(logA)

    chi_total = (
        chi_bao_val
        + chi_fs8_val
        + chi_sn_val
        + chi_s8_val
        + chi_rd_val
        + chi_ob_val
        + chi_om_val
        + chi_As_val
    )

    sigma8 = float(cosmo.sigma8())
    s8 = S8_val(cosmo, omega_m)

    mu_eff = 1.0 if model == "LCDM" else (OMEGA_B + xi * (omega_m - OMEGA_B)) / omega_m

    n = (
        sum(len(e["mean"]) for e in BAO)
        + len(FS8_DATA)
        + len(z_sn)
        + 5
    )

    AIC = chi_total + 2*k
    BIC = chi_total + k*np.log(n)

    result = {
        "model": model,
        "H0": float(100*h),
        "h": float(h),
        "Omega_m": float(omega_m),
        "Omega_b": float(OMEGA_B),
        "Omega_d": float(omega_m - OMEGA_B),
        "logA": float(logA),
        "A_s": float(A_s),
        "sigma8": sigma8,
        "S8": s8,
        "rd": float(rd),
        "xi": float(xi),
        "mu_eff": float(mu_eff),
        "Omega_growth": float(mu_eff * omega_m),
        "chi_total": float(chi_total),
        "chi_BAO": float(chi_bao_val),
        "chi_fs8": float(chi_fs8_val),
        "chi_SN": float(chi_sn_val),
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


# ==================================================
# MAIN
# ==================================================

print("\n================================================")
print("GDDv2-min FULL TEST")
print("CLASS interno + BAO DR2 + Pantheon + fσ8 + S8 + prior As")
print("================================================")

lcdm = fit_model("LCDM")
gdd = fit_model("GDDv2_min")

for r in [lcdm, gdd]:
    print("\nModelo:", r["model"])
    print("H0            =", r["H0"])
    print("Omega_m       =", r["Omega_m"])
    print("Omega_b       =", r["Omega_b"])
    print("Omega_d       =", r["Omega_d"])
    print("logA          =", r["logA"])
    print("A_s           =", r["A_s"])
    print("sigma8        =", r["sigma8"])
    print("S8            =", r["S8"])
    print("rd            =", r["rd"])
    print("xi            =", r["xi"])
    print("mu_eff        =", r["mu_eff"])
    print("Omega_growth  =", r["Omega_growth"])
    print("chi_total     =", r["chi_total"])
    print("chi_BAO       =", r["chi_BAO"])
    print("chi_fs8       =", r["chi_fs8"])
    print("chi_SN        =", r["chi_SN"])
    print("chi_S8        =", r["chi_S8"])
    print("chi_rd        =", r["chi_rd"])
    print("chi_ombh2     =", r["chi_ombh2"])
    print("chi_ommh2     =", r["chi_ommh2"])
    print("chi_As        =", r["chi_As"])
    print("AIC           =", r["AIC"])
    print("BIC           =", r["BIC"])

print("\n==============================")
print("COMPARACIÓN")
print("==============================")
print("Delta chi2 GDD - LCDM =", gdd["chi_total"] - lcdm["chi_total"])
print("Delta AIC  GDD - LCDM =", gdd["AIC"] - lcdm["AIC"])
print("Delta BIC  GDD - LCDM =", gdd["BIC"] - lcdm["BIC"])

print("\nLectura:")
print("- Si GDDv2-min mantiene Delta BIC negativo con BAO+SN+growth+S8+As, la versión mínima queda muy fortalecida.")
print("- Si pierde, la señal estaba principalmente en growth/S8 y falta pulir fondo/lensing.")
