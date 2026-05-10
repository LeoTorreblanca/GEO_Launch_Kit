from classy import Class

def run(xi):
    cosmo = Class()
    cosmo.set({
        "h": 0.67,
        "omega_b": 0.022,
        "omega_cdm": 0.12,
        "A_s": 2.1e-9,
        "n_s": 0.965,
        "tau_reio": 0.054,
        "output": "mPk",
        "P_k_max_1/Mpc": 2.0,
        "z_max_pk": 2.0,
        "xi_gdd": xi
    })
    cosmo.compute()

    print("\nxi_gdd =", xi)
    print("sigma8 =", cosmo.sigma8())
    print("pk(0.1,z=0) =", cosmo.pk(0.1, 0.0))
    print("pk(0.1,z=1) =", cosmo.pk(0.1, 1.0))

    cosmo.struct_cleanup()
    cosmo.empty()

run(1.0)
run(0.07)
run(0.0)
