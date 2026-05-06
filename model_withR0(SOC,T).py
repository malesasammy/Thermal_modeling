import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from scipy.optimize import minimize
import time

from r0_model import get_R0

# =============================================================================
# DATABASE
# =============================================================================
engine = create_engine(
    "postgresql://postgres:YIB8dgucw20aPBd0BUuaYsdo5IXC3PzN@143.198.122.92:15432/cell_data"
)

# =============================================================================
# GEOMETRY / PROPERTIES
# =============================================================================
r_pvc_outer = 0.049
r_jr = 0.047
r_can_inner = 0.05024
r_can_outer = 0.05128
H_can = 0.220

A_conv = 2 * np.pi * r_can_outer * H_can

rho_pvc, Cp_pvc = 1380.0, 900.0
rho_abs, Cp_abs = 1050.0, 1500.0

V_pvc = np.pi * (r_pvc_outer**2 - r_jr**2) * H_can
V_abs = np.pi * (r_can_outer**2 - r_can_inner**2) * H_can

C_surface = rho_pvc * V_pvc * Cp_pvc + rho_abs * V_abs * Cp_abs

m_jr = 3.0
Cp_jr = 1400.0
C_core = m_jr * Cp_jr   # FIXED

dUdT = -0.0001

h_RT = 1.461
h_40C = 2.438

print(f"C_core = {C_core:.0f} J/K (fixed)")
print(f"C_surface = {C_surface:.1f} J/K")

# =============================================================================
# TEST CONFIGURATION
# =============================================================================
TESTS = {
    "20A_RT": {
        "cell_id": "12447-1",
        "surf_id": "12448-1",
        "h_conv": h_RT,
        "T_amb": 25.0,
    },
    "40A_RT": {
        "cell_id": "12449-1",
        "surf_id": "12450-1",
        "h_conv": h_RT,
        "T_amb": 25.0,
    },
    "20A_40C": {
        "cell_id": "12451-1",
        "surf_id": "12452-1",
        "h_conv": h_40C,
        "T_amb": 40.0,
    },
    "40A_40C": {
        "cell_id": "12453-1",
        "surf_id": "12455-1",
        "h_conv": h_40C,
        "T_amb": 40.0,
    },
}

# =============================================================================
# DATA
# =============================================================================
def load_elec(cell_id):
    return pd.read_sql(f"""
        SELECT test_time, test_time/3600.0 AS t_h,
               current, voltage
        FROM pec_timeseries
        WHERE cell_id='{cell_id}'
        ORDER BY test_time
    """, engine).dropna().reset_index(drop=True)


def load_temp(cell_id, name):
    df = pd.read_sql(f"""
        SELECT test_time/3600.0 AS t_h, temperature
        FROM pec_temperature
        WHERE cell_id='{cell_id}'
    """, engine).dropna().rename(columns={"temperature": name})
    return df.reset_index(drop=True)


def build(cell_id, surf_id, T_amb):

    e = load_elec(cell_id)
    tc = load_temp(cell_id, "T_center")
    ts = load_temp(surf_id, "T_surface")

    df = pd.merge_asof(e.sort_values("t_h"),
                       tc.sort_values("t_h"),
                       on="t_h")

    df = pd.merge_asof(df.sort_values("t_h"),
                       ts.sort_values("t_h"),
                       on="t_h")

    df = df.dropna().reset_index(drop=True)

    df["t_s"] = df["test_time"] - df["test_time"].iloc[0]

    dt = np.diff(df["t_s"], prepend=df["t_s"].iloc[0])
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    df["dt"] = np.clip(dt, 1e-6, 30)

    df["SOC"] = np.linspace(1, 0, len(df))

    return df


# =============================================================================
# MODEL (FAST + CORRECT R0)
# =============================================================================
def run_model(df, h_conv, R_cs):

    n = len(df)

    Tc = np.zeros(n)
    Ts = np.zeros(n)

    Tc[0] = df["T_center"].iloc[0]
    Ts[0] = df["T_surface"].iloc[0]

    G_cs = 1.0 / max(R_cs, 1e-6)
    hA = h_conv * A_conv

    # cache R0 ONCE
    R0_cache = np.zeros(n)
    for i in range(n):
        R0_cache[i] = get_R0(Tc[0], df["SOC"].iloc[i])

    for i in range(1, n):

        I = df["current"].iloc[i]

        q = (I**2) * R0_cache[i] + I * (Tc[i-1] + 273.15) * dUdT

        Qcond = (Tc[i-1] - Ts[i-1]) * G_cs
        Qconv = hA * (Ts[i-1] - 25)

        Tc[i] = Tc[i-1] + df["dt"].iloc[i] * (q - Qcond) / C_core
        Ts[i] = Ts[i-1] + df["dt"].iloc[i] * (Qcond - Qconv) / C_surface

    return Tc, Ts


# =============================================================================
# CALIBRATION (ONLY R_cs)
# =============================================================================
def calibrate(data):

    def obj(x):
        R_cs = x[0]
        err = 0

        for d in data.values():
            Tc, Ts = run_model(d["df"], d["h_conv"], R_cs)

            err += np.mean((Tc - d["df"]["T_center"])**2)
            err += np.mean((Ts - d["df"]["T_surface"])**2)

        return err

    grid = np.linspace(0.5, 8, 10)
    best = min(grid, key=lambda r: obj([r]))

    res = minimize(obj, [best], method="Powell", options={"maxiter": 60})

    return float(res.x[0])


# =============================================================================
# ORIGINAL PLOT (UNCHANGED STRUCTURE)
# =============================================================================
def plot_results(results, R_cs):

    fig, ax = plt.subplots(2, 2, figsize=(17, 11))
    ax = ax.flatten()

    order = list(results.keys())

    for i, name in enumerate(order):

        r = results[name]
        df = r["df"]

        t = df["t_h"].values

        window = 300
        Tc_meas = df["T_center"].rolling(window, center=True, min_periods=1).mean()
        Ts_meas = df["T_surface"].rolling(window, center=True, min_periods=1).mean()

        ax[i].plot(t, Tc_meas, "k", lw=1.5, label="Meas T_center")
        ax[i].plot(t, Ts_meas, "gray", lw=1.2, ls="--", label="Meas T_surface")

        ax[i].plot(t, r["Tc"], "b", lw=1.5, label="Model T_center")
        ax[i].plot(t, r["Ts"], "orange", lw=1.5, label="Model T_surface")

        ax[i].set_title(name)
        ax[i].set_xlabel("Time (h)")
        ax[i].set_ylabel("Temp (°C)")
        ax[i].legend()

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
def main():

    results = {}

    for name, cfg in TESTS.items():

        df = build(cfg["cell_id"], cfg["surf_id"], cfg["T_amb"])
        Tc, Ts = run_model(df, cfg["h_conv"], 2.0)  # initial guess

        results[name] = {
            "df": df,
            "Tc": Tc,
            "Ts": Ts,
            "h_conv": cfg["h_conv"]
        }

    print("Calibrating R_cs...")
    R_cs = calibrate(results)

    print("R_cs =", R_cs)

    # final run
    for name, cfg in TESTS.items():
        df = results[name]["df"]
        Tc, Ts = run_model(df, cfg["h_conv"], R_cs)
        results[name]["Tc"] = Tc
        results[name]["Ts"] = Ts

    plot_results(results, R_cs)


if __name__ == "__main__":
    main()