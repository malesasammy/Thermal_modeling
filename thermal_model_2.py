import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from scipy.optimize import minimize
import time

# =============================================================================
# DATABASE
# =============================================================================
engine = create_engine(
    "postgresql://postgres:YIB8dgucw20aPBd0BUuaYsdo5IXC3PzN@143.198.122.92:15432/cell_data"
)

# =============================================================================
# GEOMETRY & PROPERTIES
# =============================================================================
r_pvc_outer = 0.049;   r_jr        = 0.047
r_can_inner = 0.05024; r_can_outer = 0.05128
H_can       = 0.220

A_conv    = 2 * np.pi * r_can_outer * H_can
rho_pvc   = 1380.0;  Cp_pvc = 900.0
rho_abs   = 1050.0;  Cp_abs = 1500.0
V_pvc     = np.pi * (r_pvc_outer**2 - r_jr**2)        * H_can
V_abs     = np.pi * (r_can_outer**2 - r_can_inner**2)  * H_can
C_surface = rho_pvc * V_pvc * Cp_pvc + rho_abs * V_abs * Cp_abs

h_RT  = 1.461;  R0_RT  = 0.002028
h_40C = 2.438;  R0_40C = 0.001967
dUdT  = -0.0001

print(f"C_surface={C_surface:.1f} J/K  A_conv={A_conv:.5f} m²")

REST_STEPS = {8, 11}

# =============================================================================
# TEST CONFIGURATION
# cell_id    — cycling data + center TC (same channel)
# surf_id    — surface TC (dedicated channel, no cell)
# =============================================================================
TESTS = {
    "20A_RT": {
        "cell_id": "12447-1", "surf_id": "12448-1",
        "R0": R0_RT,  "h_conv": h_RT,  "label": "20 A — RT",
        "T_amb": 25.0, "t_max_h": 150.0,
    },
    "40A_RT": {
        "cell_id": "12449-1", "surf_id": "12450-1",
        "R0": R0_RT,  "h_conv": h_RT,  "label": "40 A — RT",
        "T_amb": 25.0, "t_max_h": 150.0,
    },
    "20A_40C": {
        "cell_id": "12451-1", "surf_id": "12452-1",
        "R0": R0_40C, "h_conv": h_40C, "label": "20 A — 40°C",
        "T_amb": 40.0, "t_max_h": 150.0,
    },
    "40A_40C": {
        "cell_id": "12453-1", "surf_id": "12455-1",
        "R0": R0_40C, "h_conv": h_40C, "label": "40 A — 40°C",
        "T_amb": 40.0, "t_max_h": 150.0,
    },
}


# =============================================================================
# DATA LOADING
# =============================================================================
def load_electrical(cell_id):
    return pd.read_sql(f"""
        SELECT test_time, test_time/3600.0 AS t_h,
               cycle_index, step_index, current, voltage
        FROM pec_timeseries
        WHERE cell_id = '{cell_id}'
        ORDER BY test_time
    """, engine).dropna(subset=["current","voltage"]).reset_index(drop=True)


def load_temperature(cell_id, col_name, t_max_h=None):
    df = pd.read_sql(f"""
        SELECT test_time/3600.0 AS t_h, temperature
        FROM pec_temperature
        WHERE cell_id = '{cell_id}' AND temperature > -50
        ORDER BY test_time
    """, engine).dropna().rename(columns={"temperature": col_name})
    if t_max_h:
        df = df[df["t_h"] <= t_max_h]
    return df.reset_index(drop=True)


def compute_vocv_end_of_rest(df):
    rest = df[df["step_index"].isin(REST_STEPS)].copy()
    vocv_rows = []
    for (cyc, step), grp in rest.groupby(["cycle_index", "step_index"]):
        n_tail = max(1, len(grp) // 5)
        v_ocv  = grp["voltage"].iloc[-n_tail:].mean()
        vocv_rows.append({"cycle_index": cyc, "V_ocv": v_ocv})

    vocv_df = pd.DataFrame(vocv_rows).groupby("cycle_index")["V_ocv"].mean()
    df = df.join(vocv_df.rename("V_ocv"), on="cycle_index")
    df["V_ocv"] = df["V_ocv"].ffill().bfill()
    df["Q_base"] = (
        df["current"].abs() * (df["V_ocv"] - df["voltage"]).abs()
    ).astype(float)
    df.loc[df["step_index"].isin(REST_STEPS), "Q_base"] = 0.0
    return df


def build_dataset(cell_id, surf_id, t_max_h=None, T_amb=None):
    elec      = load_electrical(cell_id)
    t_center  = load_temperature(cell_id, "T_center",  t_max_h)
    t_surface = load_temperature(surf_id,  "T_surface", t_max_h)

    if t_max_h:
        elec = elec[elec["t_h"] <= t_max_h].copy()

    # Merge center TC onto electrical timeline
    df = pd.merge_asof(
        elec.sort_values("t_h"),
        t_center.sort_values("t_h"),
        on="t_h", direction="nearest", tolerance=0.02
    )
    # Merge surface TC
    df = pd.merge_asof(
        df.sort_values("t_h"),
        t_surface.sort_values("t_h"),
        on="t_h", direction="nearest", tolerance=0.02
    )
    df = df.dropna(subset=["T_center", "T_surface"]).reset_index(drop=True)
    df["t_s"] = (df["test_time"] - df["test_time"].iloc[0]).astype(float)

    df = compute_vocv_end_of_rest(df)

    dt          = np.diff(df["t_s"].values, prepend=df["t_s"].values[0])
    dt[0]       = dt[1] if len(dt) > 1 else 1.0
    dt[dt<1e-6] = 1e-6
    df["dt"]    = dt

    print(f"  {cell_id} | {len(df)} rows | {df['t_h'].max():.1f}h | "
          f"T_center={df['T_center'].min():.1f}–{df['T_center'].max():.1f}°C | "
          f"T_surface={df['T_surface'].min():.1f}–{df['T_surface'].max():.1f}°C | "
          f"Q_base mean={df['Q_base'].mean():.2f}W max={df['Q_base'].max():.2f}W")
    return df


# =============================================================================
# MODEL
# dTc/dt = (Q_gen - Q_cond) / C_core
# dTs/dt = (Q_cond - Q_conv) / C_surface
#
# Q_gen  = gamma * Q_base  +  I * T * dUdT
# Q_cond = (Tc - Ts) / R_cs
# Q_conv = h * A * (Ts - T_amb)
# =============================================================================
def extract_arrays(df, T_amb_const):
    n = len(df)
    return {
        "Q_base":    df["Q_base"].values.astype(np.float64),
        "I":         df["current"].values.astype(np.float64),
        "T_amb":     np.full(n, T_amb_const, dtype=np.float64),
        "dt":        df["dt"].values.astype(np.float64),
        "Tc0":       float(df["T_center"].iloc[0]),
        "Ts0":       float(df["T_surface"].iloc[0]),   # measured initial surface T
        "Tc_meas":   df["T_center"].values.astype(np.float64),
        "Ts_meas":   df["T_surface"].values.astype(np.float64),
    }


def downsample(arr, step=10):
    return {k: v[::step] if isinstance(v, np.ndarray) else v
            for k, v in arr.items()}


def run_model(arrays, h_conv, R_cs, C_core, gamma):
    Q_base = arrays["Q_base"]
    I      = arrays["I"]
    T_amb  = arrays["T_amb"]
    dt     = arrays["dt"]
    n      = len(Q_base)

    Tc = np.empty(n); Tc[0] = arrays["Tc0"]
    Ts = np.empty(n); Ts[0] = arrays["Ts0"]

    inv_Rcs   = 1.0 / R_cs
    inv_Ccore = 1.0 / C_core
    inv_Csurf = 1.0 / C_surface
    hA        = h_conv * A_conv

    for i in range(1, n):
        q      = gamma * Q_base[i] + I[i] * (Tc[i-1] + 273.15) * dUdT
        Q_cond = (Tc[i-1] - Ts[i-1]) * inv_Rcs
        Q_conv = hA * (Ts[i-1] - T_amb[i])
        Tc[i]  = Tc[i-1] + dt[i] * (q      - Q_cond) * inv_Ccore
        Ts[i]  = Ts[i-1] + dt[i] * (Q_cond - Q_conv) * inv_Csurf

    return Tc, Ts


# =============================================================================
# CALIBRATION — 3 free parameters: R_cs, C_core, gamma
# Fits both T_center and T_surface simultaneously
# =============================================================================
def calibrate(datasets):
    data = {
        name: {
            "arr": downsample(extract_arrays(d["df"], d["cfg"]["T_amb"]), step=10),
            "cfg": d["cfg"],
        }
        for name, d in datasets.items()
    }

    def objective(x):
        R_cs, C_core, gamma = x
        if not (0.5  <= R_cs   <= 10.0): return 1e9
        if not (500  <= C_core <= 4200):  return 1e9
        if not (0.1  <= gamma  <= 3.0):   return 1e9
        total = 0.0
        for d in data.values():
            arr = d["arr"]
            Tc, Ts = run_model(arr, d["cfg"]["h_conv"], R_cs, C_core, gamma)
            if not (np.all(np.isfinite(Tc)) and np.all(np.isfinite(Ts))):
                return 1e9
            # Fit both center and surface equally
            total += np.mean((Tc - arr["Tc_meas"])**2)
            total += np.mean((Ts - arr["Ts_meas"])**2)
        return total

    # Coarse 3-D grid search
    print("  Coarse grid search (8×8×8 = 512 points)...")
    R_grid     = np.linspace(0.5,  8.0,  8)
    C_grid     = np.linspace(500,  4000, 8)
    gamma_grid = np.linspace(0.1,  3.0,  8)

    best_val, best_x = np.inf, [3.0, 2000.0, 1.0]
    for R in R_grid:
        for C in C_grid:
            for g in gamma_grid:
                v = objective([R, C, g])
                if v < best_val:
                    best_val, best_x = v, [R, C, g]

    print(f"  Grid best: R_cs={best_x[0]:.2f}  C_core={best_x[1]:.0f}  "
          f"gamma={best_x[2]:.3f}  MSE={best_val:.4f}")

    # Polish
    print("  Polishing with Nelder-Mead...")
    res = minimize(objective, best_x, method="Nelder-Mead",
                   options={"xatol": 0.01, "fatol": 0.01, "maxiter": 600})

    R_cs, C_core, gamma = res.x
    avg_rmse = np.sqrt(res.fun / (2 * len(datasets)))   # /2 because two nodes
    return (np.clip(R_cs,   0.5,  10.0),
            np.clip(C_core, 500., 4200.),
            np.clip(gamma,  0.1,  3.0),
            avg_rmse)


def rmse(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask])**2)))


# =============================================================================
# PLOT
# =============================================================================
def plot_all(results, R_cs, C_core, gamma):
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.suptitle(
        f"Lumped 2-Node Thermal Model vs. Measured\n"
        f"R_cs={R_cs:.2f} K/W  |  C_core={C_core:.0f} J/K  |  gamma={gamma:.3f}",
        fontsize=11, fontweight="bold"
    )
    order = ["20A_RT", "40A_RT", "20A_40C", "40A_40C"]

    for ax, name in zip(axes.flat, order):
        r         = results[name]
        df        = r["df"]
        t         = df["t_h"].values
        T_amb_val = TESTS[name]["T_amb"]
        w         = 300   # smoothing window for measured TC (noisy raw signal)

        # Smooth measured signals for display only
        Tc_sm = df["T_center"].rolling(w, center=True, min_periods=1).mean()
        Ts_sm = df["T_surface"].rolling(w, center=True, min_periods=1).mean()

        # Measured
        ax.plot(t, Tc_sm, "k-",  lw=1.5, label="Meas. T_center")
        ax.plot(t, Ts_sm, color="dimgray", lw=1.2, ls="--",
                label="Meas. T_surface")
        ax.axhline(T_amb_val, color="gray", lw=1.0, ls=":",
                   label=f"T_amb={T_amb_val:.0f}°C")

        # Model
        ax.plot(t, r["Tc"], color="royalblue", lw=1.5,
                label=f"Model T_center (RMSE={r['rmse_c']:.2f}°C)")
        ax.plot(t, r["Ts"], color="darkorange", lw=1.2, ls="--",
                label=f"Model T_surface (RMSE={r['rmse_s']:.2f}°C)")

        # Current twin axis
        ax2 = ax.twinx()
        ax2.fill_between(t, df["current"].values, 0, alpha=0.08, color="green")
        ax2.plot(t, df["current"].values, color="green", lw=0.5, alpha=0.3)
        ax2.axhline(0, color="green", lw=0.4, alpha=0.3)
        ax2.set_ylabel("Current (A)", color="green", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="green", labelsize=7)
        ax2.set_ylim(-df["current"].abs().max()*3, df["current"].abs().max()*3)

        ax.set_title(TESTS[name]["label"], fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (h)", fontsize=9)
        ax.set_ylabel("Temperature (°C)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5, loc="upper left", ncol=2)

        info = (
            f"h_conv = {r['h_conv']:.3f} W/m²·K\n"
            f"R0     = {r['R0']*1000:.3f} mΩ\n"
            f"R_cs   = {R_cs:.3f} K/W\n"
            f"C_core = {C_core:.0f} J/K\n"
            f"gamma  = {gamma:.3f}\n"
            f"T_amb  = {T_amb_val:.0f}°C\n"
            f"C_surf = {C_surface:.1f} J/K"
        )
        ax.text(0.98, 0.03, info, transform=ax.transAxes,
                fontsize=7.5, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="lightyellow", ec="goldenrod", alpha=0.9))

    plt.tight_layout()
    plt.savefig("thermal_model_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: thermal_model_results.png")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\nLoading datasets...")
    datasets = {}
    for name, cfg in TESTS.items():
        print(f"  {name}  (T_amb={cfg['T_amb']}°C)...")
        df = build_dataset(cfg["cell_id"], cfg["surf_id"],
                           cfg["t_max_h"], cfg["T_amb"])
        datasets[name] = {"df": df, "cfg": cfg}

    print("\nCalibrating R_cs, C_core, gamma ...")
    t0 = time.time()
    R_cs, C_core, gamma, avg_rmse = calibrate(datasets)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  → R_cs={R_cs:.3f} K/W | C_core={C_core:.1f} J/K | "
          f"gamma={gamma:.4f} | avg RMSE={avg_rmse:.3f}°C")

    print("\nRunning final model (full resolution)...")
    results = {}
    for name, d in datasets.items():
        arr    = extract_arrays(d["df"], d["cfg"]["T_amb"])
        Tc, Ts = run_model(arr, d["cfg"]["h_conv"], R_cs, C_core, gamma)
        rc = rmse(Tc, d["df"]["T_center"].values)
        rs = rmse(Ts, d["df"]["T_surface"].values)
        results[name] = {
            "df":     d["df"], "Tc": Tc, "Ts": Ts,
            "rmse_c": rc,      "rmse_s": rs,
            "h_conv": d["cfg"]["h_conv"], "R0": d["cfg"]["R0"],
        }
        print(f"  {name:<12}  RMSE center={rc:.3f}°C  surface={rs:.3f}°C")

    plot_all(results, R_cs, C_core, gamma)

    print(f"\n{'='*65}")
    print(f"{'Test':<12} {'h_conv':>8} {'R0(mΩ)':>8} "
          f"{'R_cs':>7} {'C_core':>7} {'gamma':>7} {'RMSE_c':>8} {'RMSE_s':>8}")
    print(f"{'─'*65}")
    for name in ["20A_RT","40A_RT","20A_40C","40A_40C"]:
        r = results[name]
        print(f"{name:<12} {r['h_conv']:>8.3f} {r['R0']*1000:>8.3f} "
              f"{R_cs:>7.3f} {C_core:>7.0f} {gamma:>7.4f} "
              f"{r['rmse_c']:>8.3f} {r['rmse_s']:>8.3f}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
