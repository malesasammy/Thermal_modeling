import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy.interpolate import interp1d
import os
import pickle

# =============================================================================
# DATABASE
# =============================================================================
engine = create_engine(
    "postgresql://postgres:YIB8dgucw20aPBd0BUuaYsdo5IXC3PzN@143.198.122.92:15432/cell_data"
)

CACHE_FILE = "r0_lookup.pkl"

# =============================================================================
# GITT TEST CONFIGURATION
# =============================================================================
GITT_CELLS = {
    "RT": {
        "elec_id": "12426-1",
        "surf_id": "12426-1",
        "center_id": "12427-1",
        "label": "RT",
        "T_nominal": 25.0,
    },

    "40C": {
        "elec_id": "12426-1",
        "surf_id": "12424-1",
        "center_id": "12425-1",
        "label": "40C",
        "T_nominal": 40.0,
    },
}

# Anchor resistances from GITT
R0_GITT_RT = 0.00220     # 2.20 mΩ
R0_GITT_40C = 0.00205    # 2.05 mΩ

R0_MIN = 0.0020          # 2 mΩ minimum


# =============================================================================
# LOAD FULL DATASET
# =============================================================================
def load_full(elec_id, temp_surface_id, temp_center_id):

    df = pd.read_sql(f"""
        SELECT
            test_time,
            test_time/3600.0 AS t_h,
            current,
            voltage,
            charge_capacity,
            discharge_capacity
        FROM pec_timeseries
        WHERE cell_id = '{elec_id}'
        ORDER BY test_time
    """, engine)

    df = df.dropna(subset=["current", "voltage"]).reset_index(drop=True)

    surf = pd.read_sql(f"""
        SELECT
            test_time/3600.0 AS t_h,
            temperature
        FROM pec_temperature
        WHERE cell_id = '{temp_surface_id}'
          AND temperature > -50
        ORDER BY test_time
    """, engine)

    surf = surf.rename(columns={"temperature": "T_surface"})

    center = pd.read_sql(f"""
        SELECT
            test_time/3600.0 AS t_h,
            temperature
        FROM pec_temperature
        WHERE cell_id = '{temp_center_id}'
          AND temperature > -50
        ORDER BY test_time
    """, engine)

    center = center.rename(columns={"temperature": "T_center"})

    df = pd.merge_asof(
        df.sort_values("t_h"),
        surf.sort_values("t_h"),
        on="t_h",
        direction="nearest"
    )

    df = pd.merge_asof(
        df.sort_values("t_h"),
        center.sort_values("t_h"),
        on="t_h",
        direction="nearest"
    )

    return df.reset_index(drop=True)


# =============================================================================
# EXTRACT R0 FROM CURRENT PULSES
# =============================================================================
def extract_gitt_R0(df, nominal_Ah, I_threshold=5.0):

    I = df["current"].values
    V = df["voltage"].values
    Tc = df["T_center"].values
    Dch = df["discharge_capacity"].values
    Chg = df["charge_capacity"].values

    records = []

    for i in range(2, len(I) - 10):

        if abs(I[i - 1]) < 0.5 and abs(I[i]) > I_threshold:

            V_before = np.mean(V[max(0, i - 5):i])
            V_after = np.mean(V[i + 2:i + 7])
            I_pulse = np.mean(I[i + 2:i + 7])

            if abs(I_pulse) < I_threshold:
                continue

            R0 = abs(V_before - V_after) / abs(I_pulse)

            if I_pulse < 0:
                soc = 1.0 - Dch[i] / nominal_Ah
            else:
                soc = Chg[i] / nominal_Ah

            records.append({
                "SOC": np.clip(soc, 0.0, 1.0),
                "R0": R0,
                "T_center": Tc[i]
            })

    return pd.DataFrame(records)


# =============================================================================
# CLEAN + BIN
# =============================================================================
def clean_and_average(df):

    df = df.copy()

    df["SOC_bin"] = (df["SOC"] / 0.1).round() * 0.1
    df["SOC_bin"] = df["SOC_bin"].round(2)

    result = df.groupby("SOC_bin").agg(
        R0=("R0", "median"),
        T_center=("T_center", "mean")
    ).reset_index()

    result = result.rename(columns={"SOC_bin": "SOC"})
    return result.sort_values("SOC").reset_index(drop=True)


# =============================================================================
# BUILD MODEL
# =============================================================================
def build_model():

    print("Building R0 lookup table...")

    tables = {}

    for label, cfg in GITT_CELLS.items():

        print(f"Loading {label}...")

        df = load_full(
            cfg["elec_id"],
            cfg["surf_id"],
            cfg["center_id"]
        )

        nominal_Ah = df["discharge_capacity"].max()

        raw = extract_gitt_R0(df, nominal_Ah)
        clean = clean_and_average(raw)

        tables[label] = {
            "table": clean,
            "T_mean": df["T_center"].mean()
        }

        print(clean)

    SOC_RT = tables["RT"]["table"]["SOC"].values
    R0_RT = tables["RT"]["table"]["R0"].values

    SOC_40 = tables["40C"]["table"]["SOC"].values
    R0_40 = tables["40C"]["table"]["R0"].values

    T_RT = tables["RT"]["T_mean"]
    T_40 = tables["40C"]["T_mean"]

    shape_RT = R0_RT / np.mean(R0_RT)
    shape_40 = R0_40 / np.mean(R0_40)

    model = {
        "interp_shape_RT": interp1d(
            SOC_RT,
            shape_RT,
            kind="linear",
            fill_value="extrapolate"
        ),

        "interp_shape_40": interp1d(
            SOC_40,
            shape_40,
            kind="linear",
            fill_value="extrapolate"
        ),

        "scalar_interp": interp1d(
            [T_RT, T_40],
            [R0_GITT_RT, R0_GITT_40C],
            kind="linear",
            fill_value="extrapolate"
        ),

        "T_RT": T_RT,
        "T_40": T_40
    }

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved lookup cache: {CACHE_FILE}")
    return model


# =============================================================================
# LOAD CACHE SAFELY
# =============================================================================
def load_model():

    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                model = pickle.load(f)

            required_keys = [
                "interp_shape_RT",
                "interp_shape_40",
                "scalar_interp",
                "T_RT",
                "T_40"
            ]

            if all(k in model for k in required_keys):
                print("Loaded cached R0 lookup.")
                return model
            else:
                print("Old cache detected. Rebuilding...")

        except Exception:
            print("Cache corrupted. Rebuilding...")

    return build_model()


MODEL = load_model()


# =============================================================================
# LOOKUP FUNCTION
# =============================================================================
def get_R0(T_celsius, SOC):

    T = np.asarray(T_celsius, dtype=float)
    SOC = np.clip(np.asarray(SOC, dtype=float), 0.0, 1.0)

    alpha = np.clip(
        (T - MODEL["T_RT"]) /
        (MODEL["T_40"] - MODEL["T_RT"] + 1e-9),
        0.0,
        1.0
    )

    shape = (
        (1 - alpha) * MODEL["interp_shape_RT"](SOC)
        + alpha * MODEL["interp_shape_40"](SOC)
    )

    scalar = MODEL["scalar_interp"](T)

    R0 = shape * scalar

    # Enforce physical lower bound
    R0 = np.maximum(R0, R0_MIN)

    return float(R0) if np.size(R0) == 1 else R0


# =============================================================================
# TEST MODE ONLY
# =============================================================================
if __name__ == "__main__":

    print("\nExample lookups:")
    print(f"R0(25C,0.5) = {get_R0(25,0.5)*1000:.3f} mΩ")
    print(f"R0(40C,0.5) = {get_R0(40,0.5)*1000:.3f} mΩ")
    print(f"R0(30C,0.2) = {get_R0(30,0.2)*1000:.3f} mΩ")