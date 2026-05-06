import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from scipy.optimize import minimize
import time

from r0_model import get_R0

engine = create_engine(
    "postgresql://postgres:YIB8dgucw20aPBd0BUuaYsdo5IXC3PzN@143.198.122.92:15432/cell_data"
)

r_in=0.005; r_jr=0.047; r_pvc_o=0.049; r_can_i=0.05024; r_can_o=0.05128
H_jr=0.188; H_can=0.220
k_jr_radial=0.16; k_pvc=0.19; k_abs=0.185; k_gap=0.6

R_jr  = np.log(r_jr/r_in)      / (2*np.pi*k_jr_radial*H_jr)
R_pvc = np.log(r_pvc_o/r_jr)   / (2*np.pi*k_pvc*H_jr)
R_gap = np.log(r_can_i/r_pvc_o)/ (2*np.pi*k_gap*H_jr)
R_abs = np.log(r_can_o/r_can_i)/ (2*np.pi*k_abs*H_can)
R1_nominal = R_jr + R_pvc/2
R2_nominal = R_pvc/2 + R_gap + R_abs

m_jr=3.0; Cp_jr=1400.0; rho_pvc=1380.0; Cp_pvc=900.0; rho_abs=1050.0; Cp_abs=1500.0
V_pvc_given=88.5e-6
V_abs = np.pi*(r_can_o**2 - r_can_i**2)*H_can
C_jr  = m_jr*Cp_jr
C_pvc = rho_pvc*V_pvc_given*Cp_pvc
C_abs = rho_abs*V_abs*Cp_abs
A_conv = 2*np.pi*r_can_o*H_can

print(f"R1_nominal={R1_nominal:.4f} K/W  R2_nominal={R2_nominal:.5f} K/W")
print(f"C_jr={C_jr:.0f}  C_pvc={C_pvc:.1f}  C_abs={C_abs:.1f} J/K\n")

TESTS = {
    "20A_RT":  {"cell_id":"12447-1","surf_id":"12448-1","h_conv":1.461,"T_amb":25.0,"label":"20 A - RT", "t_max_h":150.0},
    "40A_RT":  {"cell_id":"12449-1","surf_id":"12450-1","h_conv":1.461,"T_amb":25.0,"label":"40 A - RT", "t_max_h":150.0},
    "20A_40C": {"cell_id":"12451-1","surf_id":"12452-1","h_conv":2.438,"T_amb":40.0,"label":"20 A - 40C","t_max_h":150.0},
    "40A_40C": {"cell_id":"12453-1","surf_id":"12455-1","h_conv":2.438,"T_amb":40.0,"label":"40 A - 40C","t_max_h":150.0},
}

def load_electrical(cell_id):
    return pd.read_sql(f"""
        SELECT test_time, test_time/3600.0 AS t_h, cycle_index, step_index,
               current, voltage, charge_capacity, discharge_capacity
        FROM pec_timeseries WHERE cell_id='{cell_id}' ORDER BY test_time
    """, engine).dropna(subset=["current","voltage"]).reset_index(drop=True)

def load_temperature(cell_id, col_name, t_max_h=None):
    df = pd.read_sql(f"""
        SELECT test_time/3600.0 AS t_h, temperature
        FROM pec_temperature WHERE cell_id='{cell_id}' AND temperature > -50
        ORDER BY test_time
    """, engine).dropna().rename(columns={"temperature": col_name})
    if t_max_h:
        df = df[df["t_h"] <= t_max_h]
    return df.reset_index(drop=True)

def build_dataset(cell_id, surf_id, t_max_h=None, T_amb=None):
    elec     = load_electrical(cell_id)
    t_center = load_temperature(cell_id, "T_center",  t_max_h)
    t_surf   = load_temperature(surf_id,  "T_surface", t_max_h)
    if t_max_h:
        elec = elec[elec["t_h"] <= t_max_h].copy()
    df = pd.merge_asof(elec.sort_values("t_h"), t_center.sort_values("t_h"),
                       on="t_h", direction="nearest", tolerance=0.02)
    df = pd.merge_asof(df.sort_values("t_h"),  t_surf.sort_values("t_h"),
                       on="t_h", direction="nearest", tolerance=0.02)
    df = df.dropna(subset=["T_center","T_surface"]).reset_index(drop=True)
    df["t_s"] = (df["test_time"] - df["test_time"].iloc[0]).astype(float)
    dt = np.diff(df["t_s"].values, prepend=df["t_s"].values[0])
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    df["dt"] = np.clip(dt, 1e-6, 30.0)
    nom_Ah = max(df["discharge_capacity"].max(), 1.0)
    cum_Q  = np.cumsum(df["current"].values * df["dt"].values)
    df["SOC"] = np.clip(0.5 + cum_Q / (nom_Ah*3600.0), 0.0, 1.0)
    print(f"  {cell_id} | {len(df)} rows | {df['t_h'].max():.1f} h | "
          f"Tc {df['T_center'].min():.1f}-{df['T_center'].max():.1f} C | "
          f"Ts {df['T_surface'].min():.1f}-{df['T_surface'].max():.1f} C")
    return df

# 3-Node model ODEs:
#   C_jr  * dTjr/dt  = Q_gen           - (Tjr-Tpvc)/R1
#   C_pvc * dTpvc/dt = (Tjr-Tpvc)/R1  - (Tpvc-Tabs)/R2
#   C_abs * dTabs/dt = (Tpvc-Tabs)/R2 - h*A*(Tabs-T_amb)
# Q_gen = I^2*R0(SOC,Tjr) + I*(Tjr+273.15)*dUdT
# dUdT > 0 for Zn-MnO2: heats on charge (I>0), cools on discharge (I<0)
def run_model(arrays, h_conv, T_amb_const, R1, dUdT):
    I_arr=arrays["I"]; SOC=arrays["SOC"]; dt=arrays["dt"]; n=len(I_arr)
    Tjr=np.empty(n);  Tjr[0]=arrays["Tjr0"]
    Tpvc=np.empty(n); Tpvc[0]=arrays["Tpvc0"]
    Tabs=np.empty(n); Tabs[0]=arrays["Tabs0"]
    hA=h_conv*A_conv
    inv_R1=1.0/max(R1,1e-6); inv_R2=1.0/max(R2_nominal,1e-6)
    inv_Cjr=1.0/C_jr; inv_Cpvc=1.0/C_pvc; inv_Cabs=1.0/C_abs
    for i in range(1, n):
        Ii=I_arr[i]; R0i=get_R0(Tjr[i-1], SOC[i])
        Q_gen   = Ii**2*R0i + Ii*(Tjr[i-1]+273.15)*dUdT
        Q_cond1 = (Tjr[i-1]-Tpvc[i-1])*inv_R1
        Q_cond2 = (Tpvc[i-1]-Tabs[i-1])*inv_R2
        Q_conv  = hA*(Tabs[i-1]-T_amb_const)
        Tjr[i]  = Tjr[i-1]  + dt[i]*(Q_gen  -Q_cond1)*inv_Cjr
        Tpvc[i] = Tpvc[i-1] + dt[i]*(Q_cond1-Q_cond2)*inv_Cpvc
        Tabs[i] = Tabs[i-1] + dt[i]*(Q_cond2-Q_conv )*inv_Cabs
    return Tjr, Tpvc, Tabs

def extract_arrays(df, T_amb_const):
    return {
        "I":        df["current"].values.astype(np.float64),
        "SOC":      df["SOC"].values.astype(np.float64),
        "dt":       df["dt"].values.astype(np.float64),
        "Tjr0":     float(df["T_center"].iloc[0]),
        "Tpvc0":    float((df["T_center"].iloc[0]+df["T_surface"].iloc[0])/2),
        "Tabs0":    float(df["T_surface"].iloc[0]),
        "Tjr_meas": df["T_center"].values.astype(np.float64),
        "Tabs_meas":df["T_surface"].values.astype(np.float64),
    }

def downsample(arr, step=10):
    return {k: v[::step] if isinstance(v,np.ndarray) else v for k,v in arr.items()}

def calibrate(datasets):
    data = {name: {"arr": downsample(extract_arrays(d["df"],d["cfg"]["T_amb"]),step=10),
                   "cfg": d["cfg"]} for name,d in datasets.items()}
    def objective(x):
        R1,dUdT = x
        if not (R1_nominal*0.3 <= R1  <= R1_nominal*2.0): return 1e9
        if not (1e-5           <= dUdT <= 1e-3):           return 1e9
        total = 0.0
        for d in data.values():
            arr = d["arr"]
            Tjr,Tpvc,Tabs = run_model(arr,d["cfg"]["h_conv"],d["cfg"]["T_amb"],R1,dUdT)
            if not (np.all(np.isfinite(Tjr)) and np.all(np.isfinite(Tabs))): return 1e9
            total += np.mean((Tjr -arr["Tjr_meas"])**2)
            total += np.mean((Tabs-arr["Tabs_meas"])**2)
        return total
    print("  Grid search (12x12)...")
    R1_grid=np.linspace(R1_nominal*0.4,R1_nominal*1.8,12)
    dUdT_grid=np.linspace(5e-5,8e-4,12)
    best_val,best_x = np.inf,[R1_nominal,3e-4]
    for R1 in R1_grid:
        for d in dUdT_grid:
            v=objective([R1,d])
            if v<best_val: best_val,best_x=v,[R1,d]
    print(f"  Grid best: R1={best_x[0]:.3f} K/W  dU/dT={best_x[1]*1000:.4f} mV/K")
    res=minimize(objective,best_x,method="Nelder-Mead",
                 options={"xatol":1e-4,"fatol":0.005,"maxiter":600})
    R1_opt,dUdT_opt=res.x
    return (float(np.clip(R1_opt,R1_nominal*0.3,R1_nominal*2.0)),
            float(np.clip(dUdT_opt,1e-5,1e-3)),
            np.sqrt(res.fun/(2*len(datasets))))

def rmse(a,b):
    mask=np.isfinite(a)&np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask]-b[mask])**2)))

def plot_all(results, R1, dUdT):
    fig,axes=plt.subplots(2,2,figsize=(17,11))
    fig.suptitle(f"3-Node Model: Jellyroll|PVC|ABS\n"
                 f"R1={R1:.3f} K/W  R2={R2_nominal:.4f} K/W  dU/dT={dUdT*1000:.4f} mV/K",
                 fontsize=11,fontweight="bold")
    for ax,name in zip(axes.flat,["20A_RT","40A_RT","20A_40C","40A_40C"]):
        r=results[name]; df=r["df"]; t=df["t_h"].values; T_amb_val=TESTS[name]["T_amb"]
        Tjr_sm =df["T_center"].rolling(300,center=True,min_periods=1).mean()
        Tabs_sm=df["T_surface"].rolling(300,center=True,min_periods=1).mean()
        ax.plot(t,Tjr_sm,"k-",lw=1.5,label="Meas. T_center (JR)")
        ax.plot(t,Tabs_sm,color="dimgray",lw=1.2,ls="--",label="Meas. T_surface (ABS)")
        ax.axhline(T_amb_val,color="gray",lw=1.0,ls=":",label=f"T_amb={T_amb_val:.0f}C")
        ax.plot(t,r["Tjr"],color="royalblue",lw=1.5,label=f"Model T_jr (RMSE={r['rmse_jr']:.2f}C)")
        ax.plot(t,r["Tpvc"],color="mediumseagreen",lw=1.0,ls="-.",label="Model T_pvc")
        ax.plot(t,r["Tabs"],color="darkorange",lw=1.2,ls="--",label=f"Model T_abs (RMSE={r['rmse_abs']:.2f}C)")
        ax2=ax.twinx()
        ax2.fill_between(t,df["current"].values,0,alpha=0.07,color="green")
        ax2.plot(t,df["current"].values,color="green",lw=0.5,alpha=0.3)
        ax2.set_ylabel("Current (A)",color="green",fontsize=8)
        ax2.tick_params(axis="y",labelcolor="green",labelsize=7)
        ax2.set_ylim(-df["current"].abs().max()*3,df["current"].abs().max()*3)
        ax.set_title(TESTS[name]["label"],fontsize=10,fontweight="bold")
        ax.set_xlabel("Time (h)",fontsize=9); ax.set_ylabel("Temperature (C)",fontsize=9)
        ax.tick_params(labelsize=8); ax.legend(fontsize=7.0,loc="upper left",ncol=2)
        ax.text(0.98,0.03,
                f"h_conv={r['h_conv']:.3f}\nR1={R1:.3f} K/W\nR2={R2_nominal:.4f} K/W\n"
                f"dU/dT={dUdT*1000:.4f} mV/K\nC_jr={C_jr:.0f} J/K\n"
                f"C_pvc={C_pvc:.1f} J/K\nC_abs={C_abs:.1f} J/K\nT_amb={T_amb_val:.0f}C",
                transform=ax.transAxes,fontsize=7.5,ha="right",va="bottom",
                bbox=dict(boxstyle="round,pad=0.3",fc="lightyellow",ec="goldenrod",alpha=0.9))
    plt.tight_layout()
    plt.savefig("thermal_model_3node_results.png",dpi=150,bbox_inches="tight")
    plt.show()
    print("Saved: thermal_model_3node_results.png")

def main():
    print("\nLoading datasets...")
    datasets={}
    for name,cfg in TESTS.items():
        print(f"  {name} (T_amb={cfg['T_amb']}C)...")
        df=build_dataset(cfg["cell_id"],cfg["surf_id"],cfg["t_max_h"],cfg["T_amb"])
        datasets[name]={"df":df,"cfg":cfg}
    print(f"\nAnalytical R1={R1_nominal:.4f} K/W")
    print("Calibrating R1 and dU/dT...")
    t0=time.time()
    R1,dUdT,avg_rmse=calibrate(datasets)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  -> R1={R1:.4f} K/W (ratio to analytical: {R1/R1_nominal:.3f})")
    print(f"  -> dU/dT={dUdT*1000:.5f} mV/K")
    print(f"  -> avg RMSE={avg_rmse:.3f} C")
    print("\nFinal model run...")
    results={}
    for name,d in datasets.items():
        arr=extract_arrays(d["df"],d["cfg"]["T_amb"])
        Tjr,Tpvc,Tabs=run_model(arr,d["cfg"]["h_conv"],d["cfg"]["T_amb"],R1,dUdT)
        rjr=rmse(Tjr,d["df"]["T_center"].values)
        rabs=rmse(Tabs,d["df"]["T_surface"].values)
        results[name]={"df":d["df"],"Tjr":Tjr,"Tpvc":Tpvc,"Tabs":Tabs,
                       "rmse_jr":rjr,"rmse_abs":rabs,"h_conv":d["cfg"]["h_conv"]}
        print(f"  {name:<12} RMSE T_jr={rjr:.3f}C  T_abs={rabs:.3f}C")
    plot_all(results,R1,dUdT)

if __name__=="__main__":
    main()