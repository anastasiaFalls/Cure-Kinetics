from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

# =========================
# User settings
# =========================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "DSC results"
FIG_DIR = BASE_DIR / "figures"
OUT_DIR = BASE_DIR / "processed"

# Temperature range to keep
T_MIN, T_MAX = -50, 250

# Region where cure reaction is expected
ANALYSIS_TMIN = 40
ANALYSIS_TMAX = 250

# Save figure DPI
DPI = 350

# Baseline fit settings
BASELINE_BINS = 36
PEAK_PAD_C = 8.0              # extra padding on each side of detected peak window
PEAK_THRESHOLD_FRAC = 0.02    # fraction of peak height used to find rough peak edges

# Kinetics trimming settings
KIN_START_FRAC = 0.002
KIN_END_FRAC = 0.005

# Smoothing
SG_MAX_WINDOW = 31
SG_POLYORDER = 3


# =========================
# Helpers
# =========================

_float_token = re.compile(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")


def _to_float(tok: str):
    tok = tok.strip().replace(",", "")
    if not tok:
        return None
    if tok.startswith("-."):
        tok = "-0" + tok[1:]
    elif tok.startswith("."):
        tok = "0" + tok
    try:
        return float(tok)
    except Exception:
        return None


def _looks_numeric_row(line: str, min_cols: int = 3) -> bool:
    s = line.strip().replace(",", "")
    if not s:
        return False
    toks = s.split()
    if len(toks) < min_cols:
        return False
    ok = 0
    for t in toks:
        if _float_token.match(t):
            ok += 1
    return ok >= min_cols


def _find_first_numeric_line(lines):
    for i, line in enumerate(lines):
        if _looks_numeric_row(line, min_cols=3):
            return i
    return None


def remove_marker_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out[out["t_min"] >= 0].copy()

    if "purge_mL_min" in out.columns:
        out = out[out["purge_mL_min"].isna() | (out["purge_mL_min"] > 0)].copy()

    if {"HF_mW", "Cp_mJ_per_C", "purge_mL_min"}.issubset(out.columns):
        out = out[
            ~(
                (out["HF_mW"] == 0)
                & (out["Cp_mJ_per_C"] == 0)
                & (out["purge_mL_min"] == 0)
            )
        ].copy()

    out = out[out["T_C"] > 5].copy()

    return out.reset_index(drop=True)


def keep_heating_ramp(df: pd.DataFrame, allow_small_cooling: float = -0.5) -> pd.DataFrame:
    """
    Relaxed filter: allows small temperature reversals/noise without deleting real data.
    """
    out = df.sort_values("t_min").reset_index(drop=True).copy()
    dT = out["T_C"].diff()
    keep = dT.isna() | (dT > allow_small_cooling)
    return out[keep].reset_index(drop=True)


def sanity_clip_hf_spikes(df: pd.DataFrame, z_thresh: float = 8.0) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    dhf = out["HF_mW"].diff()

    med = np.nanmedian(dhf)
    mad = np.nanmedian(np.abs(dhf - med)) + 1e-12
    z = 0.6745 * (dhf - med) / mad

    keep = z.isna() | (np.abs(z) < z_thresh)
    return out[keep].reset_index(drop=True)


def cumulative_trapz(x, y):
    dx = np.diff(x)
    avg = 0.5 * (y[:-1] + y[1:])
    return np.concatenate([[0.0], np.cumsum(dx * avg)])


def choose_savgol_window(n: int, max_window: int = SG_MAX_WINDOW):
    """
    Return an odd Savitzky-Golay window length valid for array size n.
    """
    if n < 5:
        return None
    w = min(max_window, n)
    if w % 2 == 0:
        w -= 1
    if w < 5:
        return None
    return w


def smooth_signal(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    w = choose_savgol_window(len(y))
    if w is None:
        return y.copy()
    poly = min(SG_POLYORDER, w - 2)
    if poly < 1:
        return y.copy()
    return savgol_filter(y, window_length=w, polyorder=poly, mode="interp")


def infer_exotherm_sign(meta: dict, hf: np.ndarray) -> float:
    """
    Return sign multiplier so that the curing exotherm points upward (positive).
    """
    exo = (meta.get("exotherm_dir") or "").strip().upper()

    if "UP" in exo:
        return 1.0
    if "DOWN" in exo:
        return -1.0

    # Fallback: choose the sign that makes the dominant excursion positive
    p95 = np.nanpercentile(hf, 95)
    p5 = np.nanpercentile(hf, 5)
    return 1.0 if abs(p95) >= abs(p5) else -1.0


# =========================
# Parsing TA DSC text files
# =========================

def read_ta_dsc_txt(path: Path):
    raw_text = path.read_text(errors="ignore")
    raw_text = raw_text.replace("\ufeff", "").replace("\x00", "")
    lines = raw_text.splitlines()

    meta = {"sample_mass_mg": None, "heating_rate_C_min": None, "exotherm_dir": None}

    for line in lines[:1500]:
        s = line.strip()

        if s.lower().startswith("size"):
            m = re.search(r"size\s+([0-9.]+)\s*mg", s, re.IGNORECASE)
            if m:
                meta["sample_mass_mg"] = float(m.group(1))

        if s.lower().startswith("exotherm"):
            m = re.search(r"exotherm\s+(.+)", s, re.IGNORECASE)
            if m:
                meta["exotherm_dir"] = m.group(1).strip().upper()

        if "ramp" in s.lower() and "c/min" in s.lower():
            m = re.search(r"ramp\s+([0-9.]+)\s*°?c/min", s, re.IGNORECASE)
            if m:
                meta["heating_rate_C_min"] = float(m.group(1))

        if meta["heating_rate_C_min"] is None and "c/min" in s.lower():
            m = re.search(r"at\s+([0-9.]+)\s*°?c/min", s, re.IGNORECASE)
            if m:
                meta["heating_rate_C_min"] = float(m.group(1))

    start_idx = None
    for i, line in enumerate(lines):
        key = line.strip().replace(" ", "").replace(":", "").lower()
        if key == "startofdata":
            start_idx = i + 1
            break

    if start_idx is None:
        start_idx = _find_first_numeric_line(lines)

    if start_idx is None:
        raise ValueError(f"{path.name}: Could not locate numeric data start.")

    rows = []
    for line in lines[start_idx:]:
        s = line.strip()
        if not s:
            continue

        toks = s.split()
        nums = []
        for tok in toks:
            v = _to_float(tok)
            if v is not None:
                nums.append(v)

        if len(nums) >= 3:
            nums = nums[:5]
            while len(nums) < 5:
                nums.append(np.nan)
            rows.append(nums)

    if len(rows) < 20:
        raise ValueError(f"{path.name}: Not enough numeric rows parsed ({len(rows)}).")

    raw = pd.DataFrame(
        rows,
        columns=["t_min", "T_C", "HF_mW", "Cp_mJ_per_C", "purge_mL_min"],
    )
    raw = raw.dropna(subset=["t_min", "T_C", "HF_mW"]).sort_values("t_min").reset_index(
        drop=True
    )

    return raw, meta


# =========================
# Peak detection + baseline subtraction
# =========================

def detect_reaction_window(df: pd.DataFrame, meta: dict):
    """
    Detect the main cure reaction peak window automatically.
    Returns:
        peak_left, peak_right, sign, analysis_df
    """
    dfa = df[(df["T_C"] >= ANALYSIS_TMIN) & (df["T_C"] <= ANALYSIS_TMAX)].copy()
    dfa = dfa.sort_values("T_C").reset_index(drop=True)

    if len(dfa) < 30:
        raise ValueError("Too few points in analysis temperature range.")

    T = dfa["T_C"].to_numpy(dtype=float)
    HF_raw = dfa["HF_mW"].to_numpy(dtype=float)

    sign = infer_exotherm_sign(meta, HF_raw)
    HF_oriented = sign * HF_raw
    HF_smooth = smooth_signal(HF_oriented)

    # Simple edge-based baseline for rough peak detection
    n = len(T)
    edge_n = max(5, n // 12)

    left_T = np.mean(T[:edge_n])
    right_T = np.mean(T[-edge_n:])
    left_HF = np.median(HF_smooth[:edge_n])
    right_HF = np.median(HF_smooth[-edge_n:])

    if right_T == left_T:
        raise ValueError("Degenerate temperature axis during peak detection.")

    rough_baseline = left_HF + (right_HF - left_HF) * (T - left_T) / (right_T - left_T)
    resid = HF_smooth - rough_baseline

    peak_idx = int(np.argmax(resid))
    peak_val = resid[peak_idx]

    if not np.isfinite(peak_val) or peak_val <= 0:
        raise ValueError("Could not locate a positive cure peak in the analysis window.")

    thresh = PEAK_THRESHOLD_FRAC * peak_val

    left_idx = peak_idx
    while left_idx > 0 and resid[left_idx] > thresh:
        left_idx -= 1

    right_idx = peak_idx
    while right_idx < len(resid) - 1 and resid[right_idx] > thresh:
        right_idx += 1

    peak_left = max(ANALYSIS_TMIN, T[left_idx] - PEAK_PAD_C)
    peak_right = min(ANALYSIS_TMAX, T[right_idx] + PEAK_PAD_C)

    if peak_right <= peak_left:
        raise ValueError("Detected invalid peak window.")

    return peak_left, peak_right, sign, dfa


def pchip_baseline_subtract(df: pd.DataFrame, peak_left: float, peak_right: float, sign: float, bins: int = BASELINE_BINS):
    """
    Build a shape-preserving baseline from non-peak regions, then subtract it.
    The corrected reaction signal is stored as a positive quantity in HF_corr_mW.
    """
    df_out = df.copy().sort_values("T_C").reset_index(drop=True)

    T = df_out["T_C"].to_numpy(dtype=float)
    HF_raw = df_out["HF_mW"].to_numpy(dtype=float)

    # Orient so reaction is positive
    HF_oriented = sign * HF_raw

    baseline_mask = (T < peak_left) | (T > peak_right)
    T_base = T[baseline_mask]
    HF_base = HF_oriented[baseline_mask]

    if len(T_base) < 12:
        raise ValueError(
            "Not enough baseline points outside detected peak region."
        )

    edges = np.linspace(T_base.min(), T_base.max(), bins + 1)
    centers = []
    means = []

    for i in range(len(edges) - 1):
        m = (T_base >= edges[i]) & (T_base < edges[i + 1])
        if np.sum(m) >= 3:
            centers.append(np.mean(T_base[m]))
            means.append(np.mean(HF_base[m]))

    centers = np.asarray(centers, dtype=float)
    means = np.asarray(means, dtype=float)

    if len(centers) < 4:
        raise ValueError("Too few baseline anchor points after binning.")

    interp = PchipInterpolator(centers, means, extrapolate=True)
    baseline_oriented = interp(T)
    HF_corr = HF_oriented - baseline_oriented

    df_out["HF_oriented_mW"] = HF_oriented
    df_out["baseline_oriented_mW"] = baseline_oriented
    df_out["HF_corr_mW"] = HF_corr
    df_out["peak_left_C"] = peak_left
    df_out["peak_right_C"] = peak_right
    df_out["exo_sign"] = sign

    return df_out


# =========================
# Cure and rate calculations
# =========================

def compute_alpha_and_rate(df: pd.DataFrame, kinetics_tmin: float = 50.0):
    """
    Compute degree of cure alpha and curing rate d(alpha)/dt from corrected heat flow.
    Keeps only the main contiguous reaction region containing the global peak.
    """
    if "t_min" not in df.columns or df["t_min"].isna().any():
        raise ValueError("Need valid time data in 't_min'.")

    if "HF_corr_mW" not in df.columns:
        raise ValueError("Need baseline-corrected heat flow in 'HF_corr_mW'.")

    df = df.sort_values("t_min").reset_index(drop=True).copy()

    # Optional extra protection against low-temperature startup artifacts
    df = df[df["T_C"] >= kinetics_tmin].reset_index(drop=True)
    if len(df) < 20:
        raise ValueError("Too few points after kinetics temperature cutoff.")

    t_s = df["t_min"].to_numpy(dtype=float) * 60.0
    hf = df["HF_corr_mW"].to_numpy(dtype=float)

    # Clip tiny negatives only for integration logic
    hf_pos = np.maximum(hf, 0.0)

    # Smooth corrected heat flow for more stable region detection
    hf_smooth = smooth_signal(hf_pos)
    hf_smooth = np.maximum(hf_smooth, 0.0)

    peak_idx = int(np.argmax(hf_smooth))
    peak = hf_smooth[peak_idx]

    if peak <= 0:
        raise ValueError("HF_corr_mW never becomes positive; baseline likely wrong.")

    start_threshold = KIN_START_FRAC * peak
    end_threshold = KIN_END_FRAC * peak
    region_threshold = min(start_threshold, end_threshold)

    active = hf_smooth >= region_threshold

    if np.sum(active) < 10:
        raise ValueError("Reaction region too small after thresholding.")

    # Find the contiguous active region containing the main peak
    left = peak_idx
    while left > 0 and active[left - 1]:
        left -= 1

    right = peak_idx
    while right < len(active) - 1 and active[right + 1]:
        right += 1

    # Expand slightly outward until signal is near zero
    while left > 0 and hf_smooth[left] > 0.001 * peak:
        left -= 1

    while right < len(hf_smooth) - 1 and hf_smooth[right] > 0.001 * peak:
        right += 1

    df = df.iloc[left:right + 1].reset_index(drop=True)

    t_s = df["t_min"].to_numpy(dtype=float) * 60.0
    hf = df["HF_corr_mW"].to_numpy(dtype=float)
    hf_pos = np.maximum(hf, 0.0)

    hf_smooth = smooth_signal(hf_pos)
    hf_smooth = np.maximum(hf_smooth, 0.0)

    cum = cumulative_trapz(t_s, hf_smooth)
    total = cum[-1]

    if total <= 0:
        raise ValueError("Total integrated heat is zero after trimming.")

    alpha = cum / total
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = np.maximum.accumulate(alpha)

    rate = np.gradient(alpha, t_s)
    rate = smooth_signal(rate)
    rate = np.maximum(rate, 0.0)
    rate[0] = 0.0
    rate[-1] = 0.0

    df["HF_corr_smooth_mW"] = hf_smooth
    df["alpha"] = alpha
    df["rate"] = rate
    df["cum_heat_mJ"] = cum
    df["total_heat_mJ"] = total

    return df


# =========================
# Main
# =========================

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Couldn't find folder '{DATA_DIR}'. "
            f"Make sure this script is inside your 'Isoconversion task' folder."
        )

    FIG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    files = sorted(DATA_DIR.glob("Resin*.txt"))

    print(f"\nReading folder: {DATA_DIR.resolve()}")
    print(f"Found {len(files)} file(s).")
    if len(files) == 0:
        print("No files matched 'Resin*.txt'. Check filenames.")
        return

    runs_plot = []
    runs_kinetics = []

    summary_rows = []

    for f in files:
        try:
            print(f"\n--- Processing {f.name} ---")
            df, meta = read_ta_dsc_txt(f)

            df = remove_marker_rows(df)
            df = keep_heating_ramp(df, allow_small_cooling=-0.5)
            df = sanity_clip_hf_spikes(df, z_thresh=8.0)

            print(f"  Temp range after cleaning: {df['T_C'].min():.2f} to {df['T_C'].max():.2f} °C")
            print(f"  Points after cleaning: {len(df)}")

            df = df[(df["T_C"] >= T_MIN) & (df["T_C"] <= T_MAX)].reset_index(drop=True)
            if len(df) < 30:
                raise ValueError("Too few points after temperature filtering.")

            df_T = df.sort_values("T_C").reset_index(drop=True).copy()

            peak_left, peak_right, sign, _ = detect_reaction_window(df_T, meta)

            df_T = pchip_baseline_subtract(
                df_T,
                peak_left=peak_left,
                peak_right=peak_right,
                sign=sign,
                bins=BASELINE_BINS,
            )

            # Debug baseline plot
            plt.figure()
            plt.plot(df_T["T_C"], df_T["HF_oriented_mW"], label="Raw HF (oriented)")
            plt.plot(df_T["T_C"], df_T["baseline_oriented_mW"], label="Baseline")
            plt.plot(df_T["T_C"], df_T["HF_corr_mW"], label="Corrected HF")
            plt.axvspan(peak_left, peak_right, alpha=0.2, label="Detected peak region")
            plt.xlabel("Temperature (°C)")
            plt.ylabel("Heat Flow (mW)")
            plt.title(f"Baseline check: {f.name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"debug_baseline_{f.stem}.png", dpi=DPI)
            plt.close()

            # Map corrected values back to time-ordered data
            df_time = df.sort_values("t_min").reset_index(drop=True).copy()

            T_src = df_T["T_C"].to_numpy(dtype=float)
            df_time["HF_oriented_mW"] = np.interp(df_time["T_C"], T_src, df_T["HF_oriented_mW"])
            df_time["baseline_oriented_mW"] = np.interp(df_time["T_C"], T_src, df_T["baseline_oriented_mW"])
            df_time["HF_corr_mW"] = np.interp(df_time["T_C"], T_src, df_T["HF_corr_mW"])
            df_time["peak_left_C"] = peak_left
            df_time["peak_right_C"] = peak_right
            df_time["exo_sign"] = sign

            mass_mg = meta.get("sample_mass_mg", np.nan)
            beta = meta.get("heating_rate_C_min", np.nan)

            # Full corrected data for plots / traceability
            df_full = df_time.copy()

            if np.isfinite(mass_mg) and mass_mg > 0:
                mass_g = mass_mg / 1000.0
                df_full["HF_W_per_g"] = (df_full["HF_oriented_mW"] / 1000.0) / mass_g
                df_full["HFcorr_W_per_g"] = (df_full["HF_corr_mW"] / 1000.0) / mass_g
            else:
                df_full["HF_W_per_g"] = np.nan
                df_full["HFcorr_W_per_g"] = np.nan

            df_full["file"] = f.name
            df_full["beta_C_min"] = beta
            df_full["mass_mg"] = mass_mg

            full_csv = OUT_DIR / f"{f.stem}_full_corrected.csv"
            df_full.to_csv(full_csv, index=False)

            runs_plot.append(df_full)

            # Kinetics-trimmed data
            df_kin = compute_alpha_and_rate(df_time.copy())

            if np.isfinite(mass_mg) and mass_mg > 0:
                mass_g = mass_mg / 1000.0
                df_kin["HF_W_per_g"] = (df_kin["HF_oriented_mW"] / 1000.0) / mass_g
                df_kin["HFcorr_W_per_g"] = (df_kin["HF_corr_mW"] / 1000.0) / mass_g
                df_kin["HFcorr_smooth_W_per_g"] = (df_kin["HF_corr_smooth_mW"] / 1000.0) / mass_g
                df_kin["cum_heat_J_g"] = (df_kin["cum_heat_mJ"] / 1000.0) / mass_g
                df_kin["total_heat_J_g"] = (df_kin["total_heat_mJ"] / 1000.0) / mass_g
            else:
                df_kin["HF_W_per_g"] = np.nan
                df_kin["HFcorr_W_per_g"] = np.nan
                df_kin["HFcorr_smooth_W_per_g"] = np.nan
                df_kin["cum_heat_J_g"] = np.nan
                df_kin["total_heat_J_g"] = np.nan

            df_kin["file"] = f.name
            df_kin["beta_C_min"] = beta
            df_kin["mass_mg"] = mass_mg

            kin_csv = OUT_DIR / f"{f.stem}_kinetics.csv"
            df_kin.to_csv(kin_csv, index=False)

            print(f"  Detected peak window: {peak_left:.2f} to {peak_right:.2f} °C")
            print(f"  Temp range after trimming: {df_kin['T_C'].min():.2f} to {df_kin['T_C'].max():.2f} °C")
            print(f"  Points after trimming: {len(df_kin)}")
            print(
                f"  Processed: {f.name} | beta={beta} °C/min | mass={mass_mg} mg | "
                f"total heat={df_kin['total_heat_mJ'].iloc[0]:.3f} mJ"
            )

            summary_rows.append(
                {
                    "file": f.name,
                    "beta_C_min": beta,
                    "mass_mg": mass_mg,
                    "exotherm_dir": meta.get("exotherm_dir"),
                    "exo_sign_used": sign,
                    "peak_left_C": peak_left,
                    "peak_right_C": peak_right,
                    "points_full": len(df_full),
                    "points_kinetics": len(df_kin),
                    "total_heat_mJ": df_kin["total_heat_mJ"].iloc[0],
                    "total_heat_J_g": df_kin["total_heat_J_g"].iloc[0],
                }
            )

            runs_kinetics.append(df_kin)

        except Exception as e:
            print(f"  ERROR in {f.name}: {e}")

    if not runs_plot:
        print("\nNo valid runs available for plotting.")
        return

    big_plot = pd.concat(runs_plot, ignore_index=True)
    big_kin = pd.concat(runs_kinetics, ignore_index=True) if runs_kinetics else pd.DataFrame()

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(OUT_DIR / "run_summary.csv", index=False)

    # ---- Plot 1: Raw heat flow vs temperature ----
    plt.figure()
    for name, g in big_plot.groupby("file"):
        plt.plot(g["T_C"], g["HF_W_per_g"], label=name)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Heat Flow (W/g)")
    plt.title("Raw DSC: Heat Flow vs Temperature")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_raw_heatflow_vs_temperature.png", dpi=DPI)
    plt.show()

    # ---- Plot 2: Corrected bell curves ----
    plt.figure()
    for name, g in big_plot.groupby("file"):
        plt.plot(g["T_C"], g["HFcorr_W_per_g"], label=name)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Corrected Heat Flow (W/g)")
    plt.title("Corrected DSC: Reaction Bell Curves (Baseline Subtracted)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_corrected_bell_curves.png", dpi=DPI)
    plt.show()

    if not big_kin.empty:
        # ---- Plot 3: alpha vs temperature ----
        plt.figure()
        for beta, g in big_kin.groupby("beta_C_min"):
            label = f"{beta:g} °C/min" if np.isfinite(beta) else "unknown β"
            plt.plot(g["T_C"], g["alpha"], label=label)
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Degree of Cure α")
        plt.title("Degree of Cure vs Temperature")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "03_degree_of_cure_vs_temperature.png", dpi=DPI)
        plt.show()

        # ---- Plot 4: rate vs alpha ----
        plt.figure()
        for beta, g in big_kin.groupby("beta_C_min"):
            label = f"{beta:g} °C/min" if np.isfinite(beta) else "unknown β"

            g2 = g.sort_values("alpha").copy()
            g2 = g2.groupby(np.round(g2["alpha"], 4), as_index=False)["rate"].mean()
            g2.columns = ["alpha", "rate"]

            plt.plot(g2["alpha"], g2["rate"], label=label)

        plt.xlabel("Degree of Cure α")
        plt.ylabel("Curing Rate dα/dt (1/s)")
        plt.title("Curing Rate vs Degree of Cure")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / "04_rate_vs_degree_of_cure.png", dpi=DPI)
        plt.show()

    print(f"\nSaved figures to: {FIG_DIR.resolve()}")
    print(f"Saved processed CSVs to: {OUT_DIR.resolve()}")
    print("Saved per-run full corrected files as *_full_corrected.csv")
    print("Saved per-run kinetics files as *_kinetics.csv")
    print("Saved baseline debug plots as debug_baseline_*.png")


if __name__ == "__main__":
    main()

