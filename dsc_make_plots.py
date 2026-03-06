from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


# =========================
# User settings
# =========================

DATA_DIR = Path("DSC results")
FIG_DIR = Path("figures")
OUT_DIR = Path("processed")

# Temperature range to keep
T_MIN, T_MAX = -50, 250

# Save figure DPI
DPI = 350

ANALYSIS_TMIN = 40
ANALYSIS_TMAX = 250

# Peak exclusion window for baseline fit
PEAK_EXCLUDE_LEFT = 70
PEAK_EXCLUDE_RIGHT = 170

# Number of baseline bins
BASELINE_BINS = 30


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
        out = out[~((out["HF_mW"] == 0) & (out["Cp_mJ_per_C"] == 0) & (out["purge_mL_min"] == 0))].copy()

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
            m = re.search(r"exotherm\s+(\w+)", s, re.IGNORECASE)
            if m:
                meta["exotherm_dir"] = m.group(1).upper()

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

    raw = pd.DataFrame(rows, columns=["t_min", "T_C", "HF_mW", "Cp_mJ_per_C", "purge_mL_min"])
    raw = raw.dropna(subset=["t_min", "T_C", "HF_mW"]).sort_values("t_min").reset_index(drop=True)

    return raw, meta


# =========================
# DSC processing
# =========================

def pchip_baseline_subtract(df, peak_left, peak_right, bins=30):
    """
    Shape-preserving baseline subtraction:
    - exclude the reaction peak region
    - average the remaining baseline regions in temperature bins
    - fit a PCHIP interpolator through those baseline anchor points
    - subtract baseline from raw heat flow
    """
    df_out = df.copy().sort_values("T_C").reset_index(drop=True)

    T = df_out["T_C"].to_numpy()
    HF = df_out["HF_mW"].to_numpy()

    # Keep only non-peak regions for baseline fitting
    baseline_mask = (T < peak_left) | (T > peak_right)

    T_base = T[baseline_mask]
    HF_base = HF[baseline_mask]

    if len(T_base) < 10:
        raise ValueError(
            "Not enough baseline points outside excluded peak region. "
            "Adjust PEAK_EXCLUDE_LEFT / PEAK_EXCLUDE_RIGHT."
        )

    # Bin-average the baseline points for stability
    edges = np.linspace(T_base.min(), T_base.max(), bins + 1)
    centers = []
    means = []

    for i in range(len(edges) - 1):
        m = (T_base >= edges[i]) & (T_base < edges[i + 1])
        if np.sum(m) >= 3:
            centers.append(np.mean(T_base[m]))
            means.append(np.mean(HF_base[m]))

    centers = np.array(centers)
    means = np.array(means)

    if len(centers) < 4:
        raise ValueError("Too few baseline anchor points after binning.")

    # Shape-preserving interpolator
    interp = PchipInterpolator(centers, means, extrapolate=True)

    baseline = interp(T)
    HF_corr = HF - baseline

    df_out["baseline"] = baseline
    df_out["HF_corr"] = HF_corr

    return df_out


def cumulative_trapz(x, y):
    dx = np.diff(x)
    avg = 0.5 * (y[:-1] + y[1:])
    return np.concatenate([[0.0], np.cumsum(dx * avg)])

def compute_alpha_and_rate(df, meta):
    """
    Compute alpha and curing rate in TIME-space.
    """
    if "t_min" not in df.columns or df["t_min"].isna().any():
        raise ValueError("Need valid time data in 't_min'.")

    df = df.sort_values("t_min").reset_index(drop=True)

    t_s = df["t_min"].to_numpy(dtype=float) * 60.0
    hf = df["HF_corr"].to_numpy(dtype=float)

    # Remove negative baseline noise
    hf = np.maximum(hf, 0.0)

    peak = np.max(hf)
    if peak <= 0:
        raise ValueError("HF_corr never becomes positive; baseline likely wrong.")

    # Asymmetric trimming
    start_threshold = 0.002 * peak
    end_threshold = 0.005 * peak
    start_mask = hf >= start_threshold
    end_mask = hf >= end_threshold

    if np.sum(start_mask) < 10 or np.sum(end_mask) < 10:
        raise ValueError("Reaction region too small after thresholding.")

    first = np.argmax(start_mask)
    last = len(end_mask) - np.argmax(end_mask[::-1])

    df = df.iloc[first:last].reset_index(drop=True)
    t_s = df["t_min"].to_numpy(dtype=float) * 60.0
    hf = np.maximum(df["HF_corr"].to_numpy(dtype=float), 0.0)

    # Smooth a bit
    window = min(21, len(hf) if len(hf) % 2 == 1 else len(hf) - 1)
    if window >= 5:
        kernel = np.ones(window) / window
        hf = np.convolve(hf, kernel, mode="same")

    cum = cumulative_trapz(t_s, hf)
    total = cum[-1]

    if total <= 0:
        raise ValueError("Total integrated heat is zero after trimming.")

    alpha = cum / total
    alpha = np.maximum.accumulate(alpha)

    rate = np.gradient(alpha, t_s)
    rate = np.maximum(rate, 0.0)
    rate[0] = 0.0
    rate[-1] = 0.0

    df["alpha"] = alpha
    df["rate"] = rate

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

    runs = []

    for f in files:
        try:
            print(f"\n--- Processing {f.name} ---")
            df, meta = read_ta_dsc_txt(f)

            df = remove_marker_rows(df)
            df = keep_heating_ramp(df, allow_small_cooling=-0.5)
            df = sanity_clip_hf_spikes(df, z_thresh=8.0)

            print(f"  Temp range after cleaning: {df['T_C'].min():.2f} to {df['T_C'].max():.2f} °C")
            print(f"  Points after cleaning: {len(df)}")

            df = df[(df["T_C"] >= ANALYSIS_TMIN) & (df["T_C"] <= ANALYSIS_TMAX)].reset_index(drop=True)

            df_T = df.sort_values("T_C").reset_index(drop=True)

            df_T = pchip_baseline_subtract(
                df_T,
                peak_left=PEAK_EXCLUDE_LEFT,
                peak_right=PEAK_EXCLUDE_RIGHT,
                bins=BASELINE_BINS
                )

            # DEBUG spline plot
            plt.figure()
            plt.plot(df_T["T_C"], df_T["HF_mW"], label="Raw HF")
            plt.plot(df_T["T_C"], df_T["baseline"], label="Spline baseline")
            plt.axvspan(
                PEAK_EXCLUDE_LEFT,
                PEAK_EXCLUDE_RIGHT,
                alpha=0.2,
                label="Excluded peak region"
            )
            plt.xlabel("Temperature (°C)")
            plt.ylabel("Heat Flow (mW)")
            plt.title(f"Spline baseline check: {f.name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"debug_spline_{f.stem}.png", dpi=DPI)
            plt.close()

            df_time = df.sort_values("t_min").reset_index(drop=True)
            df_time["HF_corr"] = np.interp(
                df_time["T_C"].to_numpy(),
                df_T["T_C"].to_numpy(),
                df_T["HF_corr"].to_numpy()
            )
            df_time["baseline"] = df_time["HF_mW"] - df_time["HF_corr"]

            df_time = compute_alpha_and_rate(df_time, meta)

            print(f"  Temp range after trimming: {df_time['T_C'].min():.2f} to {df_time['T_C'].max():.2f} °C")
            print(f"  Points after trimming: {len(df_time)}")

            if meta.get("sample_mass_mg") is not None and meta["sample_mass_mg"] > 0:
                mass_g = meta["sample_mass_mg"] / 1000.0
                df_time["HF_W_per_g"] = (df_time["HF_mW"] / 1000.0) / mass_g
                df_time["HFcorr_W_per_g"] = (df_time["HF_corr"] / 1000.0) / mass_g

            df_time["file"] = f.name
            df_time["beta_C_min"] = meta.get("heating_rate_C_min", np.nan)
            df_time["mass_mg"] = meta.get("sample_mass_mg", np.nan)

            out_csv = OUT_DIR / f"{f.stem}_processed.csv"
            df_time.to_csv(out_csv, index=False)

            print(f"  Processed: {f.name} | beta={df_time['beta_C_min'].iloc[0]} °C/min | mass={df_time['mass_mg'].iloc[0]} mg")
            runs.append(df_time)

        except Exception as e:
            print(f"  ERROR in {f.name}: {e}")

    if len(runs) == 0:
        print("\nNo datasets survived processing. The spline/trimming is still too aggressive.")
        return

    big = pd.concat(runs, ignore_index=True)

    # ---- Plot 1 ----
    plt.figure()
    for name, g in big.groupby("file"):
        plt.plot(g["T_C"], g["HF_W_per_g"], label=name)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Heat Flow (J/s/g)")
    plt.title("Raw DSC: Heat Flow vs Temperature")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_raw_heatflow_vs_temperature.png", dpi=DPI)
    plt.show()


# ---- Plot 2 ----
    plt.figure()
    for name, g in big.groupby("file"):
        plt.plot(g["T_C"], g["HFcorr_W_per_g"], label=name)
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Corrected Heat Flow (J/s/g)")
    plt.title("Corrected DSC: Reaction Bell Curves (Baseline Subtracted)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_corrected_bell_curves.png", dpi=DPI)
    plt.show()


# ---- Plot 3 ----
    plt.figure()
    for beta, g in big.groupby("beta_C_min"):
        label = f"{beta:g} °C/min" if np.isfinite(beta) else "unknown β"
        plt.plot(g["T_C"], g["alpha"], label=label)

    plt.xlabel("Temperature (°C)")
    plt.ylabel("Degree of Cure α")
    plt.title("Degree of Cure vs Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_degree_of_cure_vs_temperature.png", dpi=DPI)
    plt.show()


# ---- Plot 4 ----
    plt.figure()
    for beta, g in big.groupby("beta_C_min"):
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
    print("\nDebug spline plots saved as debug_spline_*.png")


if __name__ == "__main__":
    main()
