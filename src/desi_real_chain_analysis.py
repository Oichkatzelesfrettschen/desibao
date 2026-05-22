#!/usr/bin/env python3
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import gaussian_kde

# Configuration
DESI_ROOT = "https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/cobaya/base_w_wa"
DATA_DIR = Path("artifacts/data/desi_y3_real_chains")
DATA_DIR.mkdir(parents=True, exist_ok=True)

CASES = {
    "BAO+CMBcomp": f"{DESI_ROOT}/desi-bao-all_CMB-compressed-theta-ombh2-ombch2",
    "BAO+CMBcomp+DESY5SN": f"{DESI_ROOT}/desi-bao-all_CMB-compressed-theta-ombh2-ombch2_desy5sn",
}

def dl(url: str, out: Path) -> None:
    if out.exists() and out.stat().st_size > 0:
        return
    print(f"[download] {url}")
    try:
        urllib.request.urlretrieve(url, out)
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def read_chain(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    with path.open("r") as f:
        header = None
        for line in f:
            if line.startswith("#"):
                header = line[1:].strip()
                break
    if header is None: return pd.DataFrame()
    cols = header.split()
    # Handle potentially malformed lines or comments
    try:
        return pd.read_csv(path, delim_whitespace=True, comment="#", names=cols)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return pd.DataFrame()

def wedge_mask(df: pd.DataFrame) -> np.ndarray:
    if df.empty: return np.array([])
    w0 = df["w"].to_numpy()
    wa = df["wa"].to_numpy()
    return (w0 >= -1.0) & ((w0 + wa) >= -1.0)

def bestfit(df: pd.DataFrame, mask: np.ndarray | None = None):
    if df.empty: return 0.0, 0.0, 0.0
    sub = df if mask is None else df.loc[mask]
    if sub.empty: return float('inf'), 0.0, 0.0
    i = sub["chi2"].idxmin()
    r = sub.loc[i]
    return float(r["chi2"]), float(r["w"]), float(r["wa"])

def prior_bounds_from_yaml(yaml_path: Path):
    if not yaml_path.exists():
        return (-3.0, 1.0), (-3.0, 2.0) # Default DESI priors
    try:
        y = yaml.safe_load(yaml_path.read_text())
        p = y["params"]
        w0min, w0max = p["w"]["prior"]["min"], p["w"]["prior"]["max"]
        wamin, wamax = p["wa"]["prior"]["min"], p["wa"]["prior"]["max"]
        return (float(w0min), float(w0max)), (float(wamin), float(wamax))
    except:
        return (-3.0, 1.0), (-3.0, 2.0)

def sddr_bayes_factor(df: pd.DataFrame, w0_bounds, wa_bounds, point=(-1.0, 0.0)) -> float:
    if df.empty: return 0.0, 0.0, 0.0
    x = np.vstack([df["w"].to_numpy(), df["wa"].to_numpy()])
    wt = df["weight"].to_numpy().astype(float)
    try:
        kde = gaussian_kde(x, weights=wt)
        post = float(kde(point))
    except:
        post = 0.0
    
    prior = 1.0 / ((w0_bounds[1]-w0_bounds[0]) * (wa_bounds[1]-wa_bounds[0]))
    return post / prior if prior > 0 else 0, post, prior

def wedge_prior_fraction(w0_bounds, wa_bounds) -> float:
    w0min, w0max = w0_bounds
    wamin, wamax = wa_bounds
    a = max(-1.0, w0min)
    b = min(w0max, w0max) # Typo in original logic, likely meant min(w0max, ...) wait, w0max is w0max. 
    # Logic: w0 in [w0min, w0max] AND w0 >= -1. So range is [max(-1, w0min), w0max].
    b = w0max
    if b <= a: return 0.0

    grid = np.linspace(a, b, 20001)
    # w0 + wa >= -1  =>  wa >= -1 - w0
    low = np.maximum(wamin, -1.0 - grid)
    # wa also <= wamax
    width = np.maximum(0.0, wamax - low)
    area_wedge = np.trapezoid(width, grid)
    area_box = (w0max - w0min) * (wamax - wamin)
    return float(area_wedge / area_box)

def main():
    print("--- 1. DESI Y3 REAL CHAIN ANALYSIS ---")
    results = []

    for name, base in CASES.items():
        print(f"\nAnalyzing {name}...")
        case_dir = DATA_DIR / name
        case_dir.mkdir(exist_ok=True)
        
        # Download Inputs
        yml = case_dir / "chain.input.yaml"
        dl(f"{base}/chain.input.yaml", yml)
        w0b, wab = prior_bounds_from_yaml(yml)
        
        # Download Chains
        dfs = []
        for i in range(1, 5):
            fp = case_dir / f"chain.{i}.txt"
            dl(f"{base}/chain.{i}.txt", fp)
            dfs.append(read_chain(fp))
        
        df = pd.concat(dfs, ignore_index=True)
        if df.empty:
            print("No data found.")
            continue

        wt = df["weight"].to_numpy().astype(float)
        m = wedge_mask(df)
        post_wedge = wt[m].sum() / wt.sum()

        chi2_all, w0_all, wa_all = bestfit(df, None)
        chi2_w, w0_w, wa_w = bestfit(df, m)
        dchi2 = chi2_w - chi2_all

        prior_wedge = wedge_prior_fraction(w0b, wab)
        z_ratio = post_wedge / prior_wedge if prior_wedge > 0 else float("nan")

        B, postdens, priordens = sddr_bayes_factor(df, w0b, wab, (-1.0, 0.0))

        print(f"  Priors: w0 {w0b}, wa {wab}")
        print(f"  Posterior Wedge Mass P(W|D): {post_wedge:.6e}")
        print(f"  Prior Wedge Mass P(W):       {prior_wedge:.6e}")
        print(f"  Evidence Ratio Z_W/Z_box:    {z_ratio:.6e} (ln = {math.log(z_ratio) if z_ratio>0 else -99:.3f})")
        print(f"  Global Best Fit: chi2={chi2_all:.2f} (w0={w0_all:.3f}, wa={wa_all:.3f})")
        print(f"  Wedge Best Fit:  chi2={chi2_w:.2f} (w0={w0_w:.3f}, wa={wa_w:.3f})")
        print(f"  dChi2 (Wedge penalty): {dchi2:.2f}")
        print(f"  Bayes Factor B(LCDM:w0wa): {B:.4g}")
        
        results.append({
            "dataset": name,
            "post_wedge": post_wedge,
            "dchi2": dchi2,
            "bayes_lcdm": B
        })

    return results

if __name__ == "__main__":
    main()
