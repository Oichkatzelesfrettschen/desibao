# desibao: Late-2025 Cosmology & Dark Sector Analysis

A professional software repository for the analysis of DESI Y3 BAO data, Supernova systematics, and Dark Matter phenomenology as of December 2025.

## Project Structure

```
desibao/
├── artifacts/              # Versioned data and summary reports
│   ├── data/               # Fetched raw data products (DESI Y3, etc.)
│   │   ├── desi_y3_bao_chains/
│   │   └── desi_y3_bao_bestfit/
│   └── reports/            # Summary of findings and monographs
├── docs/                   # Visualizations and expanded documentation
├── src/                    # Source code for reproduction and analysis
│   └── desi_y3_comprehensive_analysis.py
├── README.md               # Project overview
└── LICENSE                 # Project license (MIT)
```

## Methodology

This repository implements a rigorous, first-principles analysis of the "November 2025" cosmology breakthroughs. Key methodologies include:
- **Bayesian Wedge Analysis:** Quantifying the posterior mass in the canonical quintessence region vs. the phantom regime.
- **DAO Stress Testing:** Forward-modeling Dark Acoustic Oscillations in Fourier space to quantify BAO scale bias.
- **Systematic Corrections:** Modeling the impact of SN Ia progenitor age bias on inferred expansion history.
- **Multi-Messenger DM Scan:** Cross-correlating indirect detection signals with cosmic-ray (AMS-02) constraints.

## Usage

The primary analysis pipeline can be executed via:

```bash
python3 src/desi_y3_comprehensive_analysis.py
```

## Data Sources

Data artifacts are sourced from the [DESI LBL Data Portal](https://data.desi.lbl.gov/public/papers/y3/bao-cosmo-params/).

---
*Created for Research/Physics context.*
