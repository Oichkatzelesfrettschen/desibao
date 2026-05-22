#!/usr/bin/env python3
"""
dao_exclusion_plot.py

Generates a falsifiable numerical exclusion plot for Dark Acoustic Oscillations (DAO)
based on Fourier-space scaling arguments anchored to DESI Y1/Y3 Full-Shape sensitivity.

Methodology:
1. Scaling Law: Delta_chi2(A, V) = Delta_chi2_ref * (A / A_ref)^2 * (V / V_ref)
2. Bias Requirement: Amplitude A_bias needed to shift alpha by 0.5%.
   From previous toy: A ~ 2% -> alpha bias ~ 0.5%.
   (Linear scaling approximation: bias propto A)
   So A_bias(0.5%) approx 0.02.
3. Detectability:
   V_ref = 10 (Gpc/h)^3
   A_ref = 0.02
   Delta_chi2_ref = 3.27
   
   We plot:
   - "Bias Needed" band (A ~ 1-3%)
   - "Excluded (2 sigma)" region (Delta_chi2 > 4)
   
   A_max(V) = A_ref * sqrt( (4 / Delta_chi2_ref) * (V_ref / V) )
"""

import matplotlib.pyplot as plt
import numpy as np

# Constants from previous analysis
A_REF = 0.02         # 2% amplitude
V_REF = 10.0         # Gpc^3/h^3 (Toy volume)
DCHI2_REF = 3.27     # Detection significance (squared) at (A_REF, V_REF)

# Bias Target
# We found 2% -> ~0.57% bias.
# Target bias range: 0.3% - 0.7% (DESI precision is ~0.7-1.0% in Y1, getting better)
# Let's say "Significant Bias" is > 0.3%.
# A_bias_min = 0.02 * (0.3 / 0.57) approx 0.01
# A_bias_max = 0.02 * (0.7 / 0.57) approx 0.025
A_BIAS_BAND = (0.01, 0.025)

def get_exclusion_curve(v_eff_array):
    # Solve for A such that Delta_chi2 = 4 (2 sigma)
    # 4 = DCHI2_REF * (A / A_REF)^2 * (v / V_REF)
    # A^2 = 4 * A_REF^2 * V_REF / (DCHI2_REF * v)
    # A = A_REF * sqrt( 4 * V_REF / (DCHI2_REF * v) )
    return A_REF * np.sqrt(4 * V_REF / (DCHI2_REF * v_eff_array))

def main():
    # Volume range: 10 to 60 (h^-1 Gpc)^3
    # DESI Y1 approx 15-20? Y3 ~ 40? Y5 ~ 50-60?
    v_eff = np.linspace(10, 60, 100)
    
    a_excl_2sigma = get_exclusion_curve(v_eff)
    a_excl_3sigma = A_REF * np.sqrt(9 * V_REF / (DCHI2_REF * v_eff)) # Delta_chi2 = 9
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Exclusion Regions
    ax.fill_between(v_eff, a_excl_2sigma, 1.0, color='red', alpha=0.2, label=r'Excluded ($>2\sigma$)')
    ax.fill_between(v_eff, a_excl_3sigma, 1.0, color='red', alpha=0.2, hatch='//', label=r'Excluded ($>3\sigma$)')
    
    # Limits lines
    ax.plot(v_eff, a_excl_2sigma, color='red', lw=2, linestyle='--')
    ax.plot(v_eff, a_excl_3sigma, color='darkred', lw=2, linestyle='-')
    
    # Bias Band
    ax.axhspan(A_BIAS_BAND[0], A_BIAS_BAND[1], color='gray', alpha=0.3, label='Amplitude needed to bias BAO')
    ax.text(12, 0.015, "Bias Danger Zone", color='black', fontsize=10, fontweight='bold')
    
    # Reference Points
    ax.scatter([10], [0.02], color='black', marker='o', label=r'Toy Baseline ($V_{eff}=10$)')
    ax.scatter([38], [0.011], color='blue', marker='*', s=150, zorder=10, label=r'DESI Y3 Upper Bound ($V_{eff}\approx 38$)')
    
    # Styling
    ax.set_xlabel(r'Effective Volume $V_{eff}$ [$(h^{-1}{\rm Gpc})^3$]')
    ax.set_ylabel(r'DAO Contaminant Amplitude $A_{DAO}$')
    ax.set_ylim(0, 0.04)
    ax.set_xlim(10, 60)
    ax.grid(True, alpha=0.3)
    ax.set_title('DAO Falsifiability: Bias vs. Full-Shape Exclusion')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('docs/dao_exclusion_plot.svg')
    print("Plot saved to docs/dao_exclusion_plot.svg")

if __name__ == "__main__":
    main()
