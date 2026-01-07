"""
Data loading utilities for DESI chains and analysis data.
"""

from pathlib import Path
from typing import Optional, Dict
import urllib.request
import pandas as pd
import numpy as np


def download_file(url: str, output_path: Path, force: bool = False) -> None:
    """
    Download file from URL if it doesn't exist.

    Args:
        url: URL to download from
        output_path: Local path to save file
        force: Force re-download even if file exists
    """
    if output_path.exists() and not force and output_path.stat().st_size > 0:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        raise


def download_desi_chains(
    base_url: str,
    output_dir: Path,
    case_name: str = "BAO+CMBcomp",
    n_chains: int = 4,
) -> Dict[str, Path]:
    """
    Download DESI MCMC chains and input files.

    Args:
        base_url: Base URL for DESI data
        output_dir: Local directory for chains
        case_name: Name of the case/configuration
        n_chains: Number of chain files to download

    Returns:
        Dictionary mapping file types to local paths
    """
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # Download input YAML
    yaml_file = case_dir / "chain.input.yaml"
    download_file(f"{base_url}/chain.input.yaml", yaml_file)
    files['yaml'] = yaml_file

    # Download chain files
    chain_files = []
    for i in range(1, n_chains + 1):
        chain_file = case_dir / f"chain.{i}.txt"
        download_file(f"{base_url}/chain.{i}.txt", chain_file)
        chain_files.append(chain_file)
    files['chains'] = chain_files

    return files


def load_mcmc_chain(chain_path: Path) -> pd.DataFrame:
    """
    Load MCMC chain from file.

    Handles cobaya-style chain files with header comments.

    Args:
        chain_path: Path to chain file

    Returns:
        DataFrame with chain samples
    """
    if not chain_path.exists():
        raise FileNotFoundError(f"Chain file not found: {chain_path}")

    # Read header to get column names
    header = None
    with open(chain_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                header = line[1:].strip()
                break

    if header is None:
        raise ValueError(f"No header found in {chain_path}")

    cols = header.split()

    # Read data
    try:
        df = pd.read_csv(
            chain_path,
            delim_whitespace=True,
            comment="#",
            names=cols,
            dtype=np.float64,
        )
        return df
    except Exception as e:
        print(f"Error reading {chain_path}: {e}")
        raise
