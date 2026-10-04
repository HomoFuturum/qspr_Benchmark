"""Opt-in downloader for classification datasets.

Classification datasets are not shipped in the repo. This module downloads the
canonical MoleculeNet CSVs from the DeepChem S3 mirror into data/, validating that
the response is a real CSV (not an HTML/XML error body) with the expected columns and
parseable SMILES before keeping it. Network access is required; this is never invoked
automatically by a benchmark run.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from .errors import ConfigError

RDLogger.DisableLog("rdApp.*")

# name -> (url, smiles_col, target_col)
SOURCES = {
    "BBBP": ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
             "smiles", "p_np"),
    "BACE": ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
             "mol", "Class"),
    "HIV": ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
            "smiles", "HIV_active"),
    "Tox21": ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
              "smiles", "NR-AR"),
}

_OUT_NAME = {"BBBP": "bbbp.csv", "BACE": "bace.csv", "HIV": "hiv.csv",
             "Tox21": "tox21.csv"}


def fetch(name, dest="data", force=False, timeout=60):
    """Download the canonical CSV for `name` into `dest`. Returns the written Path."""
    if name not in SOURCES:
        raise ConfigError(f"unknown dataset {name!r}; fetchable: {sorted(SOURCES)}")
    url, smi_col, tgt_col = SOURCES[name]
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / _OUT_NAME[name]
    if out.exists() and not force:
        raise ConfigError(f"{out} already exists; pass force=True to overwrite")

    print(f"fetching {name} from {url}\n  -> {out}")
    req = urllib.request.Request(url, headers={"User-Agent": "qspr_bench/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()

    # reject obvious error bodies (the data/ dir previously held 404/AccessDenied junk)
    head = raw[:256].lstrip().lower()
    if head.startswith(b"<") or head.startswith(b"404") or b"accessdenied" in head:
        raise ConfigError(f"{name}: server returned a non-CSV body (first bytes: "
                          f"{raw[:80]!r}); download rejected")

    compression = "gzip" if url.endswith(".gz") else "infer"
    try:
        df = pd.read_csv(io.BytesIO(raw), compression=compression)
    except Exception as e:
        raise ConfigError(f"{name}: downloaded content is not parseable CSV ({e})")

    for col in (smi_col, tgt_col):
        if col not in df.columns:
            raise ConfigError(
                f"{name}: expected column {col!r} not found in download "
                f"(got {list(df.columns)[:12]})"
            )

    # sanity: at least some SMILES parse
    sample = df[smi_col].dropna().head(20)
    ok = sum(1 for s in sample if isinstance(s, str) and Chem.MolFromSmiles(s) is not None)
    if ok == 0:
        raise ConfigError(f"{name}: no parseable SMILES in column {smi_col!r}; rejected")

    # write the (decompressed) CSV so the registry can read it directly
    df.to_csv(out, index=False)
    print(f"  ok: {len(df)} rows, columns include {smi_col!r} and {tgt_col!r}")
    return out
