# Datasets

All datasets are public molecular property benchmarks from MoleculeNet / DeepChem (and
mirrored by the Therapeutics Data Commons). Regression sets are shipped in this directory;
classification sets are downloaded on demand with `qspr-bench fetch <NAME>` (see
`qspr_bench/fetch.py` for the exact source URLs).

| Dataset | Task | Rows | SMILES col | Target col | Shipped? | Source |
|---|---|---|---|---|---|---|
| ESOL | regression (log aqueous solubility) | 1128 | `smiles` | `measured log solubility in mols per litre` | yes | MoleculeNet (Delaney 2004) |
| FreeSolv | regression (hydration free energy) | 642 | `Drug` | `Y` | yes | MoleculeNet / FreeSolv (Mobley) |
| Lipophilicity | regression (logD 7.4) | 4200 | `Drug` | `Y` | yes | MoleculeNet / ChEMBL |
| BBBP | classification (blood-brain barrier penetration) | 2050 (raw) | `smiles` | `p_np` | no (fetch) | DeepChem S3 mirror |
| BACE | classification (BACE-1 inhibition) | ~1513 | `mol` | `Class` | no (fetch) | DeepChem S3 mirror |

Additional fetchable sets defined in `fetch.py`: **HIV** (`HIV_active`) and **Tox21**
(single label `NR-AR` for v1; multi-label is out of scope).

## Provenance of fetched datasets

`qspr-bench fetch` downloads the canonical CSVs from the DeepChem S3 mirror
(`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/`) and validates, before
keeping a file, that it parses as CSV, has the expected SMILES/target columns, and that the
SMILES parse with RDKit (an HTML/XML/404 error body is rejected). Exact URLs are in the
`SOURCES` table in `qspr_bench/fetch.py`.

## Licensing / terms

These datasets are distributed by MoleculeNet and DeepChem for research use under their
respective open terms. The shipped CSVs are included here solely to make the regression
benchmark reproducible; cite the original dataset papers (ESOL: Delaney 2004; FreeSolv:
Mobley & Guthrie 2014; Lipophilicity: ChEMBL; BBBP: Martins et al. 2012; BACE: Subramanian
et al. 2016) when using them. Row counts are after RDKit SMILES validation and may differ
slightly from the raw upstream files.
