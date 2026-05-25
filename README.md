# MoE-ODE PET Progression Model

Training code for a Mixture-of-Experts ODE used to model longitudinal tau /
amyloid PET trajectories. The dynamics combine a physical
diffusion-with-reaction term on the structural connectivity Laplacian (with an
optional learned per-ROI diffusion modulator) and a per-ROI local-clearance MLP
expert, gated by a small time/state-conditioned softmax. Subject visits are
aligned to a shared global trajectory via per-subject time shift `t0` and
optional time scaling `alpha`.

The release intentionally excludes raw data, trained checkpoints, generated
figures, experiment outputs, and paper-specific post-processing scripts.

## Repository Layout

```text
.
├── configs/              # Example YAML configs (tau, amyloid)
├── data/                 # Put local CSV inputs here; ignored by git
├── models/               # Model definitions
│   ├── moe_ode_tau.py    # MoE ODE (physical + local experts, gate)
│   └── local_clearance.py
├── utils/                # Shared utilities
├── data_loader.py        # SC / PET loading and preprocessing helpers
└── run_moe_tau_full.py   # Main training entry point
```

## Data Format

The training script expects two CSV inputs:

- **Structural connectivity matrix**: first column is the ROI name/index,
  remaining columns are ROI names in the same order.
- **PET table**: one row per subject visit, with `RID` or `PTID`, `VISCODE2`,
  and ROI columns ending in `_SUVR`. `VISCODE2` should use ADNI-style month
  labels such as `m0`, `m12`, `m24`.

Raw data are not included. Place local files under `data/` and edit the paths
in `configs/*.yaml`.

## Install

```bash
conda create -n pet-moe python=3.11
conda activate pet-moe
pip install -r requirements.txt
```

## Train

Tau:

```bash
python run_moe_tau_full.py --config configs/ours_tau.yaml
```

Amyloid (same entry point, different CSV):

```bash
python run_moe_tau_full.py --config configs/ours_amyloid.yaml
```

## Outputs

Training outputs are written to `outputs/` and preprocessing artifacts to
`artifacts/` by default. Both directories are ignored by git.
